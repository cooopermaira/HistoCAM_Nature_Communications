//
//  DiskStreamer.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include <utility>

#include "pathCam.h"
#include "Poco/DirectoryIterator.h"

using Poco::DirectoryIterator;

namespace pathCam {
  Mat ConvertBGR2Bayer(Mat BGRImage) {
    /*
    Assuming a Bayer filter that looks like this:

    # // 0  1  2  3  4  5
    /////////////////////
    0 // B  G  B  G  B  G
    1 // G  R  G  R  G  R
    2 // B  G  B  G  B  G
    3 // G  R  G  R  G  R
    4 // B  G  B  G  B  G
    5 // G  R  G  R  G  R

    */


    Mat BayerImage(BGRImage.rows, BGRImage.cols, CV_8UC1);

    int channel;

    for (int row = 0; row < BayerImage.rows; row++) {
      for (int col = 0; col < BayerImage.cols; col++) {
        if (row % 2 == 0) {
          //even columns and even rows = blue = channel:0
          //even columns and uneven rows = green = channel:1
          channel = (col % 2 == 0) ? 0 : 1;
        } else {
          //uneven columns and even rows = green = channel:1
          //uneven columns and uneven rows = red = channel:2
          channel = (col % 2 == 0) ? 1 : 2;
        }

        BayerImage.at<uchar>(row, col) = BGRImage.at<Vec3b>(row, col).val[channel];
      }
    }

    return BayerImage;
  }


  DiskReader::DiskReader(StreamCam *parent) : parent(parent) {
    parent->microscopeInput = true;
  }

  void DiskReader::run() {
    std::ifstream infile(parent->input_images.toString().c_str());
    std::string imageFile;
    unsigned long image_index = 0;
    while (infile >> imageFile) {
      Image *image = new Image(parent->image_width, parent->image_height, parent->scope_radius);
      image->set_disk_file(imageFile);
      parent->pass_image(image, image_index);
      image_index++;
    }
    parent->microscopeInput = false;
    std::cout << "disk images set " << std::endl;
  }


  void DebayerRunnable::run() {
    Poco::Path o = outfile;
    image->load_raw_from_disk();



    Size image_size(image->width , image->height);


    //get host buffer
    char* bufHost = new char[image->height * image->width];
    std::ifstream stream;
    stream.open(image->image_file.toString(), std::ios::binary);
    stream.read(bufHost, image->width * image->height);
    stream.close();

    char* ffraw = new char[image->height * image->width];
    std::ifstream streamFF;
    streamFF.open("/home/pathcam/pcamdata/cal/2x_cal.Raw",std::ios::binary);
    streamFF.read(ffraw,image->width * image->height);
    stream.close();
    Mat ff(image_size,CV_8U,ffraw);
    cvtColor(ff,ff,COLOR_BayerBG2BGR);
    ff.convertTo(ff,CV_32F);
    ff*= 1/170.f;

    //allocate a managed buffer
    char* buf;
    cudaMallocManaged(&buf,image_size.width * image_size.height * 3);
    int dev = 0;
    cudaGetDevice(&dev);
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeDevice;
    loc.id   = dev;
    cudaMemPrefetchAsync(buf, image_size.width * image_size.height * 3,loc, 0);

    //prepare gpumat to receive data into managed buffer
    cuda::GpuMat rcvgpu(image_size,CV_8UC3,buf);


    Mat raw(image_size,CV_8U,image->get_Raw());
    Mat dbd(image_size,CV_8UC3), rsz(image->height / 4,image->width / 4,CV_8UC3);
    Mat tcpa(image_size,CV_8UC3);
    Mat fcpa(image_size,CV_8UC4);
    Mat cvh(image_size,CV_32FC3);
    Mat alphac(image_size,CV_8U,Scalar(255));

    double pt1 = 0,pt2 = 0,pt3 = 0;

int roiSize = 256;
    Mat dbdHost(Size(roiSize,roiSize),CV_8UC3);
    cuda::GpuMat dbdcuda({roiSize,roiSize},CV_8UC3,dbdHost.data),tcpaG({roiSize,roiSize},CV_8UC3);

    cuda::GpuMat cvhG({roiSize,roiSize},CV_32FC3);
    cuda::GpuMat ffGpu(image_size,CV_32FC3,ff.data);







    char* buf1;
    //buf1 = new char[roiSize * roiSize * 4];
    cudaMallocManaged(&buf1,roiSize * roiSize * 4);
    cuda::GpuMat rszcuda({roiSize,roiSize},CV_8UC4,buf1);
    cuda::GpuMat rawcuda(image_size,CV_8U,image->get_Raw());
    Mat rawMat(image_size,CV_8U,image->get_Raw());

    Rect roi(1024*2,1024*2,roiSize,roiSize);
    Mat alphCRoi = alphac(roi).clone();
    float iter = 1000;
    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i<int(iter);++i) {

      cvtColor(rawMat(roi),dbdHost,COLOR_BayerBG2BGR);

      auto c = std::chrono::high_resolution_clock::now();
      dbdcuda.convertTo(cvhG,CV_32F);
      pt3 += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - c).count();

      c = std::chrono::high_resolution_clock::now();
      cuda::divide(cvhG,ffGpu(roi),cvhG);
      cuda::pow(cvhG,1.1,cvhG);
      pt1 += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - c).count();

      c = std::chrono::high_resolution_clock::now();
      cvhG.convertTo(tcpaG,CV_8UC3);
      pt3 += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - c).count();

      std::vector<cuda::GpuMat> chan;


      c = std::chrono::high_resolution_clock::now();
      cuda::split(tcpaG,chan);
      chan.push_back(cuda::GpuMat({roiSize,roiSize},CV_8UC1,alphCRoi.data));
      cuda::merge(chan,rszcuda);
      pt2 += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - c).count();



      auto val = rszcuda.data[0];
    }
    std::cout<<"cuda converting: "<<pt3/iter<<std::endl;
    std::cout<<"cuda math: "<<pt1/iter<<std::endl;
    std::cout<<"cuda alpha: "<<pt2/iter<<std::endl;
    std::cout<<std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start1).count() / iter<<std::endl;

    pt1 = 0;
    pt2 = 0;
    pt3 = 0;

    dbdHost = Mat(image_size,CV_8UC3);
    dbdcuda = cuda::GpuMat(image_size,CV_8UC3,dbdHost.data);
    cvhG = cuda::GpuMat(image_size,CV_32FC3);
    ffGpu = cuda::GpuMat(image_size,CV_32FC3);
    tcpaG = cuda::GpuMat(image_size,CV_8UC3);
    cudaFree(buf1);
    buf1 = new char[image->width * image->height * 4];
    rszcuda = cuda::GpuMat(image_size,CV_8UC4,buf1);

    start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i<int(iter);++i) {

      cvtColor(raw,dbdHost,COLOR_BayerBG2BGR);

      auto c = std::chrono::high_resolution_clock::now();
      dbdcuda.convertTo(cvhG,CV_32F);
      pt3 += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - c).count();

      c = std::chrono::high_resolution_clock::now();
      cuda::divide(cvhG,ffGpu,cvhG);
      cuda::pow(cvhG,1.1,cvhG);
      pt1 += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - c).count();

      c = std::chrono::high_resolution_clock::now();
      cvhG.convertTo(tcpaG,CV_8UC3);
      pt3 += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - c).count();

      std::vector<cuda::GpuMat> chan;


      c = std::chrono::high_resolution_clock::now();
      cuda::split(tcpaG,chan);
      chan.push_back(cuda::GpuMat(image_size,CV_8UC1,alphac.data));
      cuda::merge(chan,rszcuda);
      pt2 += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - c).count();



      auto val = rszcuda.data[0];
    }
    std::cout<<"cuda converting: "<<pt3/iter<<std::endl;
    std::cout<<"cuda math: "<<pt1/iter<<std::endl;
    std::cout<<"cuda alpha: "<<pt2/iter<<std::endl;
    std::cout<<std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start1).count() / iter<<std::endl;




    start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i<100;++i) {
      cvtColor(raw,dbd,COLOR_BayerBG2BGR);
      auto c = std::chrono::high_resolution_clock::now();
      dbd.convertTo(cvh,CV_32FC3);
      pt3 += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - c).count();

      c = std::chrono::high_resolution_clock::now();
      divide(cvh,ff,cvh);
      pow(cvh,1.1,cvh);
      pt1 += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - c).count();

      c = std::chrono::high_resolution_clock::now();
      cvh.convertTo(tcpa,CV_8UC3);
      pt3 += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - c).count();

      std::vector<Mat> chan;

      c = std::chrono::high_resolution_clock::now();
      split(tcpa,chan);
      chan.push_back(alphac);
      merge(chan,fcpa);
      pt2 += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - c).count();


      auto val = fcpa.data[0];
    }
    std::cout<<"converting: "<<pt3/100<<std::endl;
    std::cout<<"math: "<<pt1/100<<std::endl;
    std::cout<<"alpha: "<<pt2/100.f<<std::endl;
    std::cout<<std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start1).count()<<std::endl;
//100 times, wrap the host buffer in a gpu mat, debayer to managed buffer, verify cpu can access data
    return;
    auto start = std::chrono::high_resolution_clock::now();
    double accCvtMs = 0, accMatWrapMs = 0, accHostReadMs = 0;

    for (int i = 0; i < 1000; ++i) {
      auto t_start_wrap = std::chrono::high_resolution_clock::now();
      cv::cuda::GpuMat readMat(image_size, CV_8U, bufHost);  // just header
      auto t_end_wrap = std::chrono::high_resolution_clock::now();
      accMatWrapMs += std::chrono::duration<double, std::milli>(t_end_wrap - t_start_wrap).count();

      auto t_start_cvt = std::chrono::high_resolution_clock::now();
      cv::cuda::cvtColor(readMat, rcvgpu, COLOR_BayerBG2BGR);
      // force GPU to finish so the timing is meaningful
      //cudaDeviceSynchronize();
      auto t_end_cvt = std::chrono::high_resolution_clock::now();
      accCvtMs += std::chrono::duration<double, std::milli>(t_end_cvt - t_start_cvt).count();

      auto t_start_host = std::chrono::high_resolution_clock::now();
      cv::Mat hostm(image_size, CV_8UC3, buf); // wrapping managed mem
      auto val = hostm.data[0];                // host read
      (void)val;
      auto t_end_host = std::chrono::high_resolution_clock::now();
      accHostReadMs += std::chrono::duration<double, std::milli>(t_end_host - t_start_host).count();
    }

    std::cout << "wrap GpuMat header: " << accMatWrapMs / 1000.0 << " ms/iter\n";
    std::cout << "cuda::cvtColor:     " << accCvtMs     / 1000.0 << " ms/iter\n";
    std::cout << "host read (UM):     " << accHostReadMs/ 1000.0 << " ms/iter\n";
    auto t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();

    //prepare a gpumat to receive data from cvtColor command, prepare mat for .download()
    cuda::GpuMat dbr(image_size.height,image_size.width,CV_8UC3);
    Mat rcvhost(image_size,CV_8UC3);

    //100 times, wrap host buffer in gpu mat, debayer into previously gpu allcated gpumat, download to host mat, verify cpu can access data
    start = std::chrono::high_resolution_clock::now();
    double accCvt2Ms = 0, accDownloadMs = 0;
    accHostReadMs = 0;
    for (int i = 0; i < 1000; ++i) {
      cv::cuda::GpuMat readMat(image_size, CV_8U, bufHost);

      auto t_start_cvt = std::chrono::high_resolution_clock::now();
      cv::cuda::cvtColor(readMat, dbr, COLOR_BayerBG2BGR);
      //cudaDeviceSynchronize();
      auto t_end_cvt = std::chrono::high_resolution_clock::now();
      accCvt2Ms += std::chrono::duration<double, std::milli>(t_end_cvt - t_start_cvt).count();

      auto t_start_dl = std::chrono::high_resolution_clock::now();
      dbr.download(rcvhost);
      //cudaDeviceSynchronize();
      auto t_end_dl = std::chrono::high_resolution_clock::now();
      accDownloadMs += std::chrono::duration<double, std::milli>(t_end_dl - t_start_dl).count();

      auto val = rcvhost.data[0];
      (void)val;
    }

    std::cout << "cvtColor (2): " << accCvt2Ms    / 1000.0 << " ms/iter\n";
    std::cout << "download():   " << accDownloadMs/ 1000.0 << " ms/iter\n";
    auto t2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();

    //cuda::resize(greyRoi,greyRoi,Size(greyRoi.cols / 4, greyRoi.rows / 4));

    auto r = outfile;
    // r.setFileName(image->get_ImageFile().getBaseName());
    // r.setExtension("png");
    // Rect roi(readMat.cols/2 - 1000,readMat.rows/2 - 1000,2000,2000);
    // Mat temp;
    // //Mat temp(dbr.rows, dbr.cols, CV_8UC3,dbr.datastart,dbr.step); //operations on this object segfault
    // readMat(roi).download(temp); //this is fine

    // imwrite(r.toString(), temp);
    // return;
    //cvtColor(readMat, temp, COLOR_BayerBG2GRAY);

    // Rect fullRoi(image_size.width/2 - 200,image_size.height/2-200,400,400);
    // Mat fullCrop = readMat(fullRoi);
    // auto val = Image::compute_sharpness(fullCrop);
    // double fullVal = min(val.x,val.y);
    cuda::GpuMat readMat(image_size, CV_8U, image->get_Raw());

    // Point2i cropSize(800,800);
    // Rect halfRoi(image_size.width/2-cropSize.x,image_size.height/2 - cropSize.y,cropSize.x * 2,cropSize.y*2);
    // Mat halfCrop = temp(halfRoi);
    // Mat halfTemp;
    // resize(halfCrop,halfTemp,Size(halfCrop.cols/2,halfCrop.rows/2));
    //
    // auto val = Image::compute_sharpness(halfTemp);
    // double halfVal = min(val.x,val.y);
    //
    // blur->at(sort_order) = halfVal;
    // names->at(sort_order) = image->get_ImageFile().getFileName();
    //return;




    bool convertAndSave = true;

    if (!image->in_memory()) {
      std::cout << "Issue loading image.\n";
      return;
    }
    // auto val = min(image->sharpness.x,image->sharpness.y);
    //     blur->at(sort_order) = val;
    //     names->at(sort_order) = image->get_ImageFile().getFileName();


    if (convertAndSave) {


      //
      // Size image_size(image->width, image->height);
      Mat image_Mat;
      // Mat readMat = Mat(image_size, CV_8U, image->get_Raw(), Mat::AUTO_STEP);
      // cvtColor(readMat, readMat, COLOR_BayerBG2BGR);

      cuda::resize(readMat,readMat,Size(image->width / 4, image->height / 4)); //resize if you want by changing it here
      auto r = outfile;

      //Rect zoomCrop(image_size.width/2-1000,image_size.height/2-1000,2000,2000);
      //Mat saveMat = readMat(zoomCrop)
      r.setFileName(image->get_ImageFile().getBaseName());
      r.setExtension("png");
      Mat temp(readMat.rows, readMat.cols, CV_8UC3,readMat.data,readMat.step);
      imwrite(r.toString(), temp);



      // int k = 0;
    //   readMat.convertTo(readMat,CV_32FC3);
    //
    //   for (int i = 0; i < ffs.size(); ++i) {
    //     Mat ff;
    //     std::ifstream stream;
    //     stream.open(ffs[i], std::ios::binary);
    //     {
    //       char *raw_buffer = new char[6464 * 4852];
    //       stream.read(raw_buffer, 6464 * 4852);
    //       stream.close();
    //       ff = Mat(Size(6464, 4852), CV_8U, raw_buffer, Mat::AUTO_STEP);
    //     }
    //     Mat gray;
    //     cvtColor(ff,gray,COLOR_BayerBG2GRAY);
    //     resize(gray,gray,Size(image->width / 4, image->height / 4));
    //     cvtColor(ff, ff, COLOR_BayerBG2BGR);
    //     imwrite("/media/max/Data/2_20/wrong_flatfield_test/png2/ff_"+std::to_string(i)+".png",gray);
    //     ff.convertTo(ff, CV_32F);
    //     ff *= 1 / 170.0;
    //     divide(readMat, ff, image_Mat, 1, CV_32F);
    //
    //     //cv::pow(image_Mat, 1.1, image_Mat);
    //
    //     image_Mat.convertTo(image_Mat, CV_8UC3);
    //
    //
    //     resize(image_Mat, image_Mat, Size(image->width / 4, image->height / 4));
    //     //
    //     //      imwrite("/Users/coopermaira/Desktop/ff.png", flatfield);
    //     //      imwrite("/Users/coopermaira/Desktop/pre_ff.png",image_Mat);
    //
    //
    //     //divide(readMat, flatfield, image_Mat, 1, CV_32F);
    //
    //     //cv::pow(image_Mat, 1.1, image_Mat);
    //
    //     //image_Mat.convertTo(image_Mat, CV_8UC3);
    //
    //     //add subdir for png
    //     auto r = outfile;
    //
    //     r.setFileName(image->get_ImageFile().getBaseName()+"_"+std::to_string(i));
    //     r.setExtension("png");
    //     imwrite(r.toString(), image_Mat);
    //   }
    //   //
    //   // cvtColor(image_Mat,image_Mat, COLOR_BayerBG2RGB);
    //   //        image_Mat = ConvertBGR2Bayer(image_Mat);
    //   //
    //   //        //save .Raw
    //   //        auto name = std::stoi(image->get_ImageFile().getBaseName());
    //   //        name += 250;
    //   //
    //   //
    //   //        r.setFileName(std::to_string(name));
    //   //        r.setExtension("Raw");
    //   //        std::fstream file;
    //   //        file = std::fstream(r.toString(), std::ios::out | std::ios::binary);
    //   //        if (file.fail()) {
    //   //          throw new std::exception;
    //   //        }
    //   //        file.write(reinterpret_cast<const char *>(image_Mat.data), image->width * image->height);
    //
    //   // } catch (cv::Exception &e) {
    //   //   int k = 0;
    //   // }
     }
    image->free_memory_RAW();
    int k = 0;
  }
}
