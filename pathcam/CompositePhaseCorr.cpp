//
// Created by cooper maira on 9/29/25.
//
#include <pathCam.h>

namespace pathCam {
  static double quadSubpixel1D(float vm1, float v0, float vp1) {
    // offset in [-1, +1], positive means peak is to the + direction
    double denom = (vm1 - 2.0 * v0 + vp1);
    if (std::abs(denom) < 1e-12) return 0.0;
    return 0.5 * (vm1 - vp1) / denom;
  }


  static Rect centeredRect(int W, int H, int w, int h) {
    int x0 = (W - w) / 2;
    int y0 = (H - h) / 2;
    return cv::Rect(std::max(0, x0), std::max(0, y0),
                    std::min(w, W), std::min(h, H));
  }

  void shuffle_sift_data(SiftData &sd) {
    // 1. Create an index vector 0..numPts-1
    std::vector<int> enm(sd.numPts);
    std::iota(enm.begin(), enm.end(), 0);

    // 2. Shuffle the indices
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(enm.begin(), enm.end(), gen);

    // 3. Create a temporary copy buffer
    std::vector<SiftPoint> tmp(sd.numPts);

    // 4. Copy shuffled data into tmp
    for (int i = 0; i < sd.numPts; ++i)
      tmp[i] = sd.h_data[enm[i]]; // direct struct assignment, not memcpy

    // 5. Write back to original
    for (int i = 0; i < sd.numPts; ++i)
      sd.h_data[i] = tmp[i];

    // Optional: re-upload to GPU if device copy exists
    if (sd.d_data)
      cudaMemcpy(sd.d_data, sd.h_data, sd.numPts * sizeof(SiftPoint), cudaMemcpyHostToDevice);
  }

  cuda::GpuMat CompositeVoronoi::preprocess_GPU(const cuda::GpuMat &bgr_or_gray, cudaStream_t stream) {
    CV_Assert(!bgr_or_gray.empty());
    cuda::GpuMat gray, f32, gx, gy, mag, magNorm, wx, wy, win, winApplied;

    if (bgr_or_gray.channels() == 3) {
      cuda::cvtColor(bgr_or_gray, gray, COLOR_BGR2GRAY, 0);
    } else {
      gray = bgr_or_gray;
    }
    gray.convertTo(f32, CV_32F);

    // Sobel (use separable filter from cudafilters)
    auto sobelX = cuda::createSobelFilter(CV_32F, CV_32F, 1, 0, 3);
    auto sobelY = cuda::createSobelFilter(CV_32F, CV_32F, 0, 1, 3);
    sobelX->apply(f32, gx);
    sobelY->apply(f32, gy);

    cuda::magnitude(gx, gy, mag);

    // Normalize to [0,1] by dividing by max
    double minv, maxv;
    cuda::minMax(mag, &minv, &maxv);
    if (maxv > 0.0) cuda::multiply(mag, Scalar(1.0f / (float) maxv), magNorm, 1.0, -1);
    else mag.copyTo(magNorm);

    // Hann window: build 1D wins and apply outer-product
    ensure1DHann(magNorm.cols, magNorm.rows, wx, wy, stream);
    win.create(magNorm.size(), CV_32F);

    launch_apply_hann_2d(wx, wy, win, magNorm, stream);

    return win; // CV_32F 1ch
  }


  std::pair<Point2d, double> CompositeVoronoi::phase_correlate_GPU(const cuda::GpuMat &A32F, const cuda::GpuMat &B32F,
                                                                   cudaStream_t stream) {
    CV_Assert(A32F.size() == B32F.size() && A32F.type() == CV_32F && B32F.type() == CV_32F);
    Size sz = A32F.size();

    // 1) Forward FFTs (complex)
    cuda::GpuMat FA, FB;
    cuda::dft(A32F, FA, sz, 0); // CV_32FC2
    cuda::dft(B32F, FB, sz, 0); // CV_32FC2

    // 2) Cross-power spectrum on GPU
    cuda::GpuMat CPS(sz, CV_32FC2);
    launch_CPS(FA, FB, CPS, 1e-9, stream);

    // 3) Inverse FFT -> real correlation surface (still on GPU)
    cuda::GpuMat corr;
    cuda::dft(CPS, corr, sz, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT); // CV_32F

    // 4) Find global max on GPU (values on GPU, location returned as host Point)
    double peakVal = 0.0;
    Point peakLoc;
    cuda::minMaxLoc(corr, /*minVal*/nullptr, &peakVal, /*minLoc*/nullptr, &peakLoc, noArray());

    // 5) Turn peak location into wrapped shift (phase-corr convention)
    double dx = (peakLoc.x <= sz.width / 2) ? peakLoc.x : (peakLoc.x - sz.width);
    double dy = (peakLoc.y <= sz.height / 2) ? peakLoc.y : (peakLoc.y - sz.height);

    // 6) Optional: subpixel refinement via 3x3 patch around peak (download only 3x3 ROI)
    //    Clamp ROI at borders
    int x0 = std::max(1, peakLoc.x) - 1;
    int y0 = std::max(1, peakLoc.y) - 1;
    int x1 = std::min(sz.width - 2, peakLoc.x) + 1;
    int y1 = std::min(sz.height - 2, peakLoc.y) + 1;

    if (x1 - x0 + 1 == 3 && y1 - y0 + 1 == 3) {
      cuda::GpuMat roi = corr(Rect(x0, y0, 3, 3));
      Mat patch3x3;
      roi.download(patch3x3); // tiny (3x3x4B) transfer

      // rows: y-1, y, y+1 at the peak column; cols: x-1, x, x+1 at the peak row
      float cxm1 = patch3x3.at<float>(1, 0), cx0 = patch3x3.at<float>(1, 1), cxp1 = patch3x3.at<float>(1, 2);
      float cym1 = patch3x3.at<float>(0, 1), cy0 = patch3x3.at<float>(1, 1), cyp1 = patch3x3.at<float>(2, 1);

      double subx = quadSubpixel1D(cxm1, cx0, cxp1);
      double suby = quadSubpixel1D(cym1, cy0, cyp1);

      dx += subx;
      dy += suby;
    }

    // 7) A lightweight response: peak / sum(corr) (sum is a host scalar; small sync)
    Scalar s = cuda::sum(corr);
    double sumCorr = s[0] + 1e-12;
    double response = (sumCorr > 0.0) ? (peakVal / sumCorr) : 0.0;

    return {Point2d(dx, dy), response};
  }

  ScaleResult CompositeVoronoi::estimate_cale_discrete_GPU(const cuda::GpuMat &dLow_bgr, const cuda::GpuMat &dHigh_bgr,
                                                           const std::vector<double> &scales, cudaStream_t stream) {
    constexpr int K = 64; // common evaluation size
    ScaleResult best;

    if (dLow_bgr.empty() || dHigh_bgr.empty()) return best;


    // 1) Center crop from LOW (assumes image is big enough)
    if (dLow_bgr.cols < K || dLow_bgr.rows < K) return best;
    //Rect lowR = centeredRect(dLow_bgr.cols, dLow_bgr.rows, K, K);
    Size newsize(dLow_bgr.cols / 2, dLow_bgr.rows / 2);
    cuda::GpuMat dlowRS;
    cuda::resize(dLow_bgr, dlowRS, newsize);
    Rect lowR((dLow_bgr.cols - K) / 2, (dLow_bgr.rows - K) / 2, K, K);
    cuda::GpuMat lowK = dLow_bgr(lowR);

    Scalar lowMean, stddev;
    cuda::meanStdDev(lowK, lowMean, stddev);

    Mat temp;
    lowK.download(temp);
    imwrite("/media/max/Data/phase_corr_test/lowk.png", temp);
    // dLow_bgr.download(temp);
    // imwrite("/media/max/Data/phase_corr_test/dlow_bgr.png", temp);
    // dHigh_bgr.download(temp);
    // imwrite("/media/max/Data/phase_corr_test/dhigh_bgr.png", temp);


    // 2) For each scale hypothesis, extract (K*s)x(K*s) from HIGH, then downscale to 256x256
    for (double s: scales) {
      if (s <= 0.0) continue;
      int big = int(std::round(K * s));

      // Must be able to take a centered (big x big) from high
      if (dHigh_bgr.cols < big || dHigh_bgr.rows < big) continue;

      Rect highR = centeredRect(dHigh_bgr.cols, dHigh_bgr.rows, big, big);
      if (highR.width != big || highR.height != big) continue; // enforce exact size

      cuda::GpuMat highBig = dHigh_bgr(highR);

      // Downscale to KxK at low-mag pixel scale
      cuda::GpuMat highK;
      cuda::resize(highBig, highK, Size(K, K), 0, 0, INTER_AREA);

      // Scalar highMean;
      // cuda::meanStdDev(highK,highMean,stddev);
      // cuda::divide(highK,highMean,highK);
      // cuda::multiply(highK,lowMean,highK);
      // highK.download(temp);
      // imwrite("/media/max/Data/phase_corr_test/"+std::to_string(s)+".png",temp);

      cuda::GpuMat temp1;
      cuda::resize(dHigh_bgr, temp1, {dHigh_bgr.cols / int(2 * s), dHigh_bgr.rows / int(2 * s)});
      temp1.convertTo(temp1,CV_32F);
      lowK.convertTo(lowK,CV_32F);
      auto div = cuda::norm(lowK, NORM_L2);
      cuda::divide(lowK, Scalar(div), lowK);
      std::vector<double> dots;
      cuda::GpuMat templow;
      double maxval = 0, secondplace = 0;
      std::pair<int, int> loc;
      for (int y = temp1.rows / 2 - 150; y < temp1.rows / 2 + 150; ++y) {
        for (int x = temp1.cols / 2 - 150; x < temp1.cols / 2 + 150; ++x) {
          Rect roi(x, y, K, K);
          auto div1 = cuda::norm(temp1(roi), NORM_L2);
          cuda::divide(temp1(roi), Scalar(div1), templow);
          cuda::multiply(templow, lowK, templow);
          Scalar dot = cuda::sum(templow);
          dots.push_back(dot[0]);
          if (dot[0] > maxval) {
            secondplace = maxval;
            maxval = dot[0];
            loc = {x, y};
          }
        }
        if (y % 100 == 0) {
          std::cout << y << std::endl;
        }
      }
      Rect roi(loc.first, loc.second, K, K);
      Mat temp11;
      temp1(roi).download(temp11);
      imwrite("/media/max/Data/phase_corr_test/maxcrop.png", temp11);
      auto max_it = std::max_element(dots.begin(), dots.end());
      double max_val = (max_it != dots.end()) ? *max_it : 0.0;
      std::cout << "max: " << max_val << std::endl;
      double mean = std::accumulate(dots.begin(), dots.end(), 0.0) / double(dots.size());
      std::cout << "mean: " << mean << std::endl;

      // 3) Photometric preprocessing AFTER both are KxK (gray -> Sobel mag -> norm -> Hann)
      cuda::GpuMat lowProc = preprocess_GPU(lowK, stream);
      cuda::GpuMat highProc = preprocess_GPU(highK, stream);

      // 4) GPU phase correlation (FFT -> CPS -> iFFT), peak via cv::cuda::minMaxLoc
      auto [shift, response] = phase_correlate_GPU(lowProc, highProc, stream);

      double shiftMag = std::hypot(shift.x, shift.y);
      double diag = std::hypot(double(K), double(K));
      double shiftNormDiag = (diag > 0.0) ? (shiftMag / diag) : 0.0;

      if (!best.valid || response > best.response) {
        best.valid = true;
        best.scale = s;
        best.response = response;
        best.shift = shift;
        best.shiftNorm = shiftMag;
        best.shiftNormDiag = shiftNormDiag;
        best.cropSize = {K, K};
      }
    }

    return best;
  }

  void CompositeVoronoi::establish_scale_between_two_centered_Images(Image *img1, Image *img2, double &scale, Point2f &offset) {
    assert(img1->siftData.numPts > 0 && img2->siftData.numPts > 0);
    MatchSiftData(img1->siftData,img2->siftData);

    std::vector<float> homography(9);
    int numMatches;
    FindHomography(img1->siftData,homography.data(),&numMatches,10000,0.8,0.9,5);


  }



  void CompositeVoronoi::establish_scale_at_root(Image *_rootImg) {
    //get my sift data
    if (!_rootImg->siftInitialized) {
      if (!_rootImg->cudaBufferReady) {
        _rootImg->move_buffer_to_gpu(parent->compositorCudaDevice,true);
      }
      auto myGray = get_grayscale(_rootImg);
      CudaImage cImgGry;
      cImgGry.Allocate(image_size.width, image_size.height, myGray.step / sizeof(float), false,
                       reinterpret_cast<float *>(myGray.data), nullptr);

      InitSiftData(_rootImg->siftData, 100000, true, true);
      _rootImg->siftInitialized = true;

      catch_ExtractSift(_rootImg->siftData, cImgGry, 5, 0.0f, 0.4f, 0.1f, false);
    }

    //get their sift data
    if (!parent->lastViewedFrame->siftInitialized) {
      if (!parent->lastViewedFrame->cudaBufferReady) {
        parent->lastViewedFrame->move_buffer_to_gpu(parent->compositorCudaDevice,true);
      }
      auto theirGray = get_grayscale(parent->lastViewedFrame);
      CudaImage cImgGry;
      cImgGry.Allocate(image_size.width, image_size.height, theirGray.step / sizeof(float), false,
                       reinterpret_cast<float *>(theirGray.data), nullptr);

      InitSiftData(parent->lastViewedFrame->siftData, 100000, true, true);
      parent->lastViewedFrame->siftInitialized = true;

      catch_ExtractSift(parent->lastViewedFrame->siftData, cImgGry, 5, 0.0f, 0.4f, 0.1f, false);
    }

    MatchSiftData(_rootImg->siftData,parent->lastViewedFrame->siftData);
    std::vector<float> homography(9);
    int numMatches;
    bool validHomography = false;
    int count = 0;
    while (!validHomography && count < 20) {
      ++count;
      FindHomography(_rootImg->siftData, homography.data(), &numMatches, 10000, 0.8, 0.9, 5.0);

      auto matchedComp = parent->composites[parent->lastViewedFrame->regInfo->component_membership];
      for (auto scale : matchedComp->candidateScaleRatios) {
        scale = 1/scale;
        if (abs(scale - homography[0]) < 0.05 * scale && abs(scale - homography[4]) < 0.05 * scale) {
          validHomography = true;
        }
      }
      if (!validHomography) {
        shuffle_sift_data(_rootImg->siftData);
      }
    }

    if (!validHomography) {
      throw std::runtime_error("not sure what to do, matching failed");
    }
    float relativeScale = (homography[0] + homography[4]) / 2;

    auto queryComponent = parent->lastViewedFrame->regInfo->component_membership;
    Point2f theirAbC(parent->lastViewedFrame->regInfo->absoluteCoords.x, parent->lastViewedFrame->regInfo->absoluteCoords.y);
    Point2f pairwiseDistance = Point2f(homography[2] , homography[5]);
    Point2f queryAbC = pairwiseDistance + theirAbC;
    auto resultantPoint = parent->get_AbC_relative_from_relative(queryComponent, queryAbC, 0);

    double scale = relativeScale * parent->composites[queryComponent]->imagePyramid->scale;
    _rootImg->regInfo->rootHomographies.emplace_back(resultantPoint, scale);

    assert(scale > 0);
    parent->composites[componentIndex]->set_scale(scale);
    parent->composites[componentIndex]->set_offset(resultantPoint/scale);

  }


  ScaleResult CompositeVoronoi::estimate_scale_auto_GPU(const cuda::GpuMat &imgA, const cuda::GpuMat &imgB,
                                                        const std::vector<double> &scales,
                                                        cudaStream_t stream) {
  }
}
