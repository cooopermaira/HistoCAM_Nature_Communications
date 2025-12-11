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
    cv::Mat makeMotionKernel(int length, float angleDeg) {
        // Ensure length is at least 1 and odd
        length = std::max(1, length);
        if (length % 2 == 0) length += 1;

        int ksize = length; // you can pad larger if you want, but this is enough
        cv::Mat kernel = cv::Mat::zeros(ksize, ksize, CV_32F);

        // Draw a horizontal line (center row) of "length" ones
        int center = ksize / 2;
        int half = length / 2;
        for (int x = center - half + 2; x <= center + half - 2; ++x) {
            kernel.at<float>(center, x) = 1.0f;
        }
        kernel.at<float>(center, center - half) = kernel.at<float>(center, center + half) = 0.3f;
        kernel.at<float>(center, center - half + 1) = kernel.at<float>(center, center + half - 1) = 0.7f;

        // Normalize to sum = 1
        float sum1 = 0;
        for (int x = center - half; x <= center + half; ++x) {
            sum1 += kernel.at<float>(center, x);
        }
        kernel /= sum1;

        // Rotate the kernel by angleDeg around its center
        cv::Point2f rotCenter(ksize / 2.0f, ksize / 2.0f);
        cv::Mat rotMat = cv::getRotationMatrix2D(rotCenter, angleDeg, 1.0);

        cv::Mat rotated;
        cv::warpAffine(kernel, rotated, rotMat, kernel.size(),
                       cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

        // Re-normalize (rotation/interpolation slightly changes sum)
        double sum = cv::sum(rotated)[0];
        if (sum != 0.0)
            rotated /= static_cast<float>(sum);

        return rotated;
    }

    cv::Mat makeSmoothMotionKernel(int length, float angleDeg) {
        // Ensure odd length
        length = std::max(1, length);
        if (length % 2 == 0) length += 1;

        int ksize = length;
        cv::Mat kernel = cv::Mat::zeros(ksize, ksize, CV_32F);

        int center = ksize / 2;
        int half = length / 2;

        // --- Create a smooth 1D line (Gaussian) ---
        float sigma = length / 6.0f; // controls softness; adjust as you like
        auto gaussian = [&](int x) {
            float dx = float(x - center);
            return std::exp(-(dx * dx) / (2.0f * sigma * sigma));
        };

        // Fill the center row with Gaussian weights
        for (int x = center - half; x <= center + half; ++x)
            kernel.at<float>(center, x) = gaussian(x);

        // Normalize initial kernel
        float sum = cv::sum(kernel)[0];
        if (sum > 0) kernel /= sum;

        // --- Rotate kernel ---
        cv::Point2f rotCenter(ksize / 2.0f, ksize / 2.0f);
        cv::Mat rotMat = cv::getRotationMatrix2D(rotCenter, angleDeg, 1.0);

        cv::Mat rotated;
        cv::warpAffine(kernel, rotated, rotMat, kernel.size(),
                       cv::INTER_LINEAR,
                       cv::BORDER_CONSTANT, cv::Scalar(0));

        // Final normalization
        double finalSum = cv::sum(rotated)[0];
        if (finalSum != 0.0)
            rotated /= static_cast<float>(finalSum);

        return rotated;
    }

    void applyMotionBlur(const cv::Mat &src, cv::Mat &dst,
                         int length, float angleDeg) {
        cv::Mat kernel = makeMotionKernel(length, angleDeg);

        // Use BORDER_REPLICATE or REFLECT to avoid dark borders
        cv::filter2D(src, dst, -1, kernel, cv::Point(-1, -1),
                     0.0, cv::BORDER_REPLICATE);
    }

    cv::Mat extractRotatedROI(const cv::Mat &src,
                              const cv::Point2f &center,
                              const cv::Size &roiSize,
                              float angleDeg) {
        // 1. Build rotation matrix around the ROI center in source image
        cv::Mat M = cv::getRotationMatrix2D(center, angleDeg, 1.0);

        float dst_cx = roiSize.width * 0.5f;
        float dst_cy = roiSize.height * 0.5f;

        // M is 2x3: [ a11 a12 tx; a21 a22 ty ]
        M.at<double>(0, 2) += dst_cx - center.x;
        M.at<double>(1, 2) += dst_cy - center.y;

        // 3. Warp the original image into the ROI-sized output using this transform
        cv::Mat dst;
        cv::warpAffine(src, dst, M, roiSize,
                       cv::INTER_LINEAR,
                       cv::BORDER_REFLECT101); // or BORDER_CONSTANT, etc.

        return dst;
    }

    Mat buildRotatedSqrMask(int s, int flag) {
        Mat mask(s, s,CV_8UC1, Scalar(flag > 0 ? 0 : 255));
        std::vector<Point2i> corners;
        corners.emplace_back(s / 2, 0);
        corners.emplace_back(s, s / 2);
        corners.emplace_back(s / 2, s);
        corners.emplace_back(0, s / 2);

        fillConvexPoly(mask, corners, Scalar(flag > 0 ? 255 : 0));
        return mask;
    }

    float estimateSigma(Mat samples) {
        float sumsq = 0.0f;

        assert(samples.rows == samples.cols);
        for (int x = 0; x < samples.cols; ++x) {
            for (int y = 0; y < samples.cols; ++y) {
                float val = samples.at<float>(x, y);
                sumsq += val * val;
            }
        }

        return std::sqrt(sumsq / static_cast<float>(samples.rows * samples.rows));
    }

    float sampleHalfGaussian(float sigma) {
        // Thread-local RNG (safe in multithreaded code)
        static thread_local std::mt19937 rng{std::random_device{}()};
        static thread_local std::normal_distribution<double> dist(0.0, 1.0);

        // Sample a standard normal, scale it, take abs
        double x = dist(rng) * static_cast<double>(sigma);

        return static_cast<float>(x);
    }

    Mat get_noise(int size, float sigma) {
        Mat noise(size, size,CV_32FC1, Scalar(0.0));
        for (int x = 0; x < size; ++x) {
            for (int y = 0; y < size; ++y) {
                noise.at<float>(x, y) = sampleHalfGaussian(sigma);
            }
        }
        return noise;
    }

    Mat getContrib(Mat img, int flag) {
        Mat grayHost;
        auto mask = buildRotatedSqrMask(img.rows, flag);
        cuda::GpuMat maskGpu(mask.rows, mask.cols,CV_8UC1, mask.data);

        cuda::Stream s;

        img.convertTo(grayHost,CV_32F);
        cuda::GpuMat gray(grayHost.rows, grayHost.cols,CV_32FC1, grayHost.data);
        cuda::multiply(gray, Image::hannWindow, gray, 1, -1, s);

        //compute the log magnitude of the dft
        cuda::GpuMat planes[] = {gray, cuda::GpuMat(gray.rows, gray.cols,CV_32F, Scalar(0))};
        cuda::GpuMat complexI, mag;
        cuda::merge(planes, 2, complexI, s);
        cuda::dft(complexI, complexI, Size(gray.cols, gray.rows), 0, s);

        complexI.setTo(Scalar(0, 0), cuda::GpuMat(maskGpu.rows, maskGpu.cols,CV_8UC1, maskGpu.data), s);

        // //show dft after being zeroed out
        // cuda::split(complexI, planes, s);
        // cuda::magnitude(planes[0], planes[1], mag, s);
        // cuda::add(Scalar(1e-6), mag, mag, noArray(), -1, s);
        // cuda::log(mag, mag, s);
        // cuda::normalize(mag, mag, 0, 255, NORM_MINMAX,CV_8U, noArray(), s);
        // Mat temp1;
        // mag.download(temp1);

        cuda::dft(complexI, complexI, complexI.size(), cv::DFT_INVERSE | DFT_SCALE, s);
        cuda::GpuMat comps[2];
        cuda::split(complexI, comps, s);
        s.waitForCompletion();

        cuda::GpuMat filtered;
        comps[0].copyTo(filtered); // real part only
        filtered.download(grayHost);
        //grayHost.convertTo(grayHost,CV_8U);
        return grayHost;
    }

    void process_debayer(Image *image, bool saveCrop, Size cropSize, bool saveFull, float downsample, bool visualizeDft,
                         bool trainDft, Size trainDftSize, int trainSamples) {
        image->load_raw_from_disk();
        Size image_size(image->width, image->height);
        Mat img(image_size,CV_8U, image->get_Raw());
        cvtColor(img,img,COLOR_BayerBG2BGR);

        auto r = image->image_file.parent().parent();

        if (saveFull) {
            Mat full;
            if (downsample < 1.f) {
                resize(img,full,Size(image->width * downsample,image->height * downsample));
            }else {
                full = img.clone();
            }
            auto f = r;
            f.pushDirectory("png");
            f.setFileName(image->image_file.getFileName());
            f.setExtension("png");
            imwrite(f.toString(),full);
        }

        if (saveCrop) {
            Rect roi((image->width - cropSize.width)/2,
                (image->height - cropSize.height)/2,
                cropSize.width,cropSize.height);
            Mat crop = img(roi);

            auto v = r;
            v.pushDirectory("png_crop");
            v.setFileName(image->image_file.getFileName());
            v.setExtension("png");
            imwrite(v.toString(),crop);
        }

    }


    void DebayerRunnable::run() {
        image->index = image_index;
        process_debayer(image, false, {2000, 2000},
            false, 0.125, false, false, {}, 0);
    }


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
            Poco::Thread::sleep(1000 / 21);
        }
        parent->microscopeInput = false;
        std::cout << "disk images set " << std::endl;
    }
}
