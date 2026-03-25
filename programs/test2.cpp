//
// Created by cooper maira on 3/23/26.
//

#include <fstream>
#include <opencv2/imgproc.hpp>

#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
int width = 6464, height = 4852;
using namespace cv;

struct PhaseScaleResult {
  double scale;
  cv::Point2d shift;
  double response;
  bool valid;
};

Mat load_raw(std::string path) {
  char *buf = (char *) malloc(height * width);
  std::ifstream stream;
  stream.open(path, std::ios::binary);
  stream.read(buf, width * height);
  stream.close();
  Mat raw(height, width,CV_8UC1, buf);
  // Rect roi(3000, 2000, 500, 500);
  // auto v = raw(roi);
  // cvtColor(raw, raw, COLOR_BayerBG2GRAY);
  return raw;
}


struct HomographyResultM
{
    cv::Mat H;
    bool valid = false;
    int inliers = 0;
    float scaleA = 1.0f;
    float scaleB = 1.0f;
    double avgReprojErr = 0.0;
};

inline HomographyResultM findHomographyAKAZE_allScalePairs(
    const cv::Mat& imageA,
    const cv::Mat& imageB)
{
    HomographyResultM best;

    if (imageA.empty() || imageB.empty())
        return best;

    // const std::vector<float> scales = {1.0f, 0.5f, 0.1f};
    const std::vector<float> scales = {0.25f,0.1f};
    // Geometric sanity thresholds for "almost no rotation or shearing"
    const double maxAnisotropyFrac = 0.08;   // |h00-h11| / avg(h00,h11)
    const double maxOffDiag = 0.08;          // small rotation/shear terms
    const double minDiag = 0.5;              // reject collapses / flips
    const double maxDiag = 2.0;              // reject wild scale jumps
    const int minGoodMatches = 4;
    const int minInliers = 4;
    const double maxAvgReprojErr = 3.0;

    auto akaze = cv::AKAZE::create();
    cv::BFMatcher matcher(cv::NORM_HAMMING);

    struct Features
    {
        float scale = 1.0f;
        cv::Mat image;
        std::vector<cv::KeyPoint> kp;
        cv::Mat desc;
    };

    std::vector<Features> featsA, featsB;
    featsA.reserve(scales.size());
    featsB.reserve(scales.size());

    auto buildFeatures = [&](const cv::Mat& src, float s) -> Features
    {
        Features f;
        f.scale = s;
        cv::resize(src, f.image, cv::Size(), s, s, cv::INTER_AREA);
        if (!f.image.empty())
            akaze->detectAndCompute(f.image, cv::noArray(), f.kp, f.desc);
        return f;
    };

    for (float s : scales)
    {
        featsA.push_back(buildFeatures(imageA, s));
        featsB.push_back(buildFeatures(imageB, s));
    }

    for (const auto& fa : featsA)
    {
        if (fa.desc.empty() /*|| fa.image.cols < 40 || fa.image.rows < 40*/)
            continue;

        for (const auto& fb : featsB)
        {
            // if (fb.desc.empty()/* || fb.image.cols < 40 || fb.image.rows < 40*/)
            //     continue;

            std::vector<std::vector<cv::DMatch>> knn;
            matcher.knnMatch(fa.desc, fb.desc, knn, 2);

            std::vector<cv::DMatch> good;
            good.reserve(knn.size());
            for (const auto& m : knn)
            {
                if (m.size() < 2)
                    continue;
                if (m[0].distance < 0.75f * m[1].distance)
                    good.push_back(m[0]);
            }

            if ((int)good.size() < minGoodMatches)
                continue;

            std::vector<cv::Point2f> ptsA, ptsB;
            ptsA.reserve(good.size());
            ptsB.reserve(good.size());

            for (const auto& m : good)
            {
                ptsA.push_back(fa.kp[m.queryIdx].pt);
                ptsB.push_back(fb.kp[m.trainIdx].pt);
            }

            std::vector<unsigned char> mask;
            cv::Mat Hscaled = cv::findHomography(ptsA, ptsB, cv::RANSAC, 3.0, mask);
            if (Hscaled.empty())
                continue;

            int inliers = std::count(mask.begin(), mask.end(), 1);
            if (inliers < minInliers)
                continue;

            // Average reprojection error in the scaled coordinate system
            double totalErr = 0.0;
            int errCount = 0;
            for (size_t i = 0; i < ptsA.size(); ++i)
            {
                if (!mask[i])
                    continue;

                const cv::Point2f& p = ptsA[i];
                const cv::Point2f& q = ptsB[i];

                cv::Mat hp = Hscaled * (cv::Mat_<double>(3,1) << p.x, p.y, 1.0);
                double w = hp.at<double>(2,0);
                if (std::abs(w) < 1e-12)
                    continue;

                cv::Point2f proj(
                    static_cast<float>(hp.at<double>(0,0) / w),
                    static_cast<float>(hp.at<double>(1,0) / w)
                );

                totalErr += cv::norm(proj - q);
                errCount++;
            }

            if (errCount == 0)
                continue;

            double avgErr = totalErr / errCount;
            if (avgErr > maxAvgReprojErr)
                continue;

            // Convert scaled-image homography back to full-resolution coordinates:
            // xB_scaled = Hscaled * xA_scaled
            // xA_scaled = SA * xA_full,  SA = diag(scaleA, scaleA, 1)
            // xB_scaled = SB * xB_full,  SB = diag(scaleB, scaleB, 1)
            // => xB_full = SB^{-1} * Hscaled * SA * xA_full
            cv::Mat SA = (cv::Mat_<double>(3,3) <<
                fa.scale, 0, 0,
                0, fa.scale, 0,
                0, 0, 1.0);

            cv::Mat SB_inv = (cv::Mat_<double>(3,3) <<
                1.0 / fb.scale, 0, 0,
                0, 1.0 / fb.scale, 0,
                0, 0, 1.0);

            cv::Mat Hfull = SB_inv * Hscaled * SA;

            // Normalize for inspection / consistency
            double h33 = Hfull.at<double>(2,2);
            if (std::abs(h33) < 1e-12)
                continue;
            Hfull /= h33;

            // Geometric sanity checks in full-resolution coordinates
            double h00 = Hfull.at<double>(0,0);
            double h01 = Hfull.at<double>(0,1);
            double h10 = Hfull.at<double>(1,0);
            double h11 = Hfull.at<double>(1,1);
            double avgDiag = 0.5 * (std::abs(h00) + std::abs(h11));

            if (avgDiag < 1e-3)
                continue;

            double anisotropyFrac = std::abs(h00 - h11) / avgDiag;

            if (anisotropyFrac > maxAnisotropyFrac)
                continue;

            if (std::abs(h01) > maxOffDiag || std::abs(h10) > maxOffDiag)
                continue;

            // Also reject strong perspective since your motion should be close to affine
            if (std::abs(Hfull.at<double>(2,0)) > 1e-3 ||
                std::abs(Hfull.at<double>(2,1)) > 1e-3)
                continue;

            if (!best.valid || inliers > best.inliers)
            {
                best.H = Hfull;
                best.valid = true;
                best.inliers = inliers;
                best.scaleA = fa.scale;
                best.scaleB = fb.scale;
                best.avgReprojErr = avgErr;
            }
        }
    }

    return best;
}

int main(int argc, char **argv) {
    auto start = std::chrono::high_resolution_clock::now();

  auto root = load_raw("/home/cm/Documents/data/blur_test/raw/574.Raw");
  auto target = load_raw("/home/cm/Documents/data/blur_test/raw/573.Raw");


  auto res = findHomographyAKAZE_allScalePairs(root, target);


  if (res.valid) {
    // std::cout << "Inliers: " << res.numInliers << std::endl;
    std::cout << "H:\n" << res.H << std::endl;
    std::cout << "scale A " << res.scaleA <<" scale B "<<res.scaleB << std::endl;
      std::cout << "inliers "<<res.inliers << std::endl;

  }

    root = load_raw("/home/cm/Documents/data/phase_corr_testing/test_images_raw/breast/B1_2x.raw");
    target = load_raw("/home/cm/Documents/data/phase_corr_testing/test_images_raw/breast/B1_20x.raw");

  auto res1 = findHomographyAKAZE_allScalePairs(target, root);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - start).count();
  if (res1.valid) {
    // std::cout << "Inliers: " << res.numInliers << std::endl;
    std::cout << "H:\n" << res1.H << std::endl;
    std::cout << "scale A " << res1.scaleA <<" scale B "<<res1.scaleB << std::endl;
      std::cout << "inliers "<<res1.inliers << std::endl;



  }
    std::cout << duration << std::endl;
  return 0;
};
