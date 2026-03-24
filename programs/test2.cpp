//
// Created by cooper maira on 3/23/26.
//

#include <fstream>
#include <opencv2/imgproc.hpp>

#include "opencv2/core.hpp"
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
  Rect roi(3000, 2000, 500, 500);
  auto v = raw(roi);
  cvtColor(raw, raw, COLOR_BayerBG2GRAY);
  return raw;
}

PhaseScaleResult estimate_scale_phase(
  const cv::Mat &root,
  const cv::Mat &target,
  const std::vector<double> &candidateScales) {
  PhaseScaleResult best{0, {}, -1.0, false};

  cv::Mat root32f, target32f;
  root.convertTo(root32f, CV_32F);
  target.convertTo(target32f, CV_32F);

  // downsample for speed
  const int maxDim = 512;
  auto downscale = [&](const cv::Mat &in) {
    double scale = std::min(1.0, maxDim / double(std::max(in.cols, in.rows)));
    cv::Mat out;
    cv::resize(in, out, {}, scale, scale, cv::INTER_AREA);
    return out;
  };

  root32f = downscale(root32f);
  target32f = downscale(target32f);

  // normalize
  auto normalize = [](cv::Mat &m) {
    cv::Scalar mean, stddev;
    cv::meanStdDev(m, mean, stddev);
    m = (m - mean[0]) / (stddev[0] + 1e-6);
  };

  normalize(root32f);
  normalize(target32f);

  // windowing (critical)
  cv::Mat hann;
  cv::createHanningWindow(hann, root32f.size(), CV_32F);

  root32f = root32f.mul(hann);

  std::vector<double> offsets = {0.95, 0.975, 1.0, 1.025, 1.05};

  double bestResponse = -1;
  cv::Point2d bestShift;
  double bestScale = 0;

  int searchRadius = 300;
  int step = 64;

  for (double scale: candidateScales) {
    cv::Mat scaled;
    cv::resize(target, scaled, {}, scale, scale, cv::INTER_LINEAR);

    for (int dx = -searchRadius; dx <= searchRadius; dx += step) {
      for (int dy = -searchRadius; dy <= searchRadius; dy += step) {
        int x = (scaled.cols - root.cols) / 2 + dx;
        int y = (scaled.rows - root.rows) / 2 + dy;

        if (x < 0 || y < 0 ||
            x + root.cols > scaled.cols ||
            y + root.rows > scaled.rows)
          continue;

        cv::Mat roi = scaled(cv::Rect(x, y, root.cols, root.rows));

        double response;
        cv::Point2d shift = cv::phaseCorrelate(root, roi, cv::noArray(), &response);

        if (response > bestResponse) {
          bestResponse = response;
          bestShift = shift + cv::Point2d(dx, dy);
          bestScale = scale;
        }
      }
    }
  }

  return best;
}

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

struct HomographyResult {
  cv::Mat H; // 3x3 homography
  std::vector<char> inliers; // mask from RANSAC
  int numInliers = 0;
  bool valid = false;
  float scale;
};

struct HomographyResultM {
  cv::Mat H;
  float scale = 1.0f;
  int inliers = 0;
  bool valid = false;
};
/*
inline HomographyResultM findHomographyAKAZE_multiscale(
  const cv::Mat &root,
  const cv::Mat &target,
  const std::vector<float> &scales) {
  float downscale = 0.5;

  HomographyResultM best;

  if (scales.empty())
    return best;

  float maxScale = *std::max_element(scales.begin(), scales.end());

  std::vector<float> normScales;
  normScales.reserve(scales.size());
  for (float s: scales)
    normScales.push_back(s / maxScale);

  cv::Mat img1_scaled;
  cv::resize(root, img1_scaled,
             cv::Size(), downscale / maxScale, downscale / maxScale,
             cv::INTER_AREA);

  auto akaze = cv::AKAZE::create();

  std::vector<cv::KeyPoint> kp1;
  cv::Mat desc1;
  akaze->detectAndCompute(img1_scaled, cv::noArray(), kp1, desc1);

  if (desc1.empty())
    return best;

  cv::BFMatcher matcher(cv::NORM_HAMMING);

  for (int i = 0; i < normScales.size(); i++) {
    float s = normScales[i];

    cv::Mat img2_scaled;
    cv::resize(target, img2_scaled,
               cv::Size(), downscale * s, downscale * s,
               cv::INTER_AREA);


    std::vector<cv::KeyPoint> kp2;
    cv::Mat desc2;
    akaze->detectAndCompute(img2_scaled, cv::noArray(), kp2, desc2);

    if (desc2.empty()) {
      continue;
    }

    std::vector<std::vector<cv::DMatch> > knn;
    matcher.knnMatch(desc1, desc2, knn, 2);

    std::vector<cv::DMatch> good;
    for (auto &m: knn) {
      if (m.size() < 2) {
        continue;
      }
      if (m[0].distance < 0.75f * m[1].distance)
        good.push_back(m[0]);
    }

    if (good.size() < 20) // raised from 10
      continue;

    std::vector<cv::Point2f> pts1, pts2;
    pts1.reserve(good.size());
    pts2.reserve(good.size());

    for (auto &m: good) {
      pts1.push_back(kp1[m.queryIdx].pt);
      pts2.push_back(kp2[m.trainIdx].pt);
    }

    std::vector<char> mask;
    cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, 3.0, mask);

    if (H.empty())
      continue;

    int inliers = std::count(mask.begin(), mask.end(), 1);

    // --- NEW: minimum inliers ---
    if (inliers < 30)
      continue;

    // --- NEW: inlier ratio ---
    float inlier_ratio = float(inliers) / float(good.size());
    if (inlier_ratio < 0.3f)
      continue;

    // --- NEW: average reprojection error ---
    double total_err = 0.0;
    int count = 0;

    for (int j = 0; j < pts1.size(); j++) {
      if (!mask[j]) {
        continue;
      }

      const cv::Point2f &p = pts1[j];
      const cv::Point2f &q = pts2[j];

      cv::Mat hp = H * (cv::Mat_<double>(3, 1) << p.x, p.y, 1.0);
      double w = hp.at<double>(2);

      if (std::abs(w) < 1e-8) {
        continue;
      }

      cv::Point2f p_proj(
        hp.at<double>(0) / w,
        hp.at<double>(1) / w
      );

      total_err += cv::norm(p_proj - q);
      count++;
    }

    if (count == 0) {
      continue;
    }

    double avg_err = total_err / count;

    if (avg_err > 2.0) {
      continue;
    }

    if (inliers > best.inliers) {
      float origScale = scales[i];

      cv::Mat S_root = (cv::Mat_<double>(3, 3) <<
                        1.0 / maxScale, 0, 0,
                        0, 1.0 / maxScale, 0,
                        0, 0, 1);

      cv::Mat S_target = (cv::Mat_<double>(3, 3) <<
                          origScale / maxScale, 0, 0,
                          0, origScale / maxScale, 0,
                          0, 0, 1);

      cv::Mat S_target_inv = (cv::Mat_<double>(3, 3) <<
                              maxScale / origScale, 0, 0,
                              0, maxScale / origScale, 0,
                              0, 0, 1);

      best.H = S_target_inv * H * S_root;
      best.scale = origScale;
      best.inliers = inliers;
      best.valid = true;
    }
  }
  if (!best.H.empty()) {
    best.H.at<double>(0, 2) /= downscale;
    best.H.at<double>(1, 2) /= downscale;
  }

  return best;
}
*/

inline HomographyResultM findHomographyAKAZE_multiscale(
    const cv::Mat& root,
    const cv::Mat& target,
    const std::vector<float>& scales)
{
  float downsample = 0.5;
    HomographyResultM best;

    if (scales.empty())
        return best;

    // --- 1. Find max scale ---
    float maxScale = *std::max_element(scales.begin(), scales.end());

    // --- 2. Normalize scales ---
    std::vector<float> normScales;
    normScales.reserve(scales.size());
    for (float s : scales)
        normScales.push_back(s / maxScale);

    // --- 3. Resize root ONCE ---
    cv::Mat img1_scaled;
    cv::resize(root, img1_scaled,
               cv::Size(), downsample / maxScale, downsample / maxScale,
               cv::INTER_AREA);

    // --- 4. Extract root features ONCE ---
    auto akaze = cv::AKAZE::create();

    std::vector<cv::KeyPoint> kp1;
    cv::Mat desc1;
    akaze->detectAndCompute(img1_scaled, cv::noArray(), kp1, desc1);

    if (desc1.empty())
        return best;

    cv::BFMatcher matcher(cv::NORM_HAMMING);

    // --- 5. Loop over normalized scales ---
    for (int i = 0; i < normScales.size(); i++)
    {
        float s = normScales[i];

        // --- Always <= 1, so always downsampling ---
        cv::Mat img2_scaled;
        cv::resize(target, img2_scaled,
                   cv::Size(), s * downsample, s * downsample,
                   cv::INTER_AREA);

        // --- Extract features ---
        std::vector<cv::KeyPoint> kp2;
        cv::Mat desc2;
        akaze->detectAndCompute(img2_scaled, cv::noArray(), kp2, desc2);

        if (desc2.empty())
            continue;

        // --- Match ---
        std::vector<std::vector<cv::DMatch>> knn;
        matcher.knnMatch(desc1, desc2, knn, 2);

        std::vector<cv::DMatch> good;
        for (auto& m : knn)
        {
            if (m.size() < 2) continue;
            if (m[0].distance < 0.75f * m[1].distance)
                good.push_back(m[0]);
        }

        if (good.size() < 10)
            continue;

        // --- Points ---
        std::vector<cv::Point2f> pts1, pts2;
        pts1.reserve(good.size());
        pts2.reserve(good.size());

        for (auto& m : good)
        {
            pts1.push_back(kp1[m.queryIdx].pt);
            pts2.push_back(kp2[m.trainIdx].pt);
        }

        // --- Homography ---
        std::vector<char> mask;
        cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, 3.0, mask);

        if (H.empty())
            continue;

        int inliers = std::count(mask.begin(), mask.end(), 1);

        if (inliers > best.inliers)
        {
            // --- Undo BOTH scalings ---
            // root scaled by 1/maxScale
            // target scaled by s = (original_scale / maxScale)

            float origScale = scales[i];

            cv::Mat S = (cv::Mat_<double>(3,3) <<
                maxScale / origScale, 0, 0,
                0, maxScale / origScale, 0,
                0, 0, 1);

            best.H = S * H;

            best.scale = origScale;
            best.inliers = inliers;
            best.valid = true;
        }
    }
  if (!best.H.empty()) {
    best.H.at<double>(0, 2) /= downsample;
    best.H.at<double>(1, 2) /= downsample;
  }

    return best;
}

int main(int argc, char **argv) {
  auto root = load_raw("/home/cm/Documents/data/phase_corr_testing/test_images_raw/breast/B1_2x.raw");
  auto target = load_raw("/home/cm/Documents/data/phase_corr_testing/test_images_raw/breast/B1_20x.raw");
  // auto root = load_raw("/home/cm/Documents/data/blur_test/raw/548.Raw");
  // auto target = load_raw("/home/cm/Documents/data/blur_test/raw/549.Raw");
  std::vector<float> cand10 = {5, 2.5, 1, 0.5, 0.25};
  std::vector<float> candidateScale = {1, 0.5, 0.2, 0.1, 0.05};
  auto start = std::chrono::high_resolution_clock::now();

  auto res = findHomographyAKAZE_multiscale(root, target, candidateScale);

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::high_resolution_clock::now() - start).count();
  if (res.valid) {
    // std::cout << "Inliers: " << res.numInliers << std::endl;
    std::cout << "H:\n" << res.H << std::endl;
    std::cout << "scale " << res.scale << std::endl;
    std::cout << duration << std::endl;
  }

  auto res1 = findHomographyAKAZE_multiscale(root, target, std::vector<float>{1});
  if (res1.valid) {
    // std::cout << "Inliers: " << res.numInliers << std::endl;
    std::cout << "H:\n" << res1.H << std::endl;
    std::cout << "scale " << res1.scale << std::endl;
    std::cout << duration << std::endl;
  }
  return 0;
};
