//
// Created by cooper maira on 9/29/25.
//

#include "pathCam.h"
#include "Poco/DirectoryIterator.h"
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>

#include <regex>

using Poco::DirectoryIterator;


void compareSiftPoints(const SiftPoint* a, const SiftPoint* b,
                       int numA, int numB, float eps = 1e-4f)
{
  if (!a || !b) {
    std::cerr << "Null pointer passed to compareSiftPoints.\n";
    return;
  }

  const int N = std::min(numA, numB);
  int total_diffs = 0;

  for (int i = 0; i < N; ++i) {
    const SiftPoint& pa = a[i];
    const SiftPoint& pb = b[i];

auto odif = std::fabs(pa.orientation - pb.orientation);

    bool same_pos =
        std::fabs(pa.xpos - pb.xpos) < eps &&
        std::fabs(pa.ypos - pb.ypos) < eps;
    bool same_scale =
        std::fabs(pa.scale - pb.scale) < eps;
    bool same_orient = odif < eps;

    int desc_diff = 0;
    float desc_sum_diff = 0.0f;
    for (int d = 0; d < 128; ++d) {
      float diff = std::fabs(pa.data[d] - pb.data[d]);
      if (diff > eps) desc_diff++;
      desc_sum_diff += diff;
    }

    if (!same_pos || !same_scale || !same_orient || desc_diff > 0) {
      if (i>0) {
        const SiftPoint& a1 = a[i-1];
        const SiftPoint& a2 = a[i+1];
        const SiftPoint& b1 = b[i-1];
        const SiftPoint& b2 = b[i+1];
        int k = 0;
      }
      total_diffs++;
      std::cout << "Idx " << i
                << " Δx=" << pa.xpos - pb.xpos
                << " Δy=" << pa.ypos - pb.ypos
                << " Δscale=" << pa.scale - pb.scale
                << " Δorient=" << pa.orientation - pb.orientation
                << " descDiffs=" << desc_diff
                << " meanDescΔ=" << (desc_sum_diff / 128.0f)
                << "\n";
    }
  }

  std::cout << "Compared " << N << " descriptors, "
            << total_diffs << " show differences.\n";

  if (numA != numB)
    std::cout << "Warning: array sizes differ ("
              << numA << " vs " << numB << ")\n";
}

void sortSiftDataByScaleHost(SiftData& sd,
                                    bool descending = false,
                                    bool upload_after = true)
{
  const int N = sd.numPts;
  if (N <= 1) {
    if (upload_after && sd.d_data && sd.h_data) {
      cudaMemcpy(sd.d_data, sd.h_data, N * sizeof(SiftPoint), cudaMemcpyHostToDevice);
    }
    return;
  }

  // Ensure we have host data; if not, create a host buffer and download first
  bool owns_temp_host = false;
  if (!sd.h_data) {
    sd.h_data = new SiftPoint[sd.maxPts]; // assumes maxPts valid
    cudaMemcpy(sd.h_data, sd.d_data, N * sizeof(SiftPoint), cudaMemcpyDeviceToHost);
    owns_temp_host = true;
  }

  // Make a copy we can reorder
  std::vector<SiftPoint> tmp(N);
  std::memcpy(tmp.data(), sd.h_data, N * sizeof(SiftPoint));

  // Sort by scale
  if (descending) {
    std::stable_sort(tmp.begin(), tmp.end(),
        [](const SiftPoint& a, const SiftPoint& b) {
            if (a.scale == b.scale)
              return a.orientation < b.orientation;  // tie-break by orientation
          return a.scale > b.scale;                   // primary: larger scale first
      });
  } else {
    std::stable_sort(tmp.begin(), tmp.end(),
        [](const SiftPoint& a, const SiftPoint& b) {
            if (a.scale == b.scale)
              return a.orientation < b.orientation;  // tie-break by orientation
          return a.scale < b.scale;                   // primary: larger scale first
      });
  }

  // Write back to host buffer
  std::memcpy(sd.h_data, tmp.data(), N * sizeof(SiftPoint));

  // Optionally upload to device
  if (upload_after && sd.d_data) {
    cudaMemcpy(sd.d_data, sd.h_data, N * sizeof(SiftPoint), cudaMemcpyHostToDevice);
  }

  // If we allocated a temporary host buffer (because sd.h_data was null), free it
  if (owns_temp_host) {
    delete[] sd.h_data;
    sd.h_data = nullptr;  // restore original state
  }
}


static inline bool hasRawExtension(const std::string &name) {
  auto pos = name.find_last_of('.');
  if (pos == std::string::npos) return false;
  std::string ext = name.substr(pos + 1);
  for (auto &c: ext) c = (char) std::tolower((unsigned char) c);
  return ext == "raw"; // accept ".raw" (case-insensitive)
}

// ---- Load a RAW (8-bit Bayer BG) into GPU gray via debayer ----
static bool loadRawToGpuGray(const std::string &path,
                             int width, int height,
                             cv::cuda::GpuMat &outGray,
                             int bayerCode = cv::COLOR_BayerBG2GRAY) {
  // Read file
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::cerr << "[WARN] Cannot open file: " << path << "\n";
    return false;
  }
  std::streamsize size = f.tellg();
  f.seekg(0, std::ios::beg);

  const size_t expected = static_cast<size_t>(width) * static_cast<size_t>(height);
  if (size != static_cast<std::streamsize>(expected)) {
    std::cerr << "[WARN] Size mismatch for " << path
        << " (got " << size << ", expected " << expected << " bytes)\n";
    return false;
  }

  std::vector<unsigned char> buf(expected);
  if (!f.read(reinterpret_cast<char *>(buf.data()), size)) {
    std::cerr << "[WARN] Failed to read data: " << path << "\n";
    return false;
  }

  // Wrap CPU Mat as single-channel 8-bit
  cv::Mat raw(height, width, CV_8UC1, buf.data());

  // Upload to GPU
  cv::cuda::GpuMat dRaw, dGray;
  dRaw.upload(raw);

  // Debayer on GPU -> grayscale
  cv::cuda::cvtColor(dRaw, dGray, bayerCode);

  // Return
  outGray = dGray; // shallow copy of GpuMat header (data stays on GPU)
  return true;
}

// ---- Main routine: pairwise scaling among all .Raw images in a directory ----
void estimateScalesForRawDirectory(const std::string &dirPath,
                                   int width, int height,
                                   const std::vector<double> &scales = {2.0, 2.5, 4.0, 5.0, 10.0}) {
  try {
    // 1) Collect .Raw file paths
    std::vector<std::string> files;
    for (Poco::DirectoryIterator it(dirPath), end; it != end; ++it) {
      if (!it->isFile()) continue;
      const std::string p = it->path();
      if (hasRawExtension(p)) files.push_back(p);
    }

    if (files.size() < 2) {
      std::cerr << "[INFO] Need at least two .Raw files in " << dirPath << "\n";
      return;
    }

    // 2) For every unordered pair (i, j), i<j:
    for (size_t i = 0; i + 1 < files.size(); ++i) {
      for (size_t j = i + 1; j < files.size(); ++j) {
        const std::string &fA = files[i];
        const std::string &fB = files[j];

        cv::cuda::GpuMat dGrayA, dGrayB;
        if (!loadRawToGpuGray(fA, width, height, dGrayA)) {
          std::cerr << "[WARN] Skipping pair due to load failure: " << fA << " & " << fB << "\n";
          continue;
        }
        if (!loadRawToGpuGray(fB, width, height, dGrayB)) {
          std::cerr << "[WARN] Skipping pair due to load failure: " << fA << " & " << fB << "\n";
          continue;
        }

        // 3) Estimate scale (auto tries both orders)
        auto r = pathCam::CompositeVoronoi::estimate_scale_auto_GPU(dGrayA, dGrayB, scales);

        // 4) Print result
        Poco::Path pa(fA), pb(fB);
        std::cout << pa.getFileName() << "  <->  " << pb.getFileName() << " : ";
        if (!r.valid) {
          std::cout << "no valid estimate\n";
        } else {
          std::cout << "scale=" << r.scale
              << "  response=" << r.response
              << "  shift=(" << r.shift.x << "," << r.shift.y << ")"
              << "  |shift|=" << r.shiftNorm
              << "  normShift=" << r.shiftNormDiag
              << "\n";
        }
      }
    }
  } catch (const Poco::Exception &e) {
    std::cerr << "[ERROR] Poco exception: " << e.displayText() << "\n";
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] std::exception: " << e.what() << "\n";
  }
}


SiftData get_sift_data_from_raw(cuda::GpuMat &_img, float initBlur, float thresh, float lowestScale) {
  SiftData siftData;
  cuda::GpuMat gry, gry2;

  //cuda::cvtColor(_img, gry, COLOR_BayerBG2GRAY);
  _img.convertTo(gry2,CV_32FC1);

  CudaImage cImgGry;
  cImgGry.Allocate(_img.cols, _img.rows, gry2.step / sizeof(float), false,
                   reinterpret_cast<float *>(gry2.data), nullptr);

  int n = 200000;
  InitSiftData(siftData, n, true, true);
  if (siftData.numPts == n) {
    int k = 0;
  }


  if (0 < ExtractSift(siftData, cImgGry, 5, initBlur, thresh, lowestScale, false)) {
    int k = 0;
  }


  return siftData;
}
void shuffle_sift_data(SiftData& sd) {
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
    tmp[i] = sd.h_data[enm[i]];  // direct struct assignment, not memcpy

  // 5. Write back to original
  for (int i = 0; i < sd.numPts; ++i)
    sd.h_data[i] = tmp[i];

  // Optional: re-upload to GPU if device copy exists
  if (sd.d_data)
    cudaMemcpy(sd.d_data, sd.h_data, sd.numPts * sizeof(SiftPoint), cudaMemcpyHostToDevice);
}

int extractMagnification(const std::string &path) {
  std::regex re("([0-9]+)x");
  std::smatch match;
  if (std::regex_search(path, match, re)) {
    return std::stoi(match[1].str());
  }
  return -1; // fallback if no match
}

int test_ordered_match(SiftData &sd1, SiftData & sd2) {
  MatchSiftData(sd1,sd2);
  std::vector<Point2f> pts1, pts2;
  for (int i = 0; i < sd1.numPts; ++i) {
    if (sd1.h_data[i].match > 0) {
      pts1.emplace_back(sd1.h_data[i].xpos,sd1.h_data[i].ypos);
      int match = sd1.h_data[i].match;
      pts2.emplace_back(sd2.h_data[match].xpos,sd2.h_data[match].ypos);
    }
  }

  if (pts1.size() < 4 || pts2.size() < 4) { return 0; }

  std::vector<uchar> inlierMask;

  auto H = findHomography(pts2, pts1, RANSAC, 3.0, inlierMask);
  int numInliers = std::count(inlierMask.begin(), inlierMask.end(), 1);
  //std::cout<<numInliers<<std::endl;
  //}
  return numInliers;
}

int find_inliers(SiftData soldier1, SiftData soldier2) {
  int numInliers;

  MatchSiftData(soldier1, soldier2);
  MatchSiftData(soldier2, soldier1);

  std::vector<DMatch> mutualMatches;
  std::vector<Point2f> pts1, pts2;

  for (int i = 0; i < soldier1.numPts; ++i) {
    int matchIdx = soldier1.h_data[i].match;
    if (matchIdx < 0 || soldier2.numPts < matchIdx) {
      continue;
    }

    if (soldier2.h_data[matchIdx].match == i) {
      auto amb = soldier2.h_data[matchIdx].ambiguity;
      //if (soldier2.h_data[matchIdx].ambiguity < 0.75) {
        mutualMatches.emplace_back(i, matchIdx, soldier1.h_data[i].match_error);

        pts1.emplace_back(soldier1.h_data[i].xpos, soldier1.h_data[i].ypos);
        pts2.emplace_back(soldier2.h_data[matchIdx].xpos, soldier2.h_data[matchIdx].ypos);
      //}
    }
  }

  // FreeSiftData(soldier1);
  // FreeSiftData(soldier2);

  if (pts1.size() < 4 || pts2.size() < 4) { return 0; }

  std::vector<uchar> inlierMask;

  auto H = findHomography(pts2, pts1, RANSAC, 3.0, inlierMask);
  numInliers = std::count(inlierMask.begin(), inlierMask.end(), 1);
  //std::cout<<numInliers<<std::endl;
  //}
  return numInliers;
}

int min_inliers_2(std::vector<cuda::GpuMat> &_files, cuda::SURF_CUDA &_surf, Ptr<cuda::DescriptorMatcher> &_matcher) {
  int minInliers = 10000000;

  std::vector<cuda::GpuMat> descriptors(_files.size()), keypoints(_files.size());
  std::vector<std::vector<KeyPoint> > kpH(_files.size());
  Mat circleMask(_files[0].size(),CV_8UC1, Scalar(0));
  circle(circleMask, {3232, 2426}, 2100, Scalar(255), -1);
  cuda::GpuMat mask2x;
  mask2x.upload(circleMask);
  for (int i = 0; i < _files.size(); ++i) {
    if (i == 0) {
      _surf(_files[i], mask2x, keypoints[i], descriptors[i]);
    } else {
      _surf(_files[i], cuda::GpuMat(), keypoints[i], descriptors[i]);
    }
    _surf.downloadKeypoints(keypoints[i], kpH[i]);
  }

  const float ratio = 0.75f;
  for (size_t i = 0; i + 1 < _files.size(); ++i) {
    for (size_t j = i + 1; j < _files.size(); ++j) {
      if (j - i >= 4) {
        continue; //skip extreme jumps in mag, 2x -> 40x
      }
      std::vector<std::vector<DMatch> > knn;
      _matcher->knnMatch(descriptors[i], descriptors[j], knn, 2);

      std::vector<DMatch> good;
      good.reserve(knn.size());
      for (auto &pair: knn) {
        if (pair.size() == 2 && pair[0].distance <= ratio * pair[1].distance)
          good.push_back(pair[0]);
      }

      if (good.size() < 4) {
        std::cout << "failure between " << i << " and " << j << std::endl;
        continue;
      }

      std::vector<Point2f> p1, p2;
      p1.reserve(good.size());
      p2.reserve(good.size());
      for (auto &m: good) {
        p1.push_back(kpH[i][m.queryIdx].pt);
        p2.push_back(kpH[j][m.trainIdx].pt);
      }
      std::vector<unsigned char> inl;
      cv::Mat H = cv::findHomography(p1, p2, cv::RANSAC, 3.0, inl);

      int inliers = std::count(inl.begin(), inl.end(), 1);
      std::cout << inliers << " " << i << " " << j << std::endl;

      if (inliers < minInliers) {
        minInliers = inliers;
      }
    }
  }

  return minInliers;
}



SiftData fill_cudasift_from_cv(Mat _descriptors, std::vector<KeyPoint> _kp) {
  assert(_kp.size() == _descriptors.rows);

  SiftData siftData;
  InitSiftData(siftData, _kp.size(), true, true);
  siftData.numPts = _kp.size();

  for (int i = 0; i < _kp.size(); ++i) {
    SiftPoint &sp = siftData.h_data[i];
    sp.xpos = _kp[i].pt.x;
    sp.ypos = _kp[i].pt.y;
    sp.scale = _kp[i].size;
    sp.orientation = _kp[i].angle;
    sp.score = _kp[i].response;
    sp.match = -1;
    sp.match_xpos = sp.match_ypos = 0.0f;
    sp.ambiguity = 0.0f;

    const float* src = _descriptors.ptr<float>(i);
    std::memcpy(sp.data, src, 128 * sizeof(float));
  }

  cudaMemcpy(siftData.d_data, siftData.h_data, _kp.size() * sizeof(SiftPoint), cudaMemcpyHostToDevice);

  return siftData;
}

int min_inliers_3(std::vector<cv::Mat> const &imgs,
                  cv::Ptr<cv::SIFT> const &sift,
                  cv::Ptr<cv::DescriptorMatcher> const &matcher) {
  int minInliers = std::numeric_limits<int>::max();

  std::vector<cv::Mat> descriptors(imgs.size());
  std::vector<std::vector<cv::KeyPoint> > kps(imgs.size());

  // Optional mask only for the first image
  cv::Mat mask;
  if (!imgs.empty()) {
    mask = cv::Mat(imgs[0].size(), CV_8UC1, cv::Scalar(0));
    cv::circle(mask, {3232, 2426}, 2100, cv::Scalar(255), -1);
  }

  // Detect & compute (CPU)
  for (size_t i = 0; i < imgs.size(); ++i) {
    const cv::Mat &m = imgs[i];
    if (m.empty() || m.type() != CV_8UC1) {
      std::cerr << "Image " << i << " empty or not CV_8UC1\n";
      return 0;
    }
    sift->detectAndCompute(m, (i == 0 ? mask : cv::Mat()), kps[i], descriptors[i]);

    // SIFT should give CV_32F. Convert if needed (safety).
    if (!descriptors[i].empty() && descriptors[i].type() != CV_32F)
      descriptors[i].convertTo(descriptors[i], CV_32F);
  }

  const float ratio = 0.75f;

  for (size_t i = 0; i + 1 < imgs.size(); ++i) {
    for (size_t j = i + 1; j < imgs.size(); ++j) {
      if (j - i >= 4) continue; // your skip

      // Guard empties BEFORE knnMatch
      if (descriptors[i].empty() || descriptors[j].empty()) {
        std::cout << "skip pair i=" << i << " j=" << j
            << " (empty descriptors: "
            << descriptors[i].rows << "x" << descriptors[i].cols << " , "
            << descriptors[j].rows << "x" << descriptors[j].cols << ")\n";
        continue;
      }
      if (descriptors[i].type() != CV_32F || descriptors[j].type() != CV_32F) {
        std::cerr << "Type mismatch (expect CV_32F) at i=" << i << " j=" << j << "\n";
        continue;
      }
      if (descriptors[i].cols != descriptors[j].cols) {
        std::cerr << "Descriptor dim mismatch ("
            << descriptors[i].cols << " vs " << descriptors[j].cols
            << ") at i=" << i << " j=" << j << "\n";
        continue;
      }

      auto sf1 = fill_cudasift_from_cv(descriptors[i], kps[i]);
      auto sf2 = fill_cudasift_from_cv(descriptors[j], kps[j]);
      auto val = find_inliers(sf1, sf2);
      std::cout << "CudaSift inliers: " << val << "  i=" << i << " j=" << j<< std::endl;
      shuffle_sift_data(sf1);
      shuffle_sift_data(sf2);
      val = find_inliers(sf1,sf2);
      std::cout << "CudaSift inliers try 2: " << val << "  i=" << i << " j=" << j<< std::endl;



      std::vector<std::vector<cv::DMatch> > knn;
      matcher->knnMatch(descriptors[i], descriptors[j], knn, 2);

      std::vector<cv::DMatch> good;
      good.reserve(knn.size());
      for (auto const &pair: knn) {
        if (pair.size() == 2 && pair[0].distance <= ratio * pair[1].distance)
          good.push_back(pair[0]);
      }
      if (good.size() < 4) {
        std::cout << "failure between " << i << " and " << j
            << " (good=" << good.size() << ")\n";
        continue;
      }

      std::vector<cv::Point2f> p1, p2;
      p1.reserve(good.size());
      p2.reserve(good.size());
      for (auto const &m: good) {
        p1.push_back(kps[i][m.queryIdx].pt);
        p2.push_back(kps[j][m.trainIdx].pt);
      }

      std::vector<unsigned char> inl;
      cv::Mat H = cv::findHomography(p1, p2, cv::RANSAC, 3.0, inl);
      int inliers = H.empty() ? 0 : int(std::count(inl.begin(), inl.end(), 1));

      std::cout << "inliers=" << inliers << "  i=" << i << " j=" << j << "\n";

      minInliers = std::min(minInliers, inliers);
    }
  }
  return (minInliers == std::numeric_limits<int>::max() ? 0 : minInliers);
}

void min_inliers4(std::vector<cuda::GpuMat> files, float initBlur, float thresh, float lowestScale) {
  std::vector<SiftData> siftData;
  for (int i = 0; i < files.size(); ++i) {
    auto sd = get_sift_data_from_raw(files[i], initBlur, thresh, lowestScale);
    sortSiftDataByScaleHost(sd);
    siftData.push_back(sd);
  }

  int i = 0,j = 2;

  std::vector<SiftPoint> saveBuffer(siftData[i].numPts);
  std::memcpy(saveBuffer.data(),siftData[i].h_data,siftData[i].numPts * sizeof(SiftPoint));

  for (int z = 0; z < 5; ++z) {
    auto val = test_ordered_match(siftData[i],siftData[j]);
    std::cout<<"static try "<<z<<" val="<<val<<std::endl;
  }

  shuffle_sift_data(siftData[i]);
  for (int z = 0; z < 5; ++z) {
    auto val = test_ordered_match(siftData[i],siftData[j]);
    std::cout<<"shuffle try "<<z<<" val="<<val<<std::endl;
  }
  sortSiftDataByScaleHost(siftData[i]);
  for (int z = 0; z < 5; ++z) {
    auto val = test_ordered_match(siftData[i],siftData[j]);
    std::cout<<"second static try "<<z<<" val="<<val<<std::endl;
  }

  compareSiftPoints(saveBuffer.data(),siftData[i].h_data,siftData[i].numPts,siftData[i].numPts);
  int k = 0;

}

int min_inliers(std::vector<cuda::GpuMat> files, float initBlur, float thresh, float lowestScale) {
  int minInliers = 10000000;
  for (int z = 0; z < 5; ++z) {
    std::vector<SiftData> siftData;

    for (int i = 0; i < files.size(); ++i) {
      // if (i == 2) {
      //   siftData.push_back(SiftData());
      // }
      auto sd = get_sift_data_from_raw(files[i], initBlur, thresh, lowestScale);
      sortSiftDataByScaleHost(sd);
      siftData.push_back(sd);
    }

    std::vector<SiftPoint> test;
    for (int i = 0; i < 1000; i+=51) {
      SiftPoint &p = siftData[i%5].h_data[i];
      test.push_back(p);
    }

    int x, y;
    int minForIter = 100000000;
    for (size_t i = 0; i + 1 < files.size(); ++i) {
      for (size_t j = i + 1; j < files.size(); ++j) {
        if (j - i >= 4) {
          continue; //skip extreme jumps in mag, 2x -> 40x
        }
        auto inliers = find_inliers(siftData[i], siftData[j]);
        std::cout<<inliers<<" i="<<i<<" j="<<j<<std::endl;

        if (inliers < minInliers) {
          minInliers = inliers;
        }
        // if (inliers < minForIter) {
        //   minForIter = inliers;
        //   std::cout<<i<<" "<<j<<std::endl;
        // }
      }
    }

    for (auto &i: siftData) {
      FreeSiftData(i);
    }

    //std::cout<<minInliers<<std::endl;
  }
  return minInliers;
}

void tune_sift(const std::string &dirPath, int width, int height, bool _sort) {
  // 1) Collect .Raw file paths
  std::vector<cuda::GpuMat> files;
  std::vector<std::string> paths;
  for (Poco::DirectoryIterator it(dirPath), end; it != end; ++it) {
    if (!it->isFile()) continue;
    const std::string p = it->path();
    if (hasRawExtension(p)) {
      paths.push_back(p);
    }
  }

  if (_sort) {
    std::sort(paths.begin(), paths.end(), [](const std::string &a, const std::string &b) {
      return extractMagnification(a) < extractMagnification(b);
    });
  }
  for (auto &p: paths) {
    cuda::GpuMat im;
    loadRawToGpuGray(p, 6464, 4852, im);
    files.push_back(im);
  }

  if (files.size() < 2) {
    std::cerr << "[INFO] Need at least two .Raw files in " << dirPath << "\n";
    return;
  }


  int bestMinInliers = 0;
  float bestInitBlur, bestThresh, bestLowestScalse;
  int iterCount = 0;
  for (float initBlur = 0.f; initBlur <= 2.001f; initBlur += 0.1) {
    //21

    for (float thresh = 0.f; thresh <= 5.001f; thresh += 0.2) {
      //21
      for (float lowestScale = 0.f; lowestScale <= 2.001f; lowestScale += 0.1) {
        //21

        auto minInliers = min_inliers(files, initBlur, thresh, lowestScale);

        if (minInliers > bestMinInliers) {
          bestMinInliers = minInliers;
          bestInitBlur = initBlur;
          bestThresh = thresh;
          bestLowestScalse = lowestScale;
        }

        if (int(iterCount) % 100 == 0) {
          std::cout << iterCount << std::endl;
        }
        ++iterCount;
      }
    }
  }
  std::cout << "best min inliers: " << bestMinInliers << std::endl;
  std::cout << "best initBlur: " << bestInitBlur << std::endl;
  std::cout << "best thresh: " << bestThresh << std::endl;
  std::cout << "best lowestScale: " << bestLowestScalse << std::endl;
  int k = 0;
}

int main(int argc, char *argv[]) {
  //estimateScalesForRawDirectory(argv[1], 6464, 4852);
  //tune_sift(argv[1], 6464, 4852, true);
  //
  // const std::string fa = "/home/max/Downloads/between_machines-selected/390.Raw";
  // const std::string fb = "/home/max/Downloads/between_machines-selected/306.Raw";
  //
  // cuda::GpuMat imga, imgb;
  //
  // loadRawToGpuGray(fa, 6464, 4852, imga);
  // loadRawToGpuGray(fb, 6464, 4852, imgb);
  //
  // auto sift1 = get_sift_data_from_raw(imga, .1, 1, 1.3);
  // auto sift2 = get_sift_data_from_raw(imgb, .1, 1, 1.3);
  //
  // auto val = find_inliers(sift1, sift2);

  std::vector<cuda::GpuMat> files;
  std::vector<Mat> files2;
  std::vector<std::string> paths;
  const std::string path = argv[1];
  for (Poco::DirectoryIterator it(path), end; it != end; ++it) {
    if (!it->isFile()) continue;
    const std::string p = it->path();
    if (hasRawExtension(p)) {
      paths.push_back(p);
    }
  }

  if (true) {
    std::sort(paths.begin(), paths.end(), [](const std::string &a, const std::string &b) {
      return extractMagnification(a) < extractMagnification(b);
    });
  }
  for (auto &p: paths) {
    cuda::GpuMat im;
    loadRawToGpuGray(p, 6464, 4852, im);
    files.push_back(im);
    Mat temp;
    im.download(temp);
    files2.push_back(temp);
  }
min_inliers4(files,.1, 1, 1.3);

  auto val = min_inliers(files,.1, 1, 1.3);

  //auto val = min_inliers(files,.1, 1, 1.3);
  // auto val = min_inliers(files, 0, 1.8, 0.5);
  // std::cout<< "inliers: "<<val<<std::endl;
  int best = 0;
  int ht = 0;
  //for (int iii = 0;iii < 20;++iii) {
  // cuda::SURF_CUDA surf(400,6,5);
  // auto matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_L2);
  //auto val = min_inliers_2(files,surf,matcher);
  // if (val > best) {
  //   best = val;
  //   ht = 45*iii;
  // }
  auto matcher2 = cv::BFMatcher::create(NORM_L2);
  auto sift = cv::SIFT::create(200000, 5, .02, 10, 1.6);
  auto val2 = min_inliers_3(files2, sift, matcher2);
  //}
  int k = 0;
}
