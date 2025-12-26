//
// Created by cooper maira on 9/29/25.
//

#include <filesystem>

#include "pathCam.h"
#include "Poco/DirectoryIterator.h"
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>

#include <regex>

using Poco::DirectoryIterator;

std::ofstream logFile("/home/max/Documents/siftTune.log", std::ios::out | std::ios::app);

float mag_ratio_lookup(int i, int j) {
  float mags[] = {2.f, 4.f, 10.f, 20.f, 40.f};
  return mags[j] / mags[i];
}

using GroupCallback = std::function<void(const std::string &, const std::array<std::string, 5> &)>;

inline int magIndex(int mag) {
  switch (mag) {
    case 2: return 0;
    case 4: return 1;
    case 10: return 2;
    case 20: return 3;
    case 40: return 4;
    default: return -1;
  }
}

struct Group {
  std::string subdir; // the subfolder where the group was found
  int caseId; // 1, 2, or 3
  std::array<std::string, 5> paths; // ordered by {2,4,10,20,40}
  std::vector<cuda::GpuMat> images;
};

std::vector<Group> collectGroups(const std::string &rootDir, bool onlyComplete = true) {
  // File name pattern: <CapitalLetter><[1|2|3]>_(2|4|10|20|40)x.raw
  // e.g., A1_20x.raw, Z3_4x.raw
  const std::regex pat(R"(^([A-Z])([123])_(2|4|10|20|40)x\.raw$)");

  std::vector<Group> out;

  Poco::File root(rootDir);
  if (!root.exists() || !root.isDirectory()) {
    std::cerr << "Root directory does not exist or is not a directory: " << rootDir << "\n";
    return out;
  }

  // Iterate immediate subfolders
  for (Poco::DirectoryIterator it(root), end; it != end; ++it) {
    if (!it->isDirectory()) continue;

    const std::string subDirPath = it->path();

    // For each subfolder, group by the first number (caseId: 1,2,3)
    // Each entry holds 5 paths ordered by magnification indices {2,4,10,20,40}
    std::unordered_map<int, std::array<std::string, 5> > groups;

    for (Poco::DirectoryIterator jt(subDirPath), jend; jt != jend; ++jt) {
      if (!jt->isFile()) continue;

      const std::string fname = Poco::Path(jt->path()).getFileName();
      std::smatch m;
      if (!std::regex_match(fname, m, pat)) continue;

      int caseId = std::stoi(m[2].str()); // 1..3
      int mag = std::stoi(m[3].str()); // 2,4,10,20,40
      int idx = magIndex(mag);
      if (idx < 0) continue;

      auto &arr = groups[caseId];
      arr[idx] = jt->path();
    }

    // Push groups
    for (auto &kv: groups) {
      const int caseId = kv.first;
      const auto &arr = kv.second;

      bool complete = true;
      for (const auto &p: arr) {
        if (p.empty()) {
          complete = false;
          break;
        }
      }

      if (!onlyComplete || complete) {
        out.push_back(Group{subDirPath, caseId, arr});
      }
    }
  }

  return out;
}

void groupMagnificationsPerCase(const std::string &rootDir, const GroupCallback &onGroup) {
  // File name pattern: <CapitalLetter><[1|2|3]>_<[2|4|10|20|40]>x.raw
  // e.g., A1_20x.raw, Z3_4x.raw
  // Capture: 1) letter, 2) caseId (1..3), 3) magnification
  const std::regex pat(R"(^([A-Z])([123])_(2|4|10|20|40)x\.raw$)");

  Poco::File root(rootDir);
  if (!root.exists() || !root.isDirectory()) {
    std::cerr << "Root directory does not exist or is not a directory: " << rootDir << "\n";
    return;
  }

  // Iterate immediate subfolders
  for (Poco::DirectoryIterator it(root), end; it != end; ++it) {
    if (!it->isDirectory()) continue;

    const std::string subDirPath = it->path();
    // For each subfolder, group by the first number (caseId: 1,2,3)
    // Each entry holds 5 paths ordered by magnification indices {2,4,10,20,40}
    std::unordered_map<int, std::array<std::string, 5> > groups;

    for (Poco::DirectoryIterator jt(subDirPath), jend; jt != jend; ++jt) {
      if (!jt->isFile()) continue;

      const std::string fname = Poco::Path(jt->path()).getFileName();
      std::smatch m;
      if (!std::regex_match(fname, m, pat)) continue;

      // Parse captures
      // std::string letter = m[1]; // unused letter
      int caseId = std::stoi(m[2].str()); // 1,2,3
      int mag = std::stoi(m[3].str()); // 2,4,10,20,40
      int idx = magIndex(mag);
      if (idx < 0) continue;

      auto &arr = groups[caseId];
      arr[idx] = jt->path();
    }

    // Emit only complete groups (all five mags present)
    for (auto &kv: groups) {
      const int caseId = kv.first;
      const auto &arr = kv.second;

      bool complete = true;
      for (const auto &p: arr) {
        if (p.empty()) {
          complete = false;
          break;
        }
      }
      if (complete) {
        onGroup(subDirPath, arr);
      } else {
        // Optional: warn about missing magnifications for this caseId
        std::cerr << "Warning: In subdir " << subDirPath
            << " case " << caseId << " is missing one or more magnifications.\n";
      }
    }
  }
}


void compareSiftPoints(const SiftPoint *a, const SiftPoint *b,
                       int numA, int numB, float eps = 1e-4f) {
  if (!a || !b) {
    std::cerr << "Null pointer passed to compareSiftPoints.\n";
    return;
  }

  const int N = std::min(numA, numB);
  int total_diffs = 0;

  for (int i = 0; i < N; ++i) {
    const SiftPoint &pa = a[i];
    const SiftPoint &pb = b[i];

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
      if (i > 0) {
        const SiftPoint &a1 = a[i - 1];
        const SiftPoint &a2 = a[i + 1];
        const SiftPoint &b1 = b[i - 1];
        const SiftPoint &b2 = b[i + 1];
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

template<typename T1 = float, typename T2 = float>
void sortSiftDataHost(SiftData &sd,
                      T1 SiftPoint::*primary = &SiftPoint::scale,
                      T2 SiftPoint::*secondary = &SiftPoint::orientation,
                      bool descending = false,
                      bool upload_after = true) {
  const int N = sd.numPts;
  if (N <= 1) {
    if (upload_after && sd.d_data && sd.h_data)
      cudaMemcpy(sd.d_data, sd.h_data, N * sizeof(SiftPoint), cudaMemcpyHostToDevice);
    return;
  }

  bool owns_temp_host = false;
  if (!sd.h_data) {
    sd.h_data = new SiftPoint[sd.maxPts];
    cudaMemcpy(sd.h_data, sd.d_data, N * sizeof(SiftPoint), cudaMemcpyDeviceToHost);
    owns_temp_host = true;
  }

  std::vector<SiftPoint> tmp(N);
  std::memcpy(tmp.data(), sd.h_data, N * sizeof(SiftPoint));

  // Generic comparator based on the member pointers
  if (descending) {
    std::stable_sort(tmp.begin(), tmp.end(),
                     [primary, secondary](const SiftPoint &a, const SiftPoint &b) {
                       if (a.*primary == b.*primary)
                         return a.*secondary < b.*secondary;
                       return a.*primary > b.*primary;
                     });
  } else {
    std::stable_sort(tmp.begin(), tmp.end(),
                     [primary, secondary](const SiftPoint &a, const SiftPoint &b) {
                       if (a.*primary == b.*primary)
                         return a.*secondary < b.*secondary;
                       return a.*primary < b.*primary;
                     });
  }

  std::memcpy(sd.h_data, tmp.data(), N * sizeof(SiftPoint));

  if (upload_after && sd.d_data)
    cudaMemcpy(sd.d_data, sd.h_data, N * sizeof(SiftPoint), cudaMemcpyHostToDevice);

  if (owns_temp_host) {
    delete[] sd.h_data;
    sd.h_data = nullptr;
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


SiftData get_sift_data_from_gry(cuda::GpuMat &_img, float initBlur, float thresh, float lowestScale,
                                int numPts = 200000) {
  SiftData siftData;
  cuda::GpuMat gry, gry2;

  _img.convertTo(gry2,CV_32FC1);

  CudaImage cImgGry;
  cImgGry.Allocate(_img.cols, _img.rows, gry2.step / sizeof(float), false,
                   reinterpret_cast<float *>(gry2.data), nullptr);

  int n = numPts;
  InitSiftData(siftData, n, true, true);

  ExtractSift(siftData, cImgGry, 5, initBlur, thresh, lowestScale, false);

  return siftData;
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

int extractMagnification(const std::string &path) {
  std::regex re("([0-9]+)x");
  std::smatch match;
  if (std::regex_search(path, match, re)) {
    return std::stoi(match[1].str());
  }
  return -1; // fallback if no match
}

int test_oneway_homography(SiftData &sd) {
  std::vector<Point2f> pts1, pts2;
  for (int i = 0; i < sd.numPts; ++i) {
    if (sd.h_data[i].match > 0) {
      pts1.emplace_back(sd.h_data[i].xpos, sd.h_data[i].ypos);
      pts2.emplace_back(sd.h_data[i].match_xpos, sd.h_data[i].match_ypos);
    }
  }

  if (pts1.size() < 4 || pts2.size() < 4) { return 0; }

  std::vector<uchar> inlierMask;

  auto H = findHomography(pts2, pts1, RANSAC, 3.0, inlierMask);
  int numInliers = std::count(inlierMask.begin(), inlierMask.end(), 1);

  return numInliers;
}

int test_ordered_match(SiftData &sd1, SiftData &sd2) {
  MatchSiftData(sd1, sd2);
  std::vector<Point2f> pts1, pts2;
  for (int i = 0; i < sd1.numPts; ++i) {
    if (sd1.h_data[i].match > 0) {
      pts1.emplace_back(sd1.h_data[i].xpos, sd1.h_data[i].ypos);
      int match = sd1.h_data[i].match;
      pts2.emplace_back(sd2.h_data[match].xpos, sd2.h_data[match].ypos);
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

    const float *src = _descriptors.ptr<float>(i);
    std::memcpy(sp.data, src, 128 * sizeof(float));
  }

  cudaMemcpy(siftData.d_data, siftData.h_data, _kp.size() * sizeof(SiftPoint), cudaMemcpyHostToDevice);

  return siftData;
}

int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity,
                      float thresh);

int min_inliers(std::vector<cuda::GpuMat> files, float initBlur, float thresh, float lowestScale) {
  int minInliers = 10000000;
  for (int z = 0; z < 5; ++z) {
    std::vector<SiftData> siftData;

    for (int i = 0; i < files.size(); ++i) {
      // if (i == 2) {
      //   siftData.push_back(SiftData());
      // }
      auto sd = get_sift_data_from_gry(files[i], initBlur, thresh, lowestScale);
      sortSiftDataHost(sd);
      siftData.push_back(sd);
    }

    std::vector<SiftPoint> test;
    for (int i = 0; i < 1000; i += 51) {
      SiftPoint &p = siftData[i % 5].h_data[i];
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
        std::cout << inliers << " i=" << i << " j=" << j << std::endl;

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


void min_inliers4(std::vector<cuda::GpuMat> files, float initBlur, float thresh, float lowestScale) {
  std::vector<SiftData> siftData;
  for (int i = 0; i < files.size(); ++i) {
    auto sd = get_sift_data_from_gry(files[i], initBlur, thresh, lowestScale);
    sortSiftDataHost(sd);
    siftData.push_back(sd);
  }

  int x = 0, y = 2;

  for (int i = 0; i < siftData[x].numPts; ++i) {
    siftData[x].h_data[i].empty[0] = float(i);
  }

  SiftData sdi;
  InitSiftData(sdi, 200000, true, true);
  sdi.numPts = siftData[x].numPts;
  memcpy(sdi.h_data, siftData[x].h_data, sdi.numPts * sizeof(SiftPoint));
  cudaMemcpy(sdi.d_data, sdi.h_data, sdi.numPts * sizeof(SiftPoint), cudaMemcpyHostToDevice);

  shuffle_sift_data(siftData[x]);

  MatchSiftData(siftData[x], siftData[y]);
  MatchSiftData(sdi, siftData[y]);

  int count = 0;
  auto sdx = siftData[x];
  for (int i = 0; i < sdx.numPts; ++i) {
    int index = static_cast<int>(sdx.h_data[i].empty[0]);
    assert(index == sdi.h_data[index].empty[0]);
    auto &spx = sdx.h_data[i];
    auto &spi = sdi.h_data[index];
    if (spx.match != spi.match) {
      ++count;
    }
  }

  sortSiftDataHost(sdx, &SiftPoint::ambiguity, &SiftPoint::score);
  auto v1 = sdx.h_data[0];
  auto v2 = sdx.h_data[10000];
  int ii = 0, ii2 = 0;
  while (ii < 10) {
    ++ii2;


    //auto val = test_oneway_homography(sdx);
    float homography[9];
    int numMatches;
    FindHomography(sdx, homography, &numMatches, 10000, 0.00f, 0.88f, 5.0);


    if (homography[0] == 0 || numMatches < 100) {
      shuffle_sift_data(sdx);
      continue;
    }

    int numFit = ImproveHomography(sdx, homography, 5, 0.00f, 1.0f, 3.0);

    for (int z = 0; z < 9; ++z) {
      std::cout << homography[z] << " ";
    }
    std::cout << std::endl;

    std::cout << "cudasift findhomography " << numFit << " " << numMatches << std::endl;
    //std::cout<<"opencv ransac "<<val<<std::endl;

    shuffle_sift_data(sdx);

    ++ii;
  }


  for (int z = 0; z < 5; ++z) {
    auto val = test_ordered_match(siftData[x], siftData[y]);
    std::cout << "static try " << z << " val=" << val << std::endl;
  }

  shuffle_sift_data(siftData[x]);
  for (int z = 0; z < 5; ++z) {
    auto val = test_ordered_match(siftData[x], siftData[y]);
    std::cout << "shuffle try " << z << " val=" << val << std::endl;
  }
  sortSiftDataHost(siftData[x]);
  for (int z = 0; z < 5; ++z) {
    auto val = test_ordered_match(siftData[x], siftData[y]);
    std::cout << "second static try " << z << " val=" << val << std::endl;
  }

  compareSiftPoints(sdi.h_data, siftData[x].h_data, siftData[x].numPts, siftData[x].numPts);
  int k = 0;
}

bool verify_group_given_params(std::vector<cuda::GpuMat> files, float initBlur, float thresh, float lowestScale,
                               float ambiguityMin,
                               float ambiguityMax, float scoreMin, int numPts) {
  std::vector<SiftData> siftData;

  for (int i = 0; i < files.size(); ++i) {
    siftData.push_back(get_sift_data_from_gry(files[i], initBlur, thresh, lowestScale, numPts));
  }

  long fhTime = 0;
  for (int i = 0; i < files.size() - 1; ++i) {
    for (int j = i + 1; j < files.size(); ++j) {
      bool failure = true;
      MatchSiftData(siftData[i], siftData[j]);

      float correctRatio = mag_ratio_lookup(i, j);
      int iters = 20;
      for (int z = 0; z < iters; ++z) {
        float homography[9];
        int numMatches;
        auto start = std::chrono::high_resolution_clock::now();
        FindHomography(siftData[i], homography, &numMatches, 100000, scoreMin, ambiguityMax, 5.0);
        fhTime += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
        float scale = (homography[0] + homography[4]) / 2.f;
        if (fabs(scale - correctRatio) < 0.05 * homography[0]) {
          failure = false;

          std::cout << "i=" << i << " j=" << j << " correct in " << z << " iters" << std::endl;
          break;
        }
        if (z < iters - 1) {
          auto start1 = std::chrono::high_resolution_clock::now();
          shuffle_sift_data(siftData[i]);
          fhTime += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start1).count();
        }
      }
      if (failure) {
        std::cout << "i=" << i << " j=" << j << " failed" << std::endl;
      }
    }
  }
  std::cout << "fhTime "<<fhTime << std::endl;
  int k = 0;
  return true;
}

void onGroupCallbackTest(const std::string &dir, const std::array<std::string, 5> &files) {
  std::vector<cuda::GpuMat> gpuMats(5);
  std::vector<SiftData> siftDatas;
  for (int i = 0; i < files.size(); ++i) {
    loadRawToGpuGray(files[i], 6464, 4852, gpuMats[i]);
  }
  verify_group_given_params(gpuMats, 0.0, 0.4f, 0.1f, 0.9, 0.96, 0.8, 100000);
  int k = 0;
}

int main(int argc, char **argv) {
  std::string dir = argv[1];
  groupMagnificationsPerCase(dir, onGroupCallbackTest);
}
