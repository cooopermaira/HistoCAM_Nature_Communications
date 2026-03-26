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
  //cv::cuda::cvtColor(dRaw, dGray, bayerCode);

  // Return
  //outGray = dGray; // shallow copy of GpuMat header (data stays on GPU)
  outGray = dRaw;
  return true;
}


int extractMagnification(const std::string &path) {
  std::regex re("([0-9]+)x");
  std::smatch match;
  if (std::regex_search(path, match, re)) {
    return std::stoi(match[1].str());
  }
  return -1; // fallback if no match
}

/*
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
*/
// int main(int argc, char **argv) {
//   std::string dir = argv[1];
//   groupMagnificationsPerCase(dir, onGroupCallbackTest);
// }

int main(int argc, char **argv) {
  std::string file = argv[1];
  pathCam::Image img(6464,4852,2100);
  img.set_disk_file(Poco::Path(file));
  img.load_raw_from_disk();
  Mat raw8(img.height,img.width,CV_8UC1,img.get_Raw());
  Mat rawDebayer,raw32,view;
  cvtColor(raw8,rawDebayer,COLOR_BayerBG2GRAY);

  rawDebayer.convertTo(raw32,CV_32F);

  int k = 0;
}