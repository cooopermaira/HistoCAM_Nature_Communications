//
// Created by cooper maira on 9/29/25.
//
#include <pathCam.h>

namespace pathCam {
  // cuda::GpuMat &getThreadConvertSpace(int width, int height);// {

  void copy_sift_data(SiftData &dst, const SiftData &src)
  {
    InitSiftData(dst, src.numPts, true, true);

    const size_t bytes = static_cast<size_t>(src.numPts) * sizeof(SiftPoint);

    cudaError_t e1 = cudaMemcpy(dst.d_data, src.d_data, bytes, cudaMemcpyDeviceToDevice);
    if (e1 != cudaSuccess)
    {
      fprintf(stderr, "cudaMemcpy D2D failed: %s\n", cudaGetErrorString(e1));
      abort();
    }

    cudaError_t e2 = cudaMemcpy(dst.h_data, dst.d_data, bytes, cudaMemcpyDeviceToHost);
    if (e2 != cudaSuccess)
    {
      fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(e2));
      abort();
    }

    dst.numPts = src.numPts;
  }


#include <vector>
#include <cmath>

  inline void findHomographyInliersCPU(
      const SiftData& data,
      const float H[9],          // [0..8], with H[8]=1
      float thresh,              // same thresh you passed to FindHomography
      float minScore,
      float maxAmbiguity,
      std::vector<uint8_t>& inlierMask,  // output: size = data.numPts
      int* outInlierCount = nullptr)
  {
    const float thresh2 = thresh * thresh;
    const int n = data.numPts;

    inlierMask.assign(n, 0);
    int inliers = 0;

    for (int i = 0; i < n; ++i) {
      const SiftPoint& p = data.h_data[i];

      // Match validity: in cudasift, p.match is typically -1 when no match
      if (p.match < 0) continue;

      // Same filters FindHomography uses:
      if (p.score <= minScore) continue;
      if (p.ambiguity >= maxAmbiguity) continue;

      const float x = p.xpos;
      const float y = p.ypos;

      // Project (x,y) with H
      const float X = H[0]*x + H[1]*y + H[2];
      const float Y = H[3]*x + H[4]*y + H[5];
      const float W = H[6]*x + H[7]*y + 1.0f;

      if (W == 0.0f) continue;
      const float u = X / W;
      const float v = Y / W;

      const float dx = u - p.match_xpos;
      const float dy = v - p.match_ypos;

      const float err2 = dx*dx + dy*dy;
      if (err2 <= thresh2) {
        inlierMask[i] = 1;
        ++inliers;
      }
    }

    if (outInlierCount) *outInlierCount = inliers;
  }


  void sort_overlaps_by_likelihood(std::vector<std::pair<Image *, Rect> > &_overlaps, const float &_targetScale) {
    auto parent = _overlaps[0].first->parent;

    std::sort(_overlaps.begin(), _overlaps.end(), [_targetScale,parent](const auto &a, const auto &b) {
      float valA = parent->composites[a.first->regInfo->component_membership]->get_scale();
      float valB = parent->composites[b.first->regInfo->component_membership]->get_scale();

      float diffA = std::abs(log(valA) - log(_targetScale));
      float diffB = std::abs(log(valB) - log(_targetScale));

      if (std::abs(diffA - diffB) > 0.0001f) {
        return diffA < diffB;
      }

      return a.second.area() > b.second.area();
    });
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


  bool Composite::establish_scale_between_pairs(Image *_rootImg, Image *_target, bool _fullImageFtExtract) {
    SiftData rootCopy,compareCopy;

    //_rootImg->siftMutex.lock();

    //get my sift data
    if ((!_rootImg->siftInitialized && !_fullImageFtExtract) || (!_rootImg->siftFullInitialized && _fullImageFtExtract)) {
      _rootImg->load_raw_from_disk();

      if (!parent->unifiedMemory && !_rootImg->cudaBufferReady) {
        _rootImg->move_buffer_to_gpu(parent->compositorCudaDevice, true);
      }

      int bufW, bufH,octaves,pts;
      if (_fullImageFtExtract) {
        bufW = _rootImg->width;
        bufH = _rootImg->height;
        octaves = 5;
        pts = 100000;
      }else {
        bufW = bufH = parent->siftWindow;
        octaves = 4;
        pts = 60000;
      }
      if (cvtBuffer.rows != bufH || cvtBuffer.cols != bufW) {
        cvtBuffer = cuda::GpuMat(bufH,bufW,CV_32FC1);
      }
      _rootImg->extract_sift(pts,octaves,0,0.4f,0.1f,cvtBuffer, !_fullImageFtExtract);
      _rootImg->free_memory_RAW();
    }
    if (_fullImageFtExtract) {
      copy_sift_data(rootCopy,_rootImg->siftDataFull);
      //rootCopy = _rootImg->siftDataFull;
    }else {
      copy_sift_data(rootCopy,_rootImg->siftData); //avoids shuffling a sorted data order needed later
    }
    //_rootImg->siftMutex.unlock();


    //get their sift data
    //_target->siftMutex.lock();
    if ((!_target->siftInitialized && !_fullImageFtExtract) || (!_target->siftFullInitialized && _fullImageFtExtract)) {
      _target->load_raw_from_disk();

      if (!parent->unifiedMemory && !_target->cudaBufferReady) {
        _target->move_buffer_to_gpu(parent->compositorCudaDevice, true);
      }

      int bufW, bufH,octaves,pts;
      if (_fullImageFtExtract) {
        bufW = _target->width;
        bufH = _target->height;
        octaves = 5;
        pts = 100000;
      }else {
        bufW = bufH = parent->siftWindow;
        octaves = 4;
        pts = 60000;
      }
      if (cvtBuffer.rows != bufH || cvtBuffer.cols != bufW) {
        cvtBuffer = cuda::GpuMat(bufH,bufW,CV_32FC1);
      }
      _target->extract_sift(pts,octaves,0,0.4,0.1f,cvtBuffer, !_fullImageFtExtract);
      _target->free_memory_RAW();
    }
    if (_fullImageFtExtract) {
      copy_sift_data(compareCopy,_target->siftDataFull);
      //compareCopy = _target->siftDataFull;
    }else {
      copy_sift_data(compareCopy,_target->siftData); //avoids shuffling a sorted data order needed later
    }
    //_target->siftMutex.unlock();

    assert(rootCopy.numPts > 0 && compareCopy.numPts > 0);
    MatchSiftData(rootCopy, compareCopy);
    FreeSiftData(compareCopy);

    std::vector<float> homography(9);
    int numMatches;
    bool validHomography = false;

    int count = 0, maxAttempts = 20;
    while (!validHomography && count < maxAttempts && xcMatchShouldContinue) {
      ++count;
      FindHomography(rootCopy, homography.data(), &numMatches, 10000, 0.8, 0.9, 5.0);

      if (numMatches > 0) {
        auto matchedComp = parent->composites[_target->regInfo->component_membership];
        for (auto scale: matchedComp->candidateScaleRatios) {
          scale = 1 / scale;
          if (abs(scale - homography[0]) < 0.05 * scale && abs(scale - homography[4]) < 0.05 * scale) {
            validHomography = true;
            break;
          }
        }
      }
      if (!validHomography && count < maxAttempts && xcMatchShouldContinue) {
        shuffle_sift_data(rootCopy);
      }
    }

    if (!validHomography) {
      FreeSiftData(rootCopy);
      return false;
    }
    float relativeScale = (homography[0] + homography[4]) / 2;

    //get component of matched-to frame
    auto theirComponentIndex = _target->regInfo->component_membership;

    //get matched-to frames absolute coordinates
    Point2f theirAbC(_target->regInfo->absoluteCoords.x, _target->regInfo->absoluteCoords.y);

    //calculate absolute coordinates of _rootImg in their component space
    Point2f pairwiseDistance = Point2f(homography[2], homography[5]);

    if (abs(relativeScale - 1.f) < 0.05) {
      //we're part of this component. suspend self, create a match and attempt registration.
      suspended = true;
      imagePyramid->suspended = true;

      for (auto &p: imagePyramid->liveTiles) {
        auto tObj = imagePyramid->level[0]->getTile(p.x, p.y);
        tObj.reset();
      }

      auto myRegInfo = _rootImg->regInfo;

      myRegInfo->root = false;
      myRegInfo->matchedTo = _target->index;
      myRegInfo->relativeCoords = pairwiseDistance;
      myRegInfo->attempt_absolute_reg(true);
      std::cout << "component " << componentIndex << " suspended and joined to component "
          << _target->regInfo->component_membership << std::endl;

      std::vector<uint8_t> inlierMask;
      std::vector<KeyPoint> kp1,kp2;
      int inlierCount;

      findHomographyInliersCPU(rootCopy,homography.data(),5.f,0.8,0.9,inlierMask,&inlierCount);
      sift_to_cvMatch(rootCopy,_rootImg,_target,inlierCount,inlierMask,kp1,kp2);
      FreeSiftData(rootCopy);

      auto comp = reinterpret_cast<MetricComposite *>(parent->composites[theirComponentIndex]);
      comp->update_mutex.lock();
      comp->extraMatches.emplace_back(_rootImg,_target,kp1,kp2);
      comp->update_mutex.unlock();
      return true;
    }
    if (!_fullImageFtExtract) {
      pairwiseDistance += (1 - relativeScale) * Point2f(float(imageSize.width - parent->siftWindow) / 2.f, float(imageSize.height - parent->siftWindow) / 2.f);
    }
    Point2f queryAbC = pairwiseDistance + theirAbC;

    //convert queryAbC to base component spce
    auto resultantPoint = parent->get_AbC_relative_from_relative(theirComponentIndex, queryAbC, 0);

    //
    double scale = relativeScale * parent->composites[theirComponentIndex]->get_scale();
    _rootImg->regInfo->rootHomographies.emplace_back(resultantPoint, scale);

    assert(scale > 0);
    set_scale(scale,true);
    auto p = resultantPoint / scale;
    set_offset(resultantPoint / scale);

    std::cout<<"component "<<componentIndex<<" XC registered"<<std::endl;
    return true;
  }

  void Composite::establish_scale_at_root(Image *_rootImg) {
    //find most recent resolved frame
    if (auto [mostRcntRslv,objChange] = parent->get_most_recent_resolved_frame(_rootImg, false);
      mostRcntRslv) {
      if (!objChange) {
        std::cout << "no objective change detected for component " << componentIndex << std::endl;
        //were probably still in the same component and couldn't match in matchRunnable due to blurry sequence.
        //_rootImg may overlap with a different component. Find this region and calculate overlaps

        Rect regionInMySpace;
        if (auto [mostRcntRslv2,objChange2] = parent->get_most_recent_resolved_frame(mostRcntRslv, false);
          mostRcntRslv2 &&
          mostRcntRslv2->regInfo->component_membership
          == mostRcntRslv->regInfo->component_membership) {
          auto forwardIndexDif = float(_rootImg->index - mostRcntRslv->index);
          auto indexDif = float(mostRcntRslv->index - mostRcntRslv2->index);
          auto distance = mostRcntRslv->regInfo->absoluteCoords - mostRcntRslv2->regInfo->absoluteCoords;

          auto projectedAbC = distance * forwardIndexDif / indexDif + mostRcntRslv->regInfo->absoluteCoords;
          regionInMySpace = Rect(projectedAbC, imageSize);
        } else {
          regionInMySpace = Rect(mostRcntRslv->regInfo->absoluteCoords, imageSize);
        }
        auto overlappingFrames = parent->get_overlapping_frames(regionInMySpace,
                                                                mostRcntRslv->regInfo->component_membership);
        sort_overlaps_by_likelihood(overlappingFrames,
                                    parent->composites[mostRcntRslv->regInfo->component_membership]->get_scale());

        int count = 0;
        //for (auto &[img,roi]: overlappingFrames) {
        for (int i = 0; i < min(10,int(overlappingFrames.size())); ++i){
          auto img = overlappingFrames[i].first;
          std::cout << count++ << std::endl;
          if (establish_scale_between_pairs(_rootImg, img, true) || !xcMatchShouldContinue) {
            break;
          }
        }
      } else {
        //we likely changed objective lens so attempt to match against most recent resolved
        establish_scale_between_pairs(_rootImg, mostRcntRslv, false);
      }
    }
  }

  void Composite::sift_to_cvMatch(const SiftData &siftData, Image *image1, Image *image2, int inlierCount,
                                  const std::vector<uint8_t> &inlierMask, std::vector<
                                    KeyPoint> &keypoints1, std::vector<KeyPoint> &keypoints2) {

    keypoints1.resize(inlierCount);
    keypoints2.resize(inlierCount);
    assert((int)inlierMask.size() == siftData.numPts);

    int loc = 0;
    for (int i = 0; i < siftData.numPts; ++i) {
      if (inlierMask[i]) {
        assert(loc < inlierCount);
        keypoints1[loc].pt = Point2f(siftData.h_data[i].xpos,siftData.h_data[i].ypos);
        keypoints2[loc].pt = Point2f(siftData.h_data[i].match_xpos,siftData.h_data[i].match_ypos);
        ++loc;
      }
    }
    assert(loc == inlierCount);
  }
}
