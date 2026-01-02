//
// Created by cooper maira on 9/29/25.
//
#include <pathCam.h>

namespace pathCam {
  cuda::GpuMat &getThreadConvertSpace(int width, int height);// {
  //   thread_local cuda::GpuMat buffer;
  //
  //   if (buffer.size().area() < height * width) {
  //     buffer.create(height, width,CV_32FC1);
  //   }
  //   return buffer;
  // }

  static cuda::GpuMat get_grayscale(Image * &_img) {
    cuda::GpuMat temp = getThreadConvertSpace(_img->width, _img->height);
    Size size(_img->width, _img->height);
    cuda::GpuMat image_Mat(size, CV_8U, _img->get_raw_cuda());
    image_Mat.convertTo(temp,CV_32FC1);
    return temp;
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


  bool Composite::establish_scale_between_pairs(Image *_rootImg, Image *_target) {
    //get my sift data
    _rootImg->siftMutex.lock();
    if (!_rootImg->siftInitialized) {
      _rootImg->load_raw_from_disk();

      if (!parent->unifiedMemory && !_rootImg->cudaBufferReady) {
        _rootImg->move_buffer_to_gpu(parent->compositorCudaDevice, true);
      }

      _rootImg->extract_sift(60000,4,0,0.4f,0.1f,getThreadConvertSpace(parent->siftWindow,parent->siftWindow));
      _rootImg->free_memory_RAW();
    }
    _rootImg->siftMutex.unlock();

    //get their sift data
    _target->siftMutex.lock();
    if (!_target->siftInitialized) {
      _target->load_raw_from_disk();

      if (!parent->unifiedMemory && !_target->cudaBufferReady) {
        _target->move_buffer_to_gpu(parent->compositorCudaDevice, true);
      }

      _target->extract_sift(60000,4,0,0.4,0.1f,getThreadConvertSpace(parent->siftWindow,parent->siftWindow));
      _target->free_memory_RAW();
    }
    _target->siftMutex.unlock();

    MatchSiftData(_rootImg->siftData, _target->siftData);
    std::vector<float> homography(9);
    int numMatches;
    bool validHomography = false;
    int count = 0, maxAttempts = 20;
    while (!validHomography && count < maxAttempts) {
      ++count;
      FindHomography(_rootImg->siftData, homography.data(), &numMatches, 10000, 0.8, 0.9, 5.0);

      if (numMatches > 0) {
        auto matchedComp = parent->composites[_target->regInfo->component_membership];
        for (auto scale: matchedComp->candidateScaleRatios) {
          scale = 1 / scale;
          if (abs(scale - homography[0]) < 0.05 * scale && abs(scale - homography[4]) < 0.05 * scale) {
            validHomography = true;
          }
        }
      }
      if (!validHomography && count < maxAttempts) {
        shuffle_sift_data(_rootImg->siftData);
      }
    }

    if (!validHomography) {
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
      return true;
    }
    pairwiseDistance += (1 - relativeScale) * Point2f(float(imageSize.width - parent->siftWindow) / 2.f, float(imageSize.height - parent->siftWindow) / 2.f);

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
        for (auto &[img,roi]: overlappingFrames) {
          std::cout << count++ << std::endl;
          if (establish_scale_between_pairs(_rootImg, img)) {
            break;
          }
        }
      } else {
        //we likely changed objective lens so attempt to match against most recent resolved
        establish_scale_between_pairs(_rootImg, mostRcntRslv);
      }
    }
  }
}
