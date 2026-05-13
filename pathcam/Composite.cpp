//
//  Composite.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/24/23.
//

#include <stdio.h>
#include <pathCam.h>

#include <memory>

namespace pathCam {


  double Composite::get_scale() const {
    return imagePyramid->scale;
  }

  void Composite::ff_correct_existing_tiles() {
    assert(flatfieldKnown);

    Mat oneChannel32f, oneChannel8u;
    std::vector<Mat> ffVec, bgraVec;

    // split flatfield into channels
    split(ff, ffVec);

    for (auto &p: imagePyramid->liveTiles) {
      auto tileObj = imagePyramid->level[0]->getTile(p.x, p.y);

      if (!tileObj->owner) {
        tileObj.reset();
      } else {
        auto abc = tileObj->owner->regInfo->absoluteCoords;

        Rect imageRect(abc, imageSize);
        Rect tileRect(p.x * imagePyramid->tile_size,
                      p.y * imagePyramid->tile_size,
                      imagePyramid->tile_size,
                      imagePyramid->tile_size);

        auto ffRoi = imageRect & tileRect;

        ffRoi.x -= abc.x;
        ffRoi.y -= abc.y;

        // split BGRA tile
        split(tileObj->image, bgraVec);

        for (int i = 0; i < 3; ++i) {
          bgraVec[i].convertTo(oneChannel32f, CV_32F);

          divide(oneChannel32f,
                 ffVec[i](ffRoi),
                 oneChannel32f);

          oneChannel32f.convertTo(bgraVec[i], CV_8U);
        }

        merge(bgraVec, tileObj->image);

        imagePyramid->level[0]->tileUpwards(
          p,
          tileRect,
          tileObj,
          Rect(0, 0,
               imagePyramid->tile_size,
               imagePyramid->tile_size));
      }
    }

    parent->update_observers();
  }

  void Composite::set_offset(const Point2f &_offset) const {
    imagePyramid->set_offset(_offset);
  }

  void Composite::correct_offset() const {
    if (xcRegLandmark && xcRegLandmark->regInfo && xcRegLandmark->regInfo->wasAligned) {
      auto queryAbC = xcPwDist + Point2f(xcRegLandmark->regInfo->absoluteCoords);
      auto resultantPoint = parent->get_AbC_relative_from_relative(xcRegLandmark->regInfo->component_membership,
                                                                   queryAbC, 0);

      set_offset(resultantPoint / get_scale());
    }
  }


  void Composite::set_scale(float _scale, bool _ffCorrectExistingTiles) {
    imagePyramid->set_scale(_scale);
    Poco::FastMutex::ScopedLock lock(update_mutex);
    deduce_label();

    if (componentMagLabel > 0) {
      for (auto &img: memberFrames) {
        if (!img->labelObserved) {
          img->label = componentMagLabel;
        }
      }
    }

    if (_ffCorrectExistingTiles) {
      ff_correct_existing_tiles();
    }
    imagePyramid->set_mag_label(componentMagLabel);
    parent->MRImageSet->sort_by_scale();
  }


  void Composite::deduce_label() {
    if (root->labelObserved) {
      componentMagLabel = root->label;
      get_flatfield();
      return;
    }
    if (componentIndex == 0) {
      componentMagLabel = parent->initialLabel;
      get_flatfield();
      return;
    }

    if (imagePyramid->scale == 0) {
      componentMagLabel = 0;
      return;
    }

    if (parent->composites[0]->componentMagLabel == 0) {
      return;
    }

    float initialComponentTrueScale = parent->labelScales[parent->composites[0]->componentMagLabel];
    float selfTrueScale = imagePyramid->scale * initialComponentTrueScale;

    float closest = parent->labelScales[0];
    float minDiff = std::abs(selfTrueScale - parent->labelScales[0]);

    // Find the closest value
    for (int i = 1; i < parent->labelScales.size(); i++) {
      float diff = std::abs(selfTrueScale - parent->labelScales[i]);
      if (diff < minDiff) {
        minDiff = diff;
        closest = parent->labelScales[i];
      } else {
        componentMagLabel = i - 1;
        get_flatfield();
        return;
      }
    }
    componentMagLabel = parent->labelScales.size() - 1;
    get_flatfield();
  }

  void Composite::get_flatfield() {
    set_candidate_scale_ratios();

    // get flatfield file name from parent and load from disk
    bool ffAlreadySet;
    std::string filename = parent->get_flatfield_path(componentMagLabel, ffAlreadySet);

    if (ffAlreadySet) {
      ff = parent->get_flatfield(componentMagLabel); // you may need to implement this
      flatfieldKnown = true;
      return;
    }

    size_t width = parent->image_width;
    size_t height = parent->image_height;
    size_t nBytes = width * height;

    std::ifstream stream(filename, std::ios::binary);
    if (!stream) {
      throw std::runtime_error("Failed to open flatfield file: " + filename);
    }

    // allocate CPU mat directly
    Mat rawMat(height, width, CV_8U);

    if (!stream.read(reinterpret_cast<char *>(rawMat.data), nBytes)) {
      throw std::runtime_error("Failed to read flatfield data");
    }

    // debayer → 3 channel
    Mat threeChannel;
    cvtColor(rawMat, threeChannel, COLOR_BayerBG2BGR);

    // convert to float
    threeChannel.convertTo(ff, CV_32FC3);

    // scale
    float scale = 1.0f / 240.0f;
    multiply(ff, Scalar(scale, scale, scale), ff);

    // cache in parent (CPU version)
    parent->set_flatfield(componentMagLabel, ff);
#ifdef PATHCAM_OPENCV_CUDA
    if (parent->CompositeType == _CompositeVoronoi) {
      ffGPU.upload(ff);
    }
#endif
    flatfieldKnown = true;
  }

  void Composite::establish_scale_at_root_cpu(Image *_rootImg) {
    _rootImg->load_raw_from_disk(true);
    if (_rootImg->akazeFeatures.empty()) {
      make_akaze(_rootImg,{0.25,0.1});
    }

    Poco::RWLock::ScopedReadLock lock(parent->component_mutex);

    //find most recent resolved frame
    if (auto [mostRcntRslv,objChange] = parent->get_most_recent_resolved_frame(_rootImg, false);
      mostRcntRslv) {
      if (!objChange) {
        std::cout << "no objective change detected for component " << componentIndex << std::endl;
        //were probably still in the same component and couldn't match in matchRunnable due to blurry sequence.
        //_rootImg may overlap with a different component. Find this region and calculate overlaps

        Rect regionInMySpace;
        if (auto [mostRcntRslv2,objChange2] = parent->get_most_recent_resolved_frame(mostRcntRslv, false);
          mostRcntRslv2 && mostRcntRslv2->regInfo->component_membership == mostRcntRslv->regInfo->
          component_membership) {
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


        OrderedSet<Image *> targets;
        targets.insert(mostRcntRslv);
        int i = 0;
        while (targets.values().size() < 3) {
          targets.insert(overlappingFrames[i++].first);
        }

        auto likelyLabel = mostRcntRslv->get_label();
        // Mat rootRaw(imageSize,CV_8UC1, _rootImg->get_Raw());
        int count = 0;


        for (auto target: targets.values()) {
          std::cout << "registration attempt " << count++ << " frame " << target->index << std::endl;

          if (target->akazeFeatures.empty()) {
            make_akaze(target,{0.25,0.1});
          }

          //target and root are swapped in this function call because we know the scales for target but not for root
          if (auto res = findHomographyAKAZE_allScalePairs(target->akazeFeatures, _rootImg->akazeFeatures); res.valid) {

            //detect scale difference between likelyLabel and img.label
            auto targetScales = Image::valid_scales_for_label(target->get_label());
            auto scaleDiff = targetScales[likelyLabel - 1];

            auto hx = res.H.at<double>(0, 0);
            auto hy = res.H.at<double>(1, 1);
            auto rsx = abs(res.H.at<double>(0, 1));
            auto rsy = abs(res.H.at<double>(1, 0));

            if (abs(hx - scaleDiff) < 0.05 * scaleDiff && abs(hy - scaleDiff) < 0.05 * scaleDiff &&
                rsx < 0.01 && rsy < 0.01) {
              //valid homography
              float relativeScale = (hx + hy) / 2;

              //get component of matched-to frame
              auto theirComponentIndex = target->regInfo->component_membership;
              auto theirComponent = parent->composites[theirComponentIndex];

              //calculate absolute coordinates of _rootImg in their component space
              Point2f pairwiseDistance = Point2f(-res.H.at<double>(0, 2), -res.H.at<double>(1, 2));

              if (abs(relativeScale - 1.f) < 0.05) {
                // Same scale — keep as independent component. CMS will find shared frames;
                // CompositeManager::combine_components() will join later.
                Point2f theirAbC(target->regInfo->absoluteCoords.x, target->regInfo->absoluteCoords.y);
                Point2f queryAbC = pairwiseDistance + theirAbC;
                auto resultantPoint = parent->get_AbC_relative_from_relative(theirComponentIndex, queryAbC, 0);
                double scale = relativeScale * theirComponent->get_scale();
                assert(scale > 0);
                set_scale(scale, !flatfieldKnown);
                set_offset(resultantPoint / scale);
                xcPwDist = pairwiseDistance;
                xcRegLandmark = target;
                std::cout << "component " << componentIndex
                          << " is same-scale; remaining independent for CMS joining" << std::endl;
                _rootImg->free_memory_RAW();
                return;
              }
            }
          }
        }

        // AKAZE failed to confirm match, but no objective change detected. Use velocity-projected
        // position to anchor the component; stay independent for CMS joining later.
        auto theirComponentIndex = mostRcntRslv->regInfo->component_membership;
        auto theirComponent = parent->composites[theirComponentIndex];
        auto projectedAbC = Point2f(regionInMySpace.tl());
        auto resultantPoint = parent->get_AbC_relative_from_relative(theirComponentIndex, projectedAbC, 0);
        double scale = theirComponent->get_scale();
        assert(scale > 0);
        set_scale(scale, !flatfieldKnown);
        set_offset(resultantPoint / scale);
        xcPwDist = projectedAbC - Point2f(mostRcntRslv->regInfo->absoluteCoords);
        xcRegLandmark = mostRcntRslv;
        std::cout << "component " << componentIndex
                  << " positioned via velocity projection; remaining independent" << std::endl;

        _rootImg->free_memory_RAW();
      } else {
        //we likely changed objective lens so attempt to match against most recent resolved

        // mostRcntRslv->load_raw_from_disk(true);
        // Mat targetRaw(imageSize,CV_8UC1, mostRcntRslv->get_Raw());
        // Mat rootRaw(imageSize,CV_8UC1, _rootImg->get_Raw());
        if (mostRcntRslv->akazeFeatures.empty()) {
          make_akaze(mostRcntRslv,{0.25,0.1});
        }

        if (auto res = findHomographyAKAZE_allScalePairs(_rootImg->akazeFeatures, mostRcntRslv->akazeFeatures); res.
          valid) {
          bool validHomography = false;
          std::vector<float> homography(9);

          auto matchedComp = parent->composites[mostRcntRslv->regInfo->component_membership];
          auto targetScales = Image::valid_scales_for_label(mostRcntRslv->get_label());
          auto hx = res.H.at<double>(0, 0);
          auto hy = res.H.at<double>(1, 1);

          if (_rootImg->labelObserved) {
            //set against known scale diff
            float scale = targetScales[_rootImg->get_label() - 1];
            auto hx = res.H.at<double>(0, 0);
            auto hy = res.H.at<double>(1, 1);
            if (abs(scale - hx) < 0.05 * scale && abs(scale - hy) < 0.05 * scale) {
              //everything checks out
              validHomography = true;
              homography[0] = hx;
              homography[4] = hy;
              homography[2] = res.H.at<double>(0, 2);
              homography[5] = res.H.at<double>(1, 2);
            }
          } else {
            for (auto scale: targetScales) {
              auto v1 = abs(scale - hx);
              auto v2 = abs(scale - hy);
              if (v1 < 0.05 * scale && v2 < 0.05 * scale) {
                validHomography = true;
                homography[0] = hx;
                homography[4] = hy;
                homography[2] = res.H.at<double>(0, 2);
                homography[5] = res.H.at<double>(1, 2);
                break;
              }
            }
          }

          if (validHomography) {
            float relativeScale = (homography[0] + homography[4]) / 2;

            //calculate relative coordinates of _rootImg in their component space
            Point2f pairwiseDistance = Point2f(homography[2], homography[5]);

            // Same scale in objective-change branch — stay independent; CMS will handle joining.

            //get matched-to frames absolute coordinates
            Point2f theirAbC(mostRcntRslv->regInfo->absoluteCoords.x, mostRcntRslv->regInfo->absoluteCoords.y);

            Point2f queryAbC = pairwiseDistance + theirAbC;

            //convert queryAbC to base component spce
            auto resultantPoint = parent->get_AbC_relative_from_relative(matchedComp->componentIndex, queryAbC, 0);

            //
            double scale = relativeScale * matchedComp->get_scale();

            assert(scale > 0);
            set_scale(scale, !flatfieldKnown);
            set_offset(resultantPoint / scale);

            xcPwDist = pairwiseDistance;
            xcRegLandmark = mostRcntRslv;
            matchedComp->add_landmark_frame(mostRcntRslv);
            std::cout << "component " << componentIndex << " XC registered" << std::endl;

            _rootImg->free_memory_RAW();
            return;
          }
          // else {
          //   if (!_rootImg->labelObserved)
          // }
        }

        if (mostRcntRslv->labelObserved) {
          //get matched-to frames absolute coordinates
          Point2f theirAbC(mostRcntRslv->regInfo->absoluteCoords.x, mostRcntRslv->regInfo->absoluteCoords.y);
          auto theirComponentIndex = mostRcntRslv->regInfo->component_membership;
          auto theirComponent = parent->composites[theirComponentIndex];

          auto scale = Image::get_mpp(_rootImg->label) / Image::get_mpp(mostRcntRslv->label);


          //calculate absolute coordinates of _rootImg in their component space
          auto px = imageSize.width * (1.f - scale) / 2.f;
          auto py = imageSize.height * (1.f - scale) / 2.f;
          Point2f pairwiseDistance = Point2f(px, py);
          Point2f queryAbC = pairwiseDistance + theirAbC;

          //convert queryAbC to base component spce
          auto resultantPoint = parent->get_AbC_relative_from_relative(theirComponentIndex, queryAbC, 0);

          scale *= theirComponent->get_scale();
          assert(scale > 0);
          set_scale(scale, !flatfieldKnown);
          set_offset(resultantPoint / scale);

          xcPwDist = pairwiseDistance;
          xcRegLandmark = mostRcntRslv;
          theirComponent->add_landmark_frame(mostRcntRslv);
          std::cout << "component " << componentIndex << " XC registered by label based guess" << std::endl;

          _rootImg->free_memory_RAW();
          return;
        }
        std::cout << "UNABLE TO DETERMINE SCALE FOR COMPONENT " << componentIndex << std::endl;
        _rootImg->free_memory_RAW();
      }
    }
  }


  void Composite::set_candidate_scale_ratios() {
    switch (componentMagLabel) {
      case Image::_2X:
        candidateScaleRatios = {1.0, 2.0, 5.0, 10.0, 20.0};
        break;
      case Image::_4X:
        candidateScaleRatios = {0.5, 1.0, 2.5, 5.0, 10.0};
        break;
      case Image::_10X:
        candidateScaleRatios = {0.2, 0.4, 1.0, 2.0, 4.0};
        break;
      case Image::_20X:
        candidateScaleRatios = {0.1, 0.2, 0.5, 1.0, 2.0};
        break;
      case Image::_40X:
        candidateScaleRatios = {0.05, 0.1, 0.25, 0.5, 1.0};
        break;
      default:
        throw std::runtime_error("unknown component mag label");
    }
  }

  void Composite::sort_overlaps_by_likelihood(std::vector<std::pair<pathCam::Image *, cv::Rect> > &_overlaps,
                                              const float &_targetScale) {
    std::sort(_overlaps.begin(), _overlaps.end(),
              [this, _targetScale](const auto &a, const auto &b) {
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

  std::vector<std::pair<Image *, Image *> > Composite::calculate_member_overlaps(std::vector<Image *> images) const {
    if (images.empty()) {
      images = std::vector(contributingImages.begin(), contributingImages.end());
    }
    std::vector<std::pair<Image *, Image *> > results;

    Point2i mDistance;
    int sqScopeRad = parent->scope_radius * parent->scope_radius * 0.7;
    for (int i = 0; i < images.size() - 1; ++i) {
      for (int j = i + 1; j < images.size(); ++j) {
        mDistance = images[i]->regInfo->absoluteCoords - images[j]->regInfo->absoluteCoords;

        if (componentMagLabel == Image::_2X) {
          if (mDistance.x * mDistance.x + mDistance.y * mDistance.y < sqScopeRad) {
            results.emplace_back(images[i], images[j]);
          }
        } else {
          if (abs(mDistance.x) < (1 - parent->crop_factor) * 0.9 * imageSize.width &&
              abs(mDistance.y) < (1 - parent->crop_factor) * 0.9 * imageSize.height) {
            results.emplace_back(images[i], images[j]);
          }
        }
      }
    }
    return results;
  }

  Composite::~Composite() {

#ifdef PATHCAM_OPENCV_CUDA
    cudaFree(threeChnBuf);
    cudaFree(fourChnBuf);
    cudaFree(rectMaskBuf);
#endif

    delete ftg;
  }

  void Composite::realtime_alignment_thread_loop() {

    std::vector<Image*> batch;
    batch.reserve(256);

    while (true) {

      realTimeAlignmentEvent.wait();

      batch.clear();

      {
        Poco::Mutex::ScopedLock lock(realTimeAlignmentMutex);

        while (!realTimeAlignmentQueue.empty()) {
          batch.push_back(realTimeAlignmentQueue.front());
          realTimeAlignmentQueue.pop();
        }

        // queue empty now
        realTimeAlignmentEvent.reset();
      }

      if (!batch.empty()) {
        realtime_align(batch);
      }

      // -----------------------------------------
      // only exit AFTER draining remaining work
      // -----------------------------------------

      if (!alignmentShouldProceed) {

        Poco::Mutex::ScopedLock lock(realTimeAlignmentMutex);

        if (realTimeAlignmentQueue.empty()) {
          break;
        } else {
          // more work arrived while aligning
          realTimeAlignmentEvent.set();
        }
      }
    }
  }

  void Composite::realtime_align(std::vector<Image *> images) {
    auto startf = std::chrono::high_resolution_clock::now();
    for (auto & img : images) {
      prep_image_for_alignment(img);
    }
    realTimeImageList.insert(realTimeImageList.end(),images.begin(),images.end());
    // auto t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startf).count();
    //
    // startf = std::chrono::high_resolution_clock::now();
    // update_mutex.lock();
    // auto imageList = find_contributing_images(true);
    // update_mutex.unlock();
    //
    // imageList.insert(root);
    // imageList.insert(images.begin(),images.end());
    //
    // std::vector imageListVec(imageList.begin(),imageList.end());
    // auto t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startf).count();
    // startf = std::chrono::high_resolution_clock::now();
    //
    //
    // auto ans = get_match_candidates(imagePyramid->bounds,memberFrames.size(),imageListVec);
    // imageListVec.insert(imageListVec.end(),ans.first.begin(),ans.first.end());
    //
    // auto t2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startf).count();
    startf = std::chrono::high_resolution_clock::now();


    std::vector<Observation*> observations;
    // observations.reserve(imageListVec.size() * 600);
    observations.reserve(realTimeImageList.size() * 600);

    for (auto &img : realTimeImageList) {
      for (auto &obs : img->observations) {
        if (obs->feature->find()->active && obs->feature->find()->live) {
          observations.push_back(obs);
        }
      }
    }

    auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startf).count();
    startf = std::chrono::high_resolution_clock::now();

    ftg->launch_inprocess_sparse_CG_iterator(observations);

    auto t4 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startf).count();
    std::cout <<"preproc time 1, 2, 3: "<</*t<<" , "<<t1<<" , "<<t2<<*/" , "<<t3 <<" solve time "<<t4<<" queue size " << images.size() << std::endl;


    // update coordinates on all regInfo objects before returning.
  }


  void Composite::prep_image_for_alignment(Image *img) const {
    Point2i coords;

    ftg->add_image(img);
    if (img->regInfo && img->regInfo->winningVote.m) {
      coords = img->regInfo->absoluteCoords;

      Poco::FastMutex::ScopedLock lock(ftg->accessMutex);
      ftg->process_match2(img->regInfo->winningVote.m);
      auto [c,valid] = ftg->estimate_image_coords_from_feature_tracks(img);
      coords = c;

      if (valid) {
        auto [matchCandidates,featTracksCovered] = get_match_candidates(Rect(coords,imageSize),4,{img->regInfo->winningVote.m->image_1}, img);

        if (!matchCandidates.empty()) {
          auto matches = pairwise_match(img,matchCandidates);

          for (auto &m : matches){
            ftg->process_match2(m);
          }
        }

        Poco::Mutex::ScopedLock lock(img->regInfo->rAccessMutex);
        img->regInfo->absoluteCoords = coords;
      }else {
        int k = 0;
      }
    }else {
      int k = 0;
    }
  }

  std::vector<std::shared_ptr<Match>> Composite::pairwise_match(Image *img, const std::vector<Image *> &targets) const {
    if (targets.empty()) {
      return {};
    }
    auto& matcher = getThreadLocalMatcher(parent->matcher_type);
    std::vector<std::shared_ptr<Match> > matches;

    for (auto &candidate : targets) {
      if (candidate == nullptr || candidate->index == img->index) {continue;}
      if (!candidate->is_good()) {continue;}
      if (img->label != Image::_NOLABEL && candidate->label != Image::_NOLABEL && img->label != candidate->label){continue;}

      auto m = std::make_shared<Match>(candidate, img);
      matcher.match(m);

      if (1 == MotionEstimator::findHomography(m, parent->estimator_type, 10)) {
        m->numMatches = std::accumulate(m->inliers.begin(), m->inliers.end(), 0);
        matches.push_back(m);
      }
    }
    return matches;
  }

  std::pair<std::vector<Image *>, int> Composite::get_match_candidates(const Rect &rect, const int n, const std::vector<Image *> &alreadyMatched, Image *self) const {
    std::vector<BAFeature*> features;

    // ---- query grid ----
    ftg->featureGrid.query(rect.x,
               rect.y,
               rect.x + rect.width,
               rect.y + rect.height,
               [&](BAFeature* f) {
                   features.push_back(f);
               });

    int radSq = parent->scope_radius * parent->scope_radius;
    Point2i center = rect.tl() + Point2i(rect.size()) / 2;

    features.erase(
        std::remove_if(features.begin(), features.end(), [&](BAFeature* feat) {
            if (componentMagLabel == Image::_2X) {
              feat = feat->find();
                auto p = center - Point2i(feat->x, feat->y);
                auto v = p.dot(p);
                return v > radSq;
            }
          return !rect.contains(Point2f(feat->x,feat->y));
        }),
        features.end()
    );


    const int F = (int)features.size();

    // ---- build image → feature list ----
    std::unordered_map<Image*, std::vector<int>> imageToFeatures;

    for (int i = 0; i < F; ++i) {
        BAFeature* f = features[i];

        for (const auto& [img, feat_id] : f->imageFeatures) {
            imageToFeatures[img].push_back(i);
        }
    }

    // ---- mark already covered features ----
    std::vector<char> covered(F, 0);
    int totalCovered = 0;

    for (Image* img : alreadyMatched) {
        auto it = imageToFeatures.find(img);
        if (it == imageToFeatures.end()) continue;

        for (int idx : it->second) {
            if (!covered[idx]) {
                covered[idx] = 1;
                totalCovered++;
            }
        }
    }

    // ---- remove alreadySelected images from candidates ----
    for (Image* img : alreadyMatched) {
        imageToFeatures.erase(img);
    }
    if (self) {
      imageToFeatures.erase(self);
    }
    // ---- greedy selection ----
    std::vector<Image*> selected;
    selected.reserve(n);

    for (int iter = 0; iter < n; ++iter) {
        Image* bestImg = nullptr;
        int bestGain = 0;

        for (auto& [img, featIdxs] : imageToFeatures) {
            int gain = 0;

            for (int idx : featIdxs) {
                if (!covered[idx]) gain++;
            }

            if (gain > bestGain) {
                bestGain = gain;
                bestImg = img;
            }
        }

        if (!bestImg || bestGain == 0)
            break;

        selected.push_back(bestImg);

        // mark newly covered features
        for (int idx : imageToFeatures[bestImg]) {
            if (!covered[idx]) {
                covered[idx] = 1;
                totalCovered++;
            }
        }

        imageToFeatures.erase(bestImg);
    }

    return {selected, totalCovered};
  }

  void Composite::launch_component_match_search(Image *img, std::vector<Image*> candidates_) {
    if (!img->subsequentMatchLaunched) {
      // img->load_raw_from_disk(false); //freed in ComponentMatchSearch::run()
      img->subsequentMatchLaunched = true;
      ++outstandingCMS_jobs;
      const auto cms = new ComponentMatchSearch(parent, img, this,candidates_);
      parent->jqSecondary->add_runnable(cms);
    }
  }

  bool Composite::prepare_4CPA_cpu(Image *img, const std::vector<Point2i> &affectedTiles, const bool forceFullImage) {
    if (affectedTiles.size() < 100 && !forceFullImage) {
      bool ans = false;

      img->regInfo->accessMutex.lock();
      auto AbC = img->regInfo->absoluteCoords;
      img->regInfo->accessMutex.unlock();

      Rect imageBoxCompSpace(AbC, imageSize);
      img->buffer_mutex.lock();
      for (auto &tile: affectedTiles) {
        Rect tileBoxCompSpace(parent->tileSize * tile, Size(parent->tileSize, parent->tileSize));

        auto intersectionInCompSpace = tileBoxCompSpace & imageBoxCompSpace;

        if (intersectionInCompSpace.empty()) { continue; }

        auto intersectionInImageSpace = intersectionInCompSpace - AbC;

        ans = ans || prepare_4CPA_cpu(img, intersectionInImageSpace);
      }
      img->buffer_mutex.unlock();
      return ans;
    }

    img->buffer_mutex.lock();
    bool ans = prepare_4CPA_cpu(img);
    img->buffer_mutex.unlock();
    return ans;
  }

  bool Composite::prepare_4CPA_cpu(Image *img, Rect roi_) {
    assert(roi_.x >= 0 && roi_.y >= 0);
    bool wholeImage = false;

    try {
      if (roi_.width * roi_.height == 0) {
        roi_ = Rect(0, 0, imageSize.width, imageSize.height);
        wholeImage = true;
      }

      assert(img->get_Raw());

      adjust_roi_for_debayer(roi_);
      Mat rawMat(imageSize, CV_8U, img->get_Raw());
      cvtColor(rawMat(roi_), threeChannelPreallocated(roi_), COLOR_BayerBG2BGR);

      if (convertHolding.empty()) {
        convertHolding = Mat(imageSize,CV_32FC3);
      }
      threeChannelPreallocated(roi_).convertTo(convertHolding(roi_),CV_32F);
      if (flatfieldKnown) {
        divide(convertHolding(roi_), ff(roi_), convertHolding(roi_), 1,CV_32F);
      }

      convertHolding(roi_).convertTo(threeChannelPreallocated(roi_), CV_8UC3);

      //add alpha
      split(threeChannelPreallocated(roi_), channels);
      channels.push_back(rectMask(roi_));
      merge(channels, fourChannelPreallocated(roi_));
    } catch (cv::Exception &e) {
      std::cout << e.what() << std::endl;
    }
    return wholeImage;
  }

  void Composite::calculate_effected_tiles_round(std::vector<Point2i> maskAsPolygon, std::vector<Point2i> &result,
                                                 Point2f absCoord) {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<Point2i> tileIndices;
    std::map<int, std::vector<float> > tilesByColumn;

    //get the tile column of the left and right edges of the image frame
    int columnBoundLow = imagePyramid->level[0]->getIJ(Point2f(absCoord.x, absCoord.y)).x;
    int columnBoundHigh = imagePyramid->level[0]->getIJ(
      Point2f(absCoord.x + imageSize.width, absCoord.y)).x;

    //get the tile row the top and bottom edges of the image frame
    long rowBoundLow = imagePyramid->level[0]->getIJ(Point2f(absCoord.x, absCoord.y)).y;
    long rowBoundHigh = imagePyramid->level[0]->getIJ(
      Point2f(absCoord.x, absCoord.y + imageSize.height)).y;

    float yPixelBoundLow = float(rowBoundLow) * float(imagePyramid->tile_size);
    float yPixelBoundHigh = float(rowBoundHigh) * float(imagePyramid->tile_size);

    //for each edge of the voronoi mask
    for (int ii = 0; ii < maskAsPolygon.size(); ii++) {
      //if we are at the last point, make the next point the first point (this makes the last edge)
      int ii2 = (ii + 1) == maskAsPolygon.size() ? 0 : ii + 1;

      //create an Point2f from the cv::Point2i
      auto p1 = Point2f(maskAsPolygon[ii].x + absCoord.x, maskAsPolygon[ii].y + absCoord.y);
      auto p2 = Point2f(maskAsPolygon[ii2].x + absCoord.x, maskAsPolygon[ii2].y + absCoord.y);

      //test for duplicate points that result from the voronoi calculation
      if (p1.x == p2.x && p1.y == p2.y) {
        continue;
      }

      //retreive the tiles these points fall within
      auto tile1 = imagePyramid->level[0]->getIJ(p1);
      auto tile2 = imagePyramid->level[0]->getIJ(p2);

      //get the column of these tiles
      int column1 = tile1.x;
      int column2 = tile2.x;

      //determine which column is on the right and which is on the left
      int xlow = min(column1, column2);
      int xhigh = max(column1, column2);

      //take the column range bounds to be the most inward of the frame edges and the voronoi cell vertices
      //this accomplishes the same thing as taking the polygon intersection of the voronoi face and the image frame
      if (columnBoundHigh < xlow || columnBoundLow > xhigh) {
        continue; //no intersection with frame tiles
      }
      xlow = max(columnBoundLow, xlow);
      xhigh = min(columnBoundHigh, xhigh);

      auto plow = p1.x < p2.x ? p1 : p2;
      auto phigh = p1.x >= p2.x ? p1 : p2;

      for (float j = xlow; j <= xhigh; j++) {
        float columnLeftEdge = j * float(imagePyramid->level[0]->getTileSize());
        float columnRightEdge = (j + 1) * float(imagePyramid->level[0]->getTileSize());

        float xloclow = max(columnLeftEdge, plow.x);
        float xlochigh = min(columnRightEdge, phigh.x);

        long enterColumn = segment_yval_at_point(xloclow, p1, p2);
        long exitColumn = segment_yval_at_point(xlochigh, p1, p2);

        tilesByColumn[j].push_back((float) enterColumn);
        tilesByColumn[j].push_back((float) exitColumn);
      }
    }

    int scopeRadSqr = pow(parent->scope_radius, 2);
    for (auto &[key, value]: tilesByColumn) {
      float firstPoint_y = *std::min_element(tilesByColumn[key].begin(), tilesByColumn[key].end());
      float lastPoint_y = *std::max_element(tilesByColumn[key].begin(), tilesByColumn[key].end());

      if (lastPoint_y < yPixelBoundLow || firstPoint_y > yPixelBoundHigh) {
        continue;
      }

      int lastTile = imagePyramid->level[0]->getIJ(
        Point2f(key * imagePyramid->level[0]->getTileSize(), lastPoint_y)).y;
      int firstTile = imagePyramid->level[0]->getIJ(
        Point2f(key * imagePyramid->level[0]->getTileSize(), firstPoint_y)).y;

      for (int ii = firstTile; ii <= lastTile; ii++) {
        if (ii <= rowBoundHigh && ii >= rowBoundLow) {
          //make sure at least one corner is closer than scope radius from center
          for (int i = 0; i < 4; i++) {
            int x = key * imagePyramid->tile_size + (i % 2 == 0 ? imagePyramid->tile_size : 0);
            int y = ii * imagePyramid->tile_size + (i % 3 == 0 ? imagePyramid->tile_size : 0);
            auto dist = pow(absCoord.x + imageSize.width / 2 - x, 2) + pow(
                          absCoord.y + imageSize.height / 2 - y, 2);
            if (dist < scopeRadSqr) {
              result.push_back(Point2i(key, ii));
              if (key < minTilex) { minTilex = key; }
              if (key > maxTilex) { maxTilex = key; }
              if (ii < minTiley) { minTiley = ii; }
              if (ii > maxTiley) { maxTiley = ii; }
              break;
            }
          }
        }
      }
    }
  }


  void Composite::save_pyramid_as_image(std::string _fileName, bool _withGrid, bool _withGridAndIndexes,
                                        bool _withEffectedTiles, bool _outline,
                                        std::vector<Point2i> effectedTiles) {
    int lineThickness = 40;
    auto level = imagePyramid->level[0];

    //    auto ul = Point2i(minTilex, minTiley);
    //    auto lr = Point2i(maxTilex, maxTiley);
    auto ul = imagePyramid->level[0]->getIJ(Point2f(root_offset.x, root_offset.y));
    auto lr = imagePyramid->level[0]->getIJ(Point2f(max_offset.x, max_offset.y));

    int tile_size = level->getTileSize();


    int width = (lr.x + 1 - ul.x) * tile_size;
    int height = (lr.y + 1 - ul.y) * tile_size;

    width = std::abs(width);
    height = std::abs(height);

    Size pyramidSize = Size(width, height);
    Mat pyramidImage = Mat(pyramidSize, CV_8UC4);

    //we need to shift the x and y tiles so we aren't writing to negtive coordinates
    int x_offset = -ul.x;
    int y_offset = -ul.y;

    Mat tile;
    Mat greyBlend = Mat(Size(tile_size, tile_size), CV_8UC4, Scalar(120, 120, 120, 255));
    for (int x = ul.x; x <= lr.x; x++) {
      for (int y = ul.y; y <= lr.y; y++) {
        try {
          if (_withEffectedTiles) {
            if (imagePyramid->level[0]->tiles(x, y) == NULL) {
              tile = Mat(Size(tile_size, tile_size), CV_8UC4, Scalar(0, 0, 0, 0));
            } else {
              tile = level->getTile(x, y)->image.clone();
            }
            auto searchForTile = Point2i(x, y);
            if (std::find(effectedTiles.begin(), effectedTiles.end(), Point2i(x, y)) != effectedTiles.
                end()) {
              tile = 0.5 * tile + 0.5 * greyBlend;
            }
          } else {
            tile = level->getTile(x, y)->image.clone();
          }

          if (_withGrid) {
            cv::line(tile, cv::Point(0, 0), cv::Point(0, tile_size - 1), Scalar(0, 0, 0, 255),
                     lineThickness);
            cv::line(tile, cv::Point(0, tile_size - 1), cv::Point(tile_size - 1, tile_size - 1),
                     Scalar(0, 0, 0, 255), lineThickness);
            cv::line(tile, cv::Point(tile_size - 1, 0), cv::Point(tile_size - 1, tile_size - 1),
                     Scalar(0, 0, 0, 255), lineThickness);
            cv::line(tile, cv::Point(tile_size - 1, 0), cv::Point(0, 0), Scalar(0, 0, 0, 255),
                     lineThickness);
          }

          if (_withGridAndIndexes) {
            putText(tile, "(" + std::to_string(x + x_offset) + "," + std::to_string(y + y_offset) + ")",
                    Point(10, 50),
                    FONT_HERSHEY_PLAIN, 3, Scalar(0, 0, 0, 255), 5);
          }

          // Mat mask;
          // cv::extractChannel(tile, mask, 3);
          // double val;
          // int maglab = componentMagLabel;
          // maglab == 1
          //   ? val = 1.0
          //   : maglab == 2
          //       ? val = 0.5
          //       : maglab == 3
          //           ? val = 0.2
          //           : maglab == 4
          //               ? val = 0.1
          //               : val = 1;
          // double beta = (log2(1.0 / val) / 3.4) * 0.7 + 0.05;
          // auto greenShade = Mat(tile.rows, tile.cols, CV_8UC3,
          //                       Scalar(200 * beta, 150 * (1 - beta), 100 * beta));
          // std::vector<Mat> v = {greenShade, mask};
          // merge(v, greenShade);
          //
          // tile = beta * greenShade + (1 - beta) * tile;

          //imwrite(std::to_string(componentIndex) + "_" + std::to_string(x) + "_" + std::to_string(y) + ".png", tile);
          tile.copyTo(pyramidImage(Rect((x + x_offset) * tile.cols, (y + y_offset) * tile.rows, tile.cols,
                                        tile.rows)));
        } catch (const cv::Exception &e) {
          auto k = e.what();
          int kk = 0;
        }
      }
    }
    if (_outline) {
      rectangle(pyramidImage, Point2i(10, 10), Point2i(pyramidImage.cols - 20, pyramidImage.rows - 20),
                Scalar(0, 0, 0, 255), 2 * lineThickness);
    }
    //currently hardcoded, maybe add an output directory in config?
    String path = "/Users/coopermaira/Desktop/pyramidImage" + std::to_string(componentIndex) + "_" +
                  std::to_string(imagePyramid->scale) +
                  ".png";
    //only write pixels with information
    //imwrite(path, pyramidImage(Rect(left_offset, top_offset, width - left_offset - right_offset, height - top_offset - bottom_offset)));
    //debug_draw_voronoi(pyramidImage,subdiv,true);
    if (_fileName != "") {
      imwrite(_fileName, pyramidImage);
    } else {
      imwrite(path, pyramidImage);
    }
  }


  Composite::Composite(StreamCam *parent, Size image_size, int _componentIndex) : parent(parent),
    componentIndex(_componentIndex), imageSize(image_size),
    root_offset(0.0, 0.0),
    max_offset(0.0, 0.0), ftg(new FeatureTrackGenerator), realTimeAlignmentEvent(false) {
    //flat_field = parent->flat_field2X;

    imagePyramid = std::make_shared<MRTiledImage>(parent);
    std::shared_ptr<TiledImage> current = std::make_shared<TiledImage>(
      imagePyramid, parent->tileSize, parent->tileSize,
      0);
    imagePyramid->level.push_back(current);
    imagePyramid->componentIndex = _componentIndex;
    parent->MRImageSet->add(imagePyramid);
    imagePyramid->MRImageSet = parent->MRImageSet;

#ifdef PATHCAM_OPENCV_CUDA
    cudaMallocManaged(&rectMaskBuf, imageSize.area());
    cudaMemset(rectMaskBuf, 255, imageSize.area());
    rectMask = Mat(image_size, CV_8UC1, rectMaskBuf);
    rectMaskGPU = cuda::GpuMat(imageSize,CV_8UC1, rectMaskBuf);



    cudaMallocManaged(&threeChnBuf, 3 * imageSize.area());
    threeChannelPreallocated = Mat(imageSize,CV_8UC3, threeChnBuf);
    threeChannelPrealGPU = cuda::GpuMat(imageSize,CV_8UC3, threeChnBuf);

    cudaMallocManaged(&fourChnBuf, 4 * imageSize.area());
    fourChannelPreallocated = Mat(imageSize,CV_8UC4, fourChnBuf);
    fourChannelPrealGPU = cuda::GpuMat(imageSize,CV_8UC4, fourChnBuf);
#else
    threeChannelPreallocated = Mat(imageSize,CV_8UC3);
    fourChannelPreallocated = Mat(imageSize,CV_8UC4);
    rectMask = Mat(image_size, CV_8UC1,Scalar(255));
#endif
    if (parent->circleMask.empty()) {
      circleMask = Mat::zeros(image_size, CV_8U);
      circle(circleMask, Point(image_size.width / 2, image_size.height / 2), parent->scope_radius,
             Scalar(255),
             -1);
    }

    realtimeAlignmentThread = std::thread(&Composite::realtime_alignment_thread_loop, this);
  }


  void Composite::update_Bbox_no_composite(std::vector<RegInfo *> new_info) {
    bool update_box = false;

    //root_offset is the distance from (0,0) of the cv image to the root frame, which is (0,0) in registration space. max_offset is the distance from (0,0) in registration space to the bottom right corner of the cv image. Total dimensions of image are max_offset - root_offset.

    // If any new frames extend beyond the current extent, expand cv image dimensions
    for (int i = 0; i < new_info.size(); i++) {
      if (new_info[i]->absoluteCoords.x < root_offset.x) {
        update_box = true;
        root_offset.x = new_info[i]->absoluteCoords.x;
      }
      if (new_info[i]->absoluteCoords.y < root_offset.y) {
        update_box = true;
        root_offset.y = new_info[i]->absoluteCoords.y;
      }

      if (new_info[i]->absoluteCoords.x + parent->image_width > max_offset.x) {
        update_box = true;
        max_offset.x = new_info[i]->absoluteCoords.x + parent->image_width;
      }
      if (new_info[i]->absoluteCoords.y + parent->image_height > max_offset.y) {
        update_box = true;
        max_offset.y = new_info[i]->absoluteCoords.y + parent->image_height;
      }
    }
    if (update_box) {
      auto topLevelBeforeAdding = imagePyramid->level.back();
      unsigned int tile_size = imagePyramid->tile_size;
      unsigned int topLogicSize = imagePyramid->level.back()->getLogicSize();
      bool addedLevel = false;
      bool canResize = std::log2(tile_size) - imagePyramid->level.size() >= 2;

      while (canResize && (topLogicSize < max_offset.x - root_offset.x || topLogicSize < max_offset.y -
                           root_offset.y)) {
        addedLevel = true;
        unsigned int logic_size = 2 * topLogicSize;
        int levelWithinPyramid = imagePyramid->level.size();
        assert(pow(2, levelWithinPyramid) == logic_size / tile_size);
        imagePyramid->level.push_back(
          std::make_shared<TiledImage>(imagePyramid, tile_size, logic_size, levelWithinPyramid));
        topLogicSize = logic_size;
        canResize = std::log2(tile_size) - imagePyramid->level.size() >= 2;
      }
      if (addedLevel) {
        auto tl = topLevelBeforeAdding->getIJ(Point2f(root_offset.x, root_offset.y));
        auto br = topLevelBeforeAdding->getIJ(Point2f(max_offset.x, max_offset.y));
        for (int x = tl.x; x <= br.x; x++) {
          for (int y = tl.y; y <= br.y; y++) {
            if (topLevelBeforeAdding->tiles(x, y) != nullptr) {
              auto tile = Point2i(x, y);
              auto myLevelRegion = cv::Rect_<float>(tile.x * tile_size, tile.y * tile_size, tile_size,
                                                    tile_size);
              topLevelBeforeAdding->tileUpwards(tile, myLevelRegion,
                                                topLevelBeforeAdding->getTile(tile.x, tile.y),
                                                Rect(0, 0, tile_size, tile_size));
            }
          }
        }
      }
    }
  }


  void Composite::update_Bbox(std::vector<RegInfo *> new_info) {
    bool update_box = false;

    //root_offset is the distance from (0,0) of the cv image to the root frame, which is (0,0) in registration space. max_offset is the distance from (0,0) in registration space to the bottom right corner of the cv image. Total dimensions of image are max_offset - root_offset.
    Point2i temp_offset = root_offset;

    // If any new frames extend beyond the current extent, expand cv image dimensions
    for (int i = 0; i < new_info.size(); i++) {
      if (new_info[i]->absoluteCoords.x < root_offset.x) {
        update_box = true;
        root_offset.x = new_info[i]->absoluteCoords.x;
      }
      if (new_info[i]->absoluteCoords.y < root_offset.y) {
        update_box = true;
        root_offset.y = new_info[i]->absoluteCoords.y;
      }

      if (new_info[i]->absoluteCoords.x + 6464 > max_offset.x) {
        update_box = true;
        max_offset.x = new_info[i]->absoluteCoords.x + 6464;
      }
      if (new_info[i]->absoluteCoords.y + 4852 > max_offset.y) {
        update_box = true;
        max_offset.y = new_info[i]->absoluteCoords.y + 4852;
      }
    }

    if (update_box) {
      //if we are updating the bounding box, create a new combined image and copy old image into the correct location
      Mat4b new_combined(int(max_offset.y - root_offset.y), int(max_offset.x - root_offset.x), Vec4b(0, 0, 0, 0));
      //Mat new_combined_z_buffer = cv::Mat::zeros(cv::Size(new_combined.cols, new_combined.rows), CV_16U);

      if (composite.data) {
        Rect copyzone = Rect(temp_offset.x - root_offset.x, temp_offset.y - root_offset.y, composite.cols,
                             composite.rows);

        composite.copyTo(new_combined(copyzone));
        //composite_z_buffer.copyTo(new_combined_z_buffer(copyzone));
      }

      composite = new_combined;
      //composite_z_buffer = new_combined_z_buffer;
      unsigned int topLogicSize = imagePyramid->level.back()->getLogicSize();
      while (topLogicSize < composite.rows || topLogicSize < composite.cols) {
        unsigned int tile_size = imagePyramid->level[0]->getTileSize();
        unsigned int logic_size = 2 * topLogicSize;
        int levelWithinPyramid = imagePyramid->level.size();
        assert(pow(2, levelWithinPyramid) == logic_size / tile_size);
        imagePyramid->level.push_back(
          std::make_shared<TiledImage>(imagePyramid, tile_size, logic_size, levelWithinPyramid));
        topLogicSize = logic_size;

        if (composite.data) {
          Mat temp;
          resize(composite, temp,
                 Size(composite.cols / pow(2, levelWithinPyramid),
                      composite.rows / pow(2, levelWithinPyramid)));
          tiledImageBounds = cv::Rect_<float>(root_offset.x, root_offset.y, composite.cols, composite.rows);
          imagePyramid->level.back()->insertMat(temp, tiledImageBounds);
        }
      }
    }
  };

  void Composite::add_images(std::vector<RegInfo *> new_info) {
    std::vector<long> indexes;

    for (int i = 0; i < new_info.size(); i++) {
      indexes.push_back(new_info[i]->index);
    }

    // get a copy of references to all images at once so that only one mutex lock is needed
    std::vector<Image *> images = parent->get_image_ref(indexes);

    //this may need to be placed inside the below for loop if frames ever vary in size. For now it is here so the mask only needs to be built once
    cv::Size image_size(images[0]->width, images[0]->height);
    /*
    Mat mask = cv::Mat::zeros(cv::Size(image_size.width, image_size.height), CV_8U);

    circle(mask, cv::Point(image_size.width/2, image_size.height/2), 2190, cv::Scalar(255), -1);
    */

    for (int i = 0; i < images.size(); i++) {
      //calculate where the new image will be copied to in the composite
      Rect copyzone = Rect(new_info[i]->absoluteCoords.x - root_offset.x,
                           new_info[i]->absoluteCoords.y - root_offset.y, images[i]->width, images[i]->height);

      //calculate which pixels of the new image will be copied into the composite
      Mat use_locations = local_quality_score > composite_z_buffer(copyzone);


      if (countNonZero(use_locations) == 0) {
        continue; //not contributing, don't bother loading from disk
      }

      images[i]->load_raw_from_disk();
      Mat image_Mat = cv::Mat(image_size, CV_8U, images[i]->get_Raw(), Mat::AUTO_STEP);
      cvtColor(image_Mat, image_Mat, COLOR_BayerBG2BGR);
      cv::divide(image_Mat, flat_field, image_Mat, 1.0, CV_8U);

      //imwrite(images[i]->get_ImageFile().getBaseName()+".png", image_Mat);
      /*
      Mat temp;
      composite_z_buffer.copyTo(temp, copyzone);
          */

      /*
      Mat temp = cv::Mat::zeros(cv::Size(image_size.width, image_size.height), CV_8U);
      image_Mat.copyTo(temp,use_locations);
      imwrite(images[i]->get_ImageFile().getBaseName() + ".png", temp);
      */

      image_Mat.copyTo(composite(copyzone), use_locations);
      local_quality_score.copyTo(composite_z_buffer(copyzone), use_locations);

      images[i]->free_memory_RAW();
    }
    /*
    imshow("display",composite);
    waitKey(10);
    */
  }

  void Composite::update() {
    std::vector<RegInfo *> new_info;
    while (!staging.empty()) {
      new_info.push_back(staging.front());
      staging.pop();
    }
    update_Bbox(new_info);
    add_images(new_info);
  }

  Mat Composite::get_composite() {
    return composite;
  }

  void ImageToTileCopyRunnable::run() {
  }

  cv::Mat Composite::score_image_2X(int rows, int cols, int radius) {
    cv::Mat img = cv::Mat::zeros(cv::Size(cols, rows), CV_16U);
    //cv::Mat img = cv::Mat_<uint16_t>(rows,cols);
    double idist, jdist, rad_sq;
    rad_sq = pow(radius, 2);
    double k = 0;
    for (int i = 0; i < radius; i++) {
      for (int j = 0; j < radius; j++) {
        int16_t x = rows / 2 - radius + i + 1;
        int16_t y = cols / 2 - radius + j + 1;
        /*
        //pyramid
        if (j < i) {
            img.at<uint16_t>(x,y) = j;
            img.at<uint16_t>(img.rows - x, y) = j;
            img.at<uint16_t>(x,img.cols - y) = j;
            img.at<uint16_t>(img.rows -x, img.cols - y) = j;
        }
        else {
            img.at<uint16_t>(x, y) = i;
            img.at<uint16_t>(img.rows - x, y) = i;
            img.at<uint16_t>(x, img.cols - y) = i;
            img.at<uint16_t>(img.rows - x, img.cols - y) = i;
        }
        */

        //cone
        img.at<uint16_t>(x, y) = radius - max(sqrt(pow(img.rows / 2 - x, 2) + pow(img.cols / 2 - y, 2)), 0.0);
        img.at<uint16_t>(img.rows - x, y) =
            radius - max(sqrt(pow(img.rows / 2 - x, 2) + pow(img.cols / 2 - y, 2)), 0.0);
        img.at<uint16_t>(x, img.cols - y) =
            radius - max(sqrt(pow(img.rows / 2 - x, 2) + pow(img.cols / 2 - y, 2)), 0.0);
        img.at<uint16_t>(img.rows - x, img.cols - y) =
            radius - max(sqrt(pow(img.rows / 2 - x, 2) + pow(img.cols / 2 - y, 2)), 0.0);
      }
    }
    Mat mask = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_16U);

    circle(mask, cv::Point(img.cols / 2, img.rows / 2), parent->scope_radius, cv::Scalar(1), -1);
    cv::Mat temp = mask.mul(img);
    imwrite("mask.png", temp);
    //imwrite("img.png", img);
    return temp;
  }
}
