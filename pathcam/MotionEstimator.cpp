//
//  MotionEstimator.cpp
//  pathCam
//
//  Created by Brian Summa on 11/1/22.
//

#include <filesystem>

#include "pathCam.h"


namespace pathCam {
  void RegInfo::attempt_absolute_reg(bool queue_for_compositing) {
    auto them = parent->get_reg_ref(matchedTo);
    Point2i theirAbCs;
    int componentMembership;

    if (them->get_abc(this, theirAbCs, componentMembership)) {
      Point2i myAbCs = relativeCoords + theirAbCs;
      set_abc(myAbCs, componentMembership, queue_for_compositing);
    }
  }

  bool RegInfo::poll_abc(std::shared_ptr<Match> m, Point2i &_absoluteCoords, int &_componentMembership) {
    Poco::Mutex::ScopedLock lock(rAccessMutex);

    if (resolved) {
      _absoluteCoords = absoluteCoords;
      _componentMembership = component_membership;
      return true;
    }

    pollers.push_back(m);
    return false;
  }

  bool RegInfo::get_abc(pathCam::RegInfo *caller, Point2i &_absoluteCoords, int &_componentMembership) {
    accessMutex.lock();

    if (resolved) {
      children.push_back(caller);
      _absoluteCoords = absoluteCoords;
      _componentMembership = component_membership;
      accessMutex.unlock();
      return true;
    }

    callersWaiting.push_back(caller);
    accessMutex.unlock();
    return false;
  }

  void RegInfo::vote_abc(std::shared_ptr<Match> m, Point2i _absoluteCoords, int _componentMembership) {
    {
      Poco::Mutex::ScopedLock lock(rAccessMutex);
      --outstandingPolls;

      vote v{_componentMembership, _absoluteCoords, m};
      votes.push_back(v);
    }

    if (outstandingPolls == 0 && matchSearchComplete) {
      int totalVotes = 0;
      for (const auto &vote: votes) {
        totalVotes += vote.m->inlierCount;
      }
      if (totalVotes > minimumVote) {
        count_votes();
      }
    }
  }

  void RegInfo::count_votes() {
    if (!votes.empty()) {
      Poco::ScopedReadRWLock lock(registrationProcessHalt);

      sort(votes.begin(), votes.end(),
           [](const vote &a, const vote &b) { return a.m->inlierCount > b.m->inlierCount; });

      int R = image->parent->tileSize * 2;
      int minCluster = 3;
      int bestClusterScore = -1;
      int bestClusterIndex = -1;


      for (size_t i = 0; i < votes.size(); ++i) {
        int clusterCount = 0;
        int clusterScore = 0;

        for (size_t j = 0; j < votes.size(); ++j) {
          int dx = (votes[i].abc.x - votes[i].m->t_x) - (votes[j].abc.x - votes[j].m->t_x);
          int dy = (votes[i].abc.y - votes[i].m->t_y) - (votes[j].abc.y - votes[j].m->t_y);

          if (dx * dx + dy * dy <= R * R) {
            clusterCount++;
            clusterScore += votes[j].m->inlierCount;
          }
        }

        if (clusterCount >= minCluster) {
          if (clusterScore > bestClusterScore) {
            bestClusterScore = clusterScore;
            bestClusterIndex = i;
          }
        }
      }

      if (bestClusterIndex != -1) {
        winningVote = votes[bestClusterIndex];
      } else {
        winningVote = votes[0];
      }
      auto wImg = winningVote.m->image_1;
      matchedTo = wImg->index;


      {
        Poco::Mutex::ScopedLock lock(wImg->regInfo->rAccessMutex);
        absoluteCoords = winningVote.abc - Point2i(winningVote.m->t_x, winningVote.m->t_y);
        relativeCoords = Point2i(winningVote.m->t_x, winningVote.m->t_y);
        component_membership = wImg->regInfo->component_membership;

        long dt = index - wImg->index;
        if (dt > 0) {
          Point2f dist = Point2f(relativeCoords) / static_cast<float>(dt);
          image->motionBlur = dist.dot(dist);
          if (image->motionBlur < 10) {
            Mat raw(image->height, image->width,CV_8UC1, image->get_Raw());
            Rect roi((image->width - 256) / 2, (image->height - 256) / 2, 256, 256);
            image->focusBlur = sobel_focus_green_even_odd(raw(roi));
          }
        }

        wImg->regInfo->children.push_back(this);
        wImg->regInfo->matchedBy = index;

        // std::scoped_lock lock2(wImg->blurMutex);
        // if (!wImg->blurSet) {
        //   auto dist_i = wImg->regInfo->relativeCoords + relativeCoords;
        //   long dt = index - wImg->regInfo->matchedTo;
        //
        //   if (dt > 0) {
        //     Point2f dist = Point2f(dist_i) / static_cast<float>(dt);
        //     wImg->motionBlur = dist.dot(dist);
        //     wImg->blurSet = true;
        //   }
        // }
      }
    } else {
      absoluteCoords = {0, 0};
    }

    resolved = true;
    parent->push_compositeQ(this);

    cast_requested_votes();
  }

  void RegInfo::cast_requested_votes() {
    Poco::Mutex::ScopedLock lock(rAccessMutex);
    for (auto &m: pollers) {
      m->image_2->regInfo->vote_abc(m, absoluteCoords, component_membership);
    }
  }

  void RegInfo::set_abc(Point2f _absoluteCoords, int _componentMembership, bool queue_for_compositing) {
    _absoluteCoords.x = std::round(_absoluteCoords.x);
    _absoluteCoords.y = std::round(_absoluteCoords.y);
    rAccessMutex.lock();
    absoluteCoords = _absoluteCoords;
    component_membership = _componentMembership;
    resolved = true;
    waitOnResolve.set();
    rAccessMutex.unlock();

    // image = parent->get_image_ref(index);
    image->regInfo = this;
    image->absoluteCoords = Point2i(absoluteCoords.x, absoluteCoords.y);

    bool proceed = queue_for_compositing; // && parent->sufficient_distance(absoluteCoords,component_membership);

    if (proceed) {
      tryComposite = true;
    }

    parent->push_compositeQ(this);

    for (auto &cw: callersWaiting) {
      children.push_back(cw);
    }
    callersWaiting.clear();

    for (auto &child: children) {
      auto theirRelCoords = -child->relativeCoords;
      Point2f theirAbCs;
      theirAbCs.x = theirRelCoords.x + absoluteCoords.x;
      theirAbCs.y = theirRelCoords.y + absoluteCoords.y;
      child->set_abc(theirAbCs, component_membership, queue_for_compositing);
    }


    if (!proceed) {
      //image->free_memory_RAW();
      return;
    }

#ifdef HAVE_OPENCV_CUDAARITHM
    if (!parent->unifiedMemory) {
      image->move_buffer_to_gpu(parent->compositorCudaDevice);
    }
#endif

    // if(parent->recordingMode){
    //   image->write_to_path();
    // }

#ifdef HAVE_OPENCV_CUDAARITHM
    if (!parent->unifiedMemory) {
      image->free_memory_RAW();
    }
#endif
  }

  void RegInfo::set_waiting_component(unsigned int componentIndex, Match *m) {
    componentCallersWaiting.emplace_back(componentIndex, m);
  }

  void RegInfo::average_from_homographies(Point2f &_rootGuess, double &_scale) {
    Point2f rootGuess(0, 0);
    double scale = 0;
    for (auto &guessPoint: rootHomographies) {
      rootGuess += guessPoint.first;
      scale += guessPoint.second;
    }

    auto div = static_cast<double>(rootHomographies.size());
    rootGuess /= div;
    scale /= div;

    _rootGuess = rootGuess;
    _scale = scale;
  }


  Point2f RegInfo::get_AbC_relative_from_local(unsigned int _relativeComponentSpace) {
    /*returns images coordinates in requested component space*/
    accessMutex.lock();
    auto imP = parent->composites[component_membership]->imagePyramid;
    assert(imP->scale != 0);

    assert(parent->composites.size() - 1 >= _relativeComponentSpace);
    auto imP_R = parent->composites[_relativeComponentSpace]->imagePyramid;
    assert(imP_R->scale != 0);

    //convert absolute coordinates to base (first component) space
    auto resInBaseSpace = imP->scale * (Point2f(absoluteCoords.x, absoluteCoords.y) + imP->offset);

    accessMutex.unlock();
    //convert to requested component space
    return resInBaseSpace / imP_R->scale - imP_R->offset;
  }

  void RegInfo::set_AbC_local_from_relative(unsigned int _relativeComponentSpace, Point2f _AbCInRelativeSpace) {
    /*sets absolute coordinates of image in its own component space given absolute coordinates in another component's
     * space
     */
    accessMutex.lock();
    auto imP = parent->composites[component_membership]->imagePyramid;
    assert(imP->scale != 0);

    assert(parent->composites.size() - 1 >= _relativeComponentSpace);
    auto imP_R = parent->composites[_relativeComponentSpace]->imagePyramid;
    assert(imP_R->scale != 0);

    //convert given coordinates to base space
    auto resInBaseSpace = imP_R->scale * (_AbCInRelativeSpace + imP_R->offset);

    //convert to self space
    auto resInMySpace = resInBaseSpace / imP->scale - imP->offset;

    absoluteCoords.x = resInMySpace.x;
    absoluteCoords.y = resInMySpace.y;
    accessMutex.unlock();
  }


  int MotionEstimator::findHomography(std::shared_ptr<Match> m, int estimator_type, int requiredGoodMatches,
                                      double ransacReprojThreshold,
                                      int maxIters, double confidence) {
    //-- Localize the object
    std::vector<Point2f> image_1_pts;
    std::vector<Point2f> image_2_pts;

    for (size_t i = 0; i < m->good_matches.size(); i++) {
      //-- Get the keypoints from the good matches
      image_1_pts.push_back(m->image_1->keypoints[m->good_matches[i].queryIdx].pt);
      image_2_pts.push_back(m->image_2->keypoints[m->good_matches[i].trainIdx].pt);
    }

    if (image_1_pts.size() < requiredGoodMatches || image_2_pts.size() < requiredGoodMatches) {
      return -1;
    }

    m->inliers.clear();
    m->H = cv::findHomography(image_1_pts, image_2_pts, estimator_type,
                              ransacReprojThreshold, m->inliers, maxIters,
                              confidence);

    if (m->H.empty()) {
      return -2;
    }

    auto a = m->H.at<double>(0, 0);
    auto b = m->H.at<double>(0, 1);
    auto c = m->H.at<double>(1, 0);
    auto d = m->H.at<double>(1, 1);

    auto tx = m->H.at<double>(0, 2);
    auto ty = m->H.at<double>(1, 2);

    // reject perspective terms
    const double p0 = m->H.at<double>(2, 0);
    const double p1 = m->H.at<double>(2, 1);
    const double p2 = m->H.at<double>(2, 2);

    if (std::abs(p0) > 1e-3 ||
        std::abs(p1) > 1e-3 ||
        std::abs(p2 - 1.0) > 1e-3) {
      return -1;
    }

    // reject rotation/skew
    // ideal rectilinear form:
    //
    // [ s  0  tx ]
    // [ 0  s  ty ]
    // [ 0  0   1 ]

    if (std::abs(b) > 0.05 ||
        std::abs(c) > 0.05) {
      return -1;
    }

    // reject anisotropic scaling
    if (std::abs(a - d) > 0.05) {
      return -1;
    }

    m->t_x = tx * (1.0 / m->image_2->get_reg_scale());
    m->t_y = ty * (1.0 / m->image_2->get_reg_scale());

    m->scale = (a + d) / 2.0;

    if (std::abs(m->scale - 1.0) > 0.05) {
      // multiresolution matches are not handled here
      return -1;
    }

    m->inlierCount = std::accumulate(m->inliers.begin(), m->inliers.end(), 0);
    // if (m->inlierCount < requiredGoodMatches) {
    //   return -1;
    // }
    return 1;
  }

  void MotionEstimator::phaseCorrelate(pathCam::Match *m, Image *image_1, Image *image_2) {
    Point2d p = cv::phaseCorrelate(image_1->get_reg_image(), image_2->get_reg_image());
    m->t_x = p.x * (1.0 / image_2->get_reg_scale());
    m->t_y = p.y * (1.0 / image_2->get_reg_scale());
  }
};
