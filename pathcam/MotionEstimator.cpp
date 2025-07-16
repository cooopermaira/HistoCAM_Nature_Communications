//
//  MotionEstimator.cpp
//  pathCam
//
//  Created by Brian Summa on 11/1/22.
//

#include "pathCam.h"


namespace pathCam {

  void RegInfo::attempt_absolute_reg(bool queue_for_compositing) {
    auto them = parent->get_reg_ref(matchedTo);
    Vec2 theirAbCs;
    unsigned int componentMembership;

    if(them->get_abc(this, theirAbCs, componentMembership)){
      Vec2 myAbCs;
      myAbCs.x = relativeCoords.x + theirAbCs.x;
      myAbCs.y = relativeCoords.y + theirAbCs.y;
      set_abc(myAbCs, componentMembership, queue_for_compositing);
    }
  }

  bool RegInfo::get_abc(pathCam::RegInfo *caller, pathCam::Vec2 &_absoluteCoords, unsigned int &_componentMembership) {
    accessMutex->lock();

    if (resolved) {
      _absoluteCoords = absoluteCoords;
      _componentMembership = component_membership;
      accessMutex->unlock();
      return true;
    }

    callersWaiting.push_back(caller);
    accessMutex->unlock();
    return false;
  }

  void RegInfo::set_abc(pathCam::Vec2 _absoluteCoords, unsigned int _componentMembership, bool queue_for_compositing) {
    _absoluteCoords.x = std::round(_absoluteCoords.x);
    _absoluteCoords.y = std::round(_absoluteCoords.y);
    accessMutex->lock();
    absoluteCoords = _absoluteCoords;
    component_membership = _componentMembership;
    resolved = true;
    waitOnResolve.set();
    accessMutex->unlock();

    auto image = parent->get_image_ref(index);
    image->regInfo = this;

    bool proceed = queue_for_compositing && parent->sufficient_distance(absoluteCoords,component_membership);

    if (proceed) {
      parent->push_compositeQ(this);
    }


    for (auto cw: callersWaiting) {
      auto theirRelCoords = cw->relativeCoords;
      Vec2 theirAbCs;
      theirAbCs.x = theirRelCoords.x + absoluteCoords.x;
      theirAbCs.y = theirRelCoords.y + absoluteCoords.y;
      cw->set_abc(theirAbCs, component_membership,queue_for_compositing);
    }

    for (auto el: componentCallersWaiting) {
      double myScale;
      Point2f myOffset;
      while (!parent->get_scale_and_offset(component_membership, myScale, myOffset)) {
        Poco::Thread::sleep(50);
      }

      auto theirScale = (el.second->H.at<double>(0, 0) + el.second->H.at<double>(1, 1)) / 2.0;
      auto theirOffset = Point2f((el.second->t_x / theirScale + absoluteCoords.x + myOffset.x) / theirScale,
                                 (el.second->t_y / theirScale + absoluteCoords.y + myOffset.y) / theirScale);
      parent->set_scale_and_offset(el.first, theirScale * myScale, theirOffset);
    }


    if(!proceed){
      image->free_memory_RAW();
      return;
    }

#ifdef HAVE_OPENCV_CUDAARITHM
    image->move_buffer_to_gpu(parent->compositorCudaDevice);
#endif

    if(parent->recordingMode){
      image->write_to_path();
    }

#ifdef HAVE_OPENCV_CUDAARITHM
    //image->free_memory_RAW();
#endif
  }

  void RegInfo::set_waiting_component(unsigned int componentIndex, Match *m) {
    componentCallersWaiting.push_back({componentIndex, m});
  }

  void RegInfo::average_from_homographies(Point2f &_rootGuess, double &_scale) {
    Point2f rootGuess(0,0);
    double scale = 0;
    for (auto &guessPoint :rootHomographies) {
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
    accessMutex->lock();
    auto imP = parent->composites[component_membership]->imagePyramid;
    assert(imP->scale != 0);

    assert(parent->composites.size() - 1 >= _relativeComponentSpace);
    auto imP_R = parent->composites[_relativeComponentSpace]->imagePyramid;
    assert(imP_R->scale != 0);

    //convert absolute coordinates to base (first component) space
    auto resInBaseSpace = imP->scale * ( Point2f(absoluteCoords.x, absoluteCoords.y) + imP->offset);

    accessMutex->unlock();
    //convert to requested component space
    return  resInBaseSpace / imP_R->scale - imP_R->offset;
  }

  void RegInfo::set_AbC_local_from_relative(unsigned int _relativeComponentSpace, Point2f _AbCInRelativeSpace) {
    /*sets absolute coordinates of image in its own component space given absolute coordinates in another component's
     * space
     */
    accessMutex->lock();
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
    accessMutex->unlock();
  }



  int MotionEstimator::findHomography(pathCam::Match *m, int estimator_type, int requiredGoodMatches, int flag,
                                      double ransacReprojThreshold,
                                      int maxIters, double confidence) {


    //-- Localize the object
    std::vector<Point2f> image_1_pts;
    std::vector<Point2f> image_2_pts;
    switch (flag) {
      case 0:
        for (size_t i = 0; i < m->good_matches.size(); i++) {
          //-- Get the keypoints from the good matches
          image_1_pts.push_back(m->image_1->keypoints[m->good_matches[i].queryIdx].pt);
          image_2_pts.push_back(m->image_2->keypoints[m->good_matches[i].trainIdx].pt);
        }
        break;
      case 1:
        for (size_t i = 0; i < m->good_matches.size(); i++) {
          //-- Get the keypoints from the good matches
          image_1_pts.push_back(m->image_1->keypointsMultilevel[m->good_matches[i].queryIdx].pt);
          image_2_pts.push_back(m->image_2->keypointsMultilevel[m->good_matches[i].trainIdx].pt);
        }
        break;
      case 2:
        for (size_t i = 0; i < m->good_matches.size(); i++) {
          //-- Get the keypoints from the good matches
          image_1_pts.push_back(m->image_1->keypointsFull[m->good_matches[i].queryIdx].pt);
          image_2_pts.push_back(m->image_2->keypointsFull[m->good_matches[i].trainIdx].pt);
        }
        break;
    }

    if (image_1_pts.size() < requiredGoodMatches || image_2_pts.size() < requiredGoodMatches) {
      return -1;
    }

    m->H = cv::findHomography(image_1_pts, image_2_pts, estimator_type,
                              ransacReprojThreshold, noArray(), maxIters,
                              confidence);

    if (m->H.empty()) {
      return -2;
    }
    if (flag == 1) {

      for (int i = 0; i < m->H.rows; i++) {
        for (int j = 0; j < m->H.cols; j++) {
          std::cout << std::to_string(i) + " " + std::to_string(j) + " " + std::to_string(m->H.at<double>(i, j))
                    << std::endl;
        }
      }
    }
    auto a = m->H.at<double>(0, 0);
    auto d = m->H.at<double>(1, 1);
    auto a2 = m->H.at<double>(0, 2);
    auto d2 = m->H.at<double>(1, 2);
    m->t_x = a2 * (1.0 / m->image_2->get_reg_scale());
    m->t_y = d2 * (1.0 / m->image_2->get_reg_scale());
    m->scale = (a + d) / 2;
    if(flag == 1){
      m->t_x *= a;
      m->t_x += (m->image_2->reg_crop_initial / 4.0) * (1 - a) * m->image_2->width;

      m->t_y *= d;
      m->t_y += (m->image_2->reg_crop_initial / 4.0) * (1 - d) * m->image_2->height;
    }
    return 1;
  }

  void MotionEstimator::phaseCorrelate(pathCam::Match *m, Image *image_1, Image *image_2) {

    Point2d p = cv::phaseCorrelate(image_1->get_reg_image(), image_2->get_reg_image());
    m->t_x = p.x * (1.0 / image_2->get_reg_scale());
    m->t_y = p.y * (1.0 / image_2->get_reg_scale());

  }


};
