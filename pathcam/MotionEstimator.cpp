//
//  MotionEstimator.cpp
//  pathCam
//
//  Created by Brian Summa on 11/1/22.
//

#include "pathCam.h"


namespace pathCam {

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
      int k = 0;
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
    m->t_x = a * a2 * (1.0 / m->image_2->get_reg_scale());
    m->t_y = d * d2 * (1.0 / m->image_2->get_reg_scale());
    m->scale = (a + d) / 2;
    return 1;
  }

  void MotionEstimator::phaseCorrelate(pathCam::Match *m, Image *image_1, Image *image_2) {

    Point2d p = cv::phaseCorrelate(image_1->get_reg_image(), image_2->get_reg_image());
    m->t_x = p.x * (1.0 / image_2->get_reg_scale());
    m->t_y = p.y * (1.0 / image_2->get_reg_scale());

  }


};
