//
//  util.h
//  pathCam
//
//  Created by Brian on 3/16/23.
//

#ifndef util_h
#define util_h

namespace pathCam {

  struct PointComparator {
    bool operator()(const cv::Point2i& lhs, const cv::Point2i& rhs) const {
      if (lhs.x != rhs.x) {
        return lhs.x < rhs.x; // Compare x-coordinates
      }
      return lhs.y < rhs.y;     // Compare y-coordinates if x-coordinates are equal
    }
  };

  class Vec2 {
  public:
    double x, y;

    Vec2() : x(0.0), y(0.0) {};

    Vec2(double x, double y) : x(x), y(y) {};

    std::string toString() {
      std::stringstream ss;
      ss << "(" << x << "," << y << ")";
      return ss.str();
    }
  };

  class Bbox {
  public:
    double min_x, min_y, max_x, max_y;

    Bbox(double min_x = std::numeric_limits<double>::infinity(),
         double min_y = std::numeric_limits<double>::infinity(),
         double max_x = -std::numeric_limits<double>::infinity(),
         double max_y = -std::numeric_limits<double>::infinity()) :
        min_x(min_x), min_y(min_y), max_x(max_x), max_y(max_y) {};

    bool intersect(Bbox b) {
      return (min_x <= b.max_x && max_x >= b.min_x) &&
             (min_y <= b.max_y && max_y >= b.min_y);
    }

    cv::Rect as_cvRect() {
      return cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
    }

    double area(Bbox b) {
      double Omin_x = fmax(min_x, b.min_x);
      double Omin_y = fmax(min_y, b.min_y);
      double Omax_x = fmin(max_x, b.max_x);
      double Omax_y = fmin(max_y, b.max_y);

      //Assuming both boxes are the same size
      double one_area = (max_x - min_x) * (max_x - min_x) + (max_y - min_y) * (max_y - min_y);
      double Oarea = (Omax_x - Omin_x) * (Omax_x - Omin_x) + (Omax_y - Omin_y) * (Omax_y - Omin_y);

      return Oarea / one_area;
    }


    std::string toString() {
      std::stringstream ss;
      ss << "[" << min_x << "," << min_y << "," << max_x << "," << max_y << "]";
      return ss.str();
    }

  };


  class ThreadQueue {
  private:
    Poco::ThreadPool *pool;
    std::queue<Poco::Runnable *> jobQueue;

  public:
    ThreadQueue(int min_threads, int max_threads) {
      pool = new Poco::ThreadPool(min_threads, max_threads, 60, POCO_THREAD_STACK_SIZE);
    }

    void run_jobs(std::vector<Poco::Runnable *> jobs) {
      for (unsigned int i = 0; i < jobs.size(); i++) {
        jobQueue.push(jobs[i]);
      }

      while (!jobQueue.empty()) {
        if (pool->available() > 0) {
          pool->start(*jobQueue.front());
          jobQueue.pop();
        } else {
          Poco::Thread::sleep(100);
        }
      }

      pool->joinAll();
    }


  };

}

#endif /* util_h */
