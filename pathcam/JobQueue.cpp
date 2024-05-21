//
//  JobQueue.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include "pathCam.h"

namespace pathCam {
    JobQueue::JobQueue(int min_threads, int max_threads) {
        queue_mutex = new Poco::FastMutex(),
                pool = new Poco::ThreadPool(min_threads, max_threads, 60, POCO_THREAD_STACK_SIZE);
    }

    void JobQueue::add_runnable(RunnableIntermediate *job) {
        queue_mutex->lock();
        jobQueue.push(job);
        queue_mutex->unlock();
    };

    bool JobQueue::run_jobs(bool join_all, bool order_before_run) {
        queue_mutex->lock();
        int batchSize = std::min(20, (int) jobQueue.size());
        queue_mutex->unlock();

        for (int i = 0; i < batchSize; i++) {

            if (pool->available() > 0) {
                queue_mutex->lock();
                pool->start(*jobQueue.top());
                jobQueue.pop();
                queue_mutex->unlock();
            } else {
                Poco::Thread::sleep(100);
            }
        }

        if (join_all) {
            pool->joinAll();
        }
        return true;
    };

    bool JobQueue::comp_sort_order(const RunnableIntermediate *a, const RunnableIntermediate *b) {
        unsigned long a1 = a->sort_order;
        unsigned long b1 = b->sort_order;
        bool res = a1 < b1;
        return res;
    }

    bool JobQueue::CompareRunnable::operator()(const RunnableIntermediate *a, const RunnableIntermediate *b) {
        return a->sort_order > b->sort_order;
    }
}
