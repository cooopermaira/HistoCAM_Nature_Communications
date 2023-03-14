//
//  process_list_truth.cpp
//  pathCam
//
//  Created by Brian Summa on 11/11/22.
//

#include "pathCam.h"
#include "Poco/Runnable.h"
#include "Poco/Thread.h"

using namespace std;

class Worker:public Poco::Runnable{
    public:
        Worker(int n):_id(n){}
        virtual void run() {
            cout << "i'm worker:" << _id << endl;
        }
    private:
        int _id;
};

int main(int argc, char **argv)
{
    Worker work1(1);
    Worker work2(2);
    
    Poco::Thread thread1;
    Poco::Thread thread2;
    
    thread1.start(work1);
    thread2.start(work2);
    
    thread1.join();
    thread2.join();
    
    return 0;
}
