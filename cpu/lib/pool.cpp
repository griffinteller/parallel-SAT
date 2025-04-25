#include <pthread.h>
#include <queue>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include <pool.h>

using namespace std;

#pragma push_macro("cerr")
#undef cerr
#define cerr if (false) std::cerr

/**
 * Worker thread function.
 */
void* threadPoolWorker(void* arg) {
    ThreadPool* pool = static_cast<ThreadPool*>(arg);

    while (true) {
        // acquire thread pool lock
        pthread_mutex_lock(&pool->lock);

        // wait for work to appear
        while (pool->tasks.empty() && !pool->stop) {
            // release the lock and sleep the thread
            pthread_cond_wait(&pool->cond, &pool->lock);
        }

        // return if the pool is stopped
        if (pool->tasks.empty() && pool->stop) {
            cerr << "[worker " << pthread_self() << "] exiting\n";
            pthread_mutex_unlock(&pool->lock);
            break;
        }

        // take work
        ThreadPoolTask task = pool->tasks.front();
        pool->tasks.pop();
        pool->queuedTasks--;
        pool->activeTasks++;

        // release thread pool lock
        pthread_mutex_unlock(&pool->lock);
        
        // execute function
        cerr << "[worker " << pthread_self() << "] got a task, running...\n";
        task.function(task.arg);
        cerr << "[worker " << pthread_self() << "] task done\n";

        // pthread_mutex_lock(&pool->lock);
        pool->activeTasks--;
        // pthread_mutex_unlock(&pool->lock);
    }
    return nullptr;
}

/**
 * Initialize the thread pool.
 * 
 * Arguments:
 *  - pool: the thread pool
 *  - numWorkers: number of worker threads to spawn
 */
void threadPoolInit(ThreadPool* pool, int numWorkers) {
    pool->stop = false;
    pool->queuedTasks = 0;
    pool->activeTasks = 0;

    pthread_mutex_init(&pool->lock, nullptr);
    pthread_cond_init(&pool->cond, nullptr);

    for (int i = 0; i < numWorkers; i++) {
        pthread_t thread;
        pthread_create(&thread, nullptr, threadPoolWorker, pool);
        pool->workers.push_back(thread);
    }
}

/**
 * Submits a task to the thread pool.
 * 
 * Arguments:
 *  - pool: the thread pool
 *  - task: the task to submit
 */
void threadPoolSubmit(ThreadPool* pool, ThreadPoolTask task) {
    // acquire thread pool lock
    pthread_mutex_lock(&pool->lock);

    pool->tasks.push(task);
    pool->queuedTasks++;

    // release thread pool lock and signal
    pthread_mutex_unlock(&pool->lock);
    pthread_cond_signal(&pool->cond);
}

/**
 * Cleans up a thread pool.
 * 
 * Arguments:
 *  - pool: the pool to remove
 */
void threadPoolDestroy(ThreadPool* pool) {
    // acquire thread pool lock
    pthread_mutex_lock(&pool->lock);

    // signal worker threads to stop
    pool->stop = true;

    pthread_mutex_unlock(&pool->lock);

    // wake up and reap all worker threads
    pthread_cond_broadcast(&pool->cond);
    for (size_t i = 0; i < pool->workers.size(); i++) {
        pthread_join(pool->workers[i], nullptr);
    }
    
    pthread_mutex_destroy(&pool->lock);
    pthread_cond_destroy(&pool->cond);
}

#pragma pop_macro("cerr")