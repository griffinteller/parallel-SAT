#include <pthread.h>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <atomic>

#include "pool.h"

#pragma push_macro("cerr")
#undef cerr
#define cerr if (false) std::cerr

atomic<long long> totalLockNs{0};
inline auto now() { return std::chrono::high_resolution_clock::now(); }

/**
 * Worker thread function.
 */
void* threadPoolWorker(void* arg) {
    ThreadPool* pool = static_cast<ThreadPool*>(arg);
    cerr << "[worker " << pthread_self() << "] started" << endl;

    while (true) {
        // acquire thread pool lock
        auto t2 = now();
        pthread_mutex_lock(&pool->lock);
        auto t3 = now();
        totalLockNs.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count(),
            std::memory_order_relaxed
        );

        // wait for work to appear
        while (pool->tasks.empty() && !pool->stop) {
            pthread_cond_wait(&pool->cond, &pool->lock);
        }

        // return if the pool is stopped
        if (pool->tasks.empty() && pool->stop) {
            pthread_mutex_unlock(&pool->lock);
            break;
        }

        cerr << "[worker " << pthread_self() << "] waking up, queue size " 
          << pool->tasks.size() << endl;

        // take work
        ThreadPoolTask task = pool->tasks.back();
        pool->tasks.pop_back();
        pool->queuedTasks--;
        pool->activeTasks++;

        cerr << "[worker " << pthread_self() << "] popped task, queue size now " 
          << pool->tasks.size() << endl;

        // release thread pool lock
        pthread_mutex_unlock(&pool->lock);
        pthread_cond_signal(&pool->cond);

        // execute function
        task.function(task.arg);

        cerr << "[worker " << pthread_self() << "] finished task" << endl;

        pthread_mutex_lock(&pool->lock);
        pool->activeTasks--;
        pthread_mutex_unlock(&pool->lock);
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
        cerr << "[pool] created worker tid: " << i << endl;
        pthread_t thread;
        pthread_create(&thread, nullptr, threadPoolWorker, pool);
        pool->workers.push_back(thread);
    }
}

/**
 * Submit a task to the thread pool.
 * Arguments:
 *  - pool: the thread pool
 *  - task: the task to submit
 */
void threadPoolSubmit(ThreadPool* pool, ThreadPoolTask task) {
    // acquire thread pool lock
    pthread_mutex_lock(&pool->lock);

    // push task onto stack
    pool->tasks.push_back(task);
    pool->queuedTasks++;

    pthread_mutex_unlock(&pool->lock);
    pthread_cond_signal(&pool->cond);

    cerr << "[submit] pushing task, queue size now " << pool->tasks.size() << endl;
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
    for (pthread_t& t : pool->workers) {
        pthread_join(t, nullptr);
    }

    pthread_mutex_destroy(&pool->lock);
    pthread_cond_destroy(&pool->cond);
}

#pragma pop_macro("cerr")
