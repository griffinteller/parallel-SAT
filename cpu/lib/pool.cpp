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
        while (pool->count == 0 && !pool->stop) {
            pthread_cond_wait(&pool->cond, &pool->lock);
        }

        // return if the pool is stopped
        if (pool->count == 0 && pool->stop) {
            pthread_mutex_unlock(&pool->lock);
            break;
        }

        // take work
        ThreadPoolTask task = pool->tasksBuffer[pool->head];
        pool->head = (pool->head + 1) % pool->capacity;
        pool->count--;
        pool->queuedTasks--;
        pool->activeTasks++;

        // release thread pool lock
        pthread_mutex_unlock(&pool->lock);

        // execute function
        task.function(task.arg);

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

    pool->capacity = numWorkers;
    pool->tasksBuffer = (ThreadPoolTask*)malloc(
        pool->capacity * sizeof(ThreadPoolTask)
    );
    pool->head = pool->tail = pool->count = 0;

    for (int i = 0; i < numWorkers; i++) {
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

    while (pool->count == pool->capacity && !pool->stop) {
        pthread_cond_wait(&pool->cond, &pool->lock);
    }

    // push onto work queue
    pool->tasksBuffer[pool->tail] = task;
    pool->tail = (pool->tail + 1) % pool->capacity;
    pool->count++;
    pool->queuedTasks++;

    // wake up worker thread
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
    for (pthread_t& t : pool->workers) {
        pthread_join(t, nullptr);
    }

    free(pool->tasksBuffer);
    pthread_mutex_destroy(&pool->lock);
    pthread_cond_destroy(&pool->cond);
}

#pragma pop_macro("cerr")
