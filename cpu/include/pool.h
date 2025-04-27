#pragma once

#include <pthread.h>
#include <queue>
#include <vector>
#include <atomic>

using namespace std;

extern atomic<long long> totalLockNs;

/**
 * Stucture for thread pool tasks
 *  - function: task function
 *  - arg: arguments for task function
 */
struct ThreadPoolTask {
    void* (*function)(void*);
    void* arg;
};

/**
 * Structure for thread pool
 *  - lock: mutex for the thread pool
 *  - cond: wake up condition for worker threads
 *  - tasks: task FIFO
 *  - workers: worker thread pool
 *  - stop: reap condition
 *  - workerCount: returns the number of worker threads
 */
struct ThreadPool {
    pthread_mutex_t lock;
    pthread_cond_t  cond;
    bool            stop;
    size_t          queuedTasks;
    size_t          activeTasks;
    vector<pthread_t> workers;

    deque<ThreadPoolTask> tasks;

    size_t workerCount() const { return workers.size(); }
};

/**
 * Worker thread function.
 */
void* threadPoolWorker(void* arg);

/**
 * Initialize the thread pool.
 * 
 * Arguments:
 *  - pool: the thread pool
 *  - numWorkers: number of worker threads to spawn
 */
void threadPoolInit(ThreadPool* pool, int numWorkers);

/**
 * Submits a task to the thread pool.
 * 
 * Arguments:
 *  - pool: the thread pool
 *  - task: the task to submit
 */
void threadPoolSubmit(ThreadPool* pool, ThreadPoolTask task);

/**
 * Cleans up a thread pool.
 * 
 * Arguments:
 *  - pool: the pool to remove
 */
void threadPoolDestroy(ThreadPool* pool);