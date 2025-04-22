#pragma once

#include <pthread.h>
#include <queue>
#include <vector>

using namespace std;

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
 */
struct ThreadPool {
    pthread_mutex_t lock;
    pthread_cond_t cond;
    queue<ThreadPoolTask> tasks;
    vector<pthread_t> workers;
    bool stop;
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