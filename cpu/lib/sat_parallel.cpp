#include <sat.h>
#include <pool.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <atomic>
#include <unistd.h>
#include <chrono>
#include <atomic>


using namespace std;

#pragma push_macro("cerr")
#undef cerr
#define cerr if (false) std::cerr

using Clause = vector<int>; // positive literal -> true, negative literal -> false
using Formula = vector<Clause>;

int MIN_DEPTH = 0;
int MAX_DEPTH = 100000;

atomic<long long> totalUnitNs{0}, totalPureNs{0}, totalCopyNs{0}, totalSubmitNs{0}, totalSpinNs{0}, totalWorkNs;
inline auto now() { return std::chrono::high_resolution_clock::now(); }

/**
 * Check if the DPLL algorithm has finished. It is finished if:
 *  1. there is a clause that evaluates to false (and all literals are assigned within the clause) or
 *  2. all clauses evaluate to true
 * 
 * Arguments:
 *  - formula: the CNF formula
 *  - assignment: the literal assignment
 *  - satisfied: the satisfiability of the CNF (if finished)
 * Returns: whether the DPLL algorithm has terminated
 */
bool dpllFinished_parallel(const Formula &formula, Assignment &assignment, bool* satisfied) {
    bool allClausesTrue = true;

    // inspect each clause
    for (const auto &clause : formula) {
        bool clauseTrue      = false;
        bool clauseHasUnseen = false;

        // inspect each literal
        for (int lit : clause) {
            int var = std::abs(lit);
            litAssign val = assignment[var];

            // unassigned literal seen
            if (val == litAssign::UNASSIGNED) {
                clauseHasUnseen = true;
            }
            else {
                bool litValue = (val == litAssign::TRUE);
                // if literal sign matches assigned value, clause is true
                if ((lit > 0 && litValue) ||
                    (lit < 0 && !litValue))
                {
                    clauseTrue = true;
                    break;
                }
            }
        }

        // clause isn’t currently true
        if (!clauseTrue) {
            // fully assigned and no literal is true -> UNSAT and done
            if (!clauseHasUnseen) {
                *satisfied = false;
                return true;
            }

            // not yet true, but has unassigned vars -> not done
            allClausesTrue = false;
        }
    }

    if (allClausesTrue) {
        // every clause already has a true literal -> SAT and done
        *satisfied = true;
        return true;
    }

    // not done
    return false;
}

/**
 * Returns the literal contained in the first unit clause found.
 * 
 * Arguments:
 *  - formula: the CNF formula
 * Returns: literal contained within unit clause, otherwise 0
 */
int findUnitClause_parallel(const Formula &formula, Assignment &assignment) {
    // inspect each clause
    for (const auto &clause : formula) {
        int unassignedCount = 0;
        int lastUnassignedLit = 0;
        bool clauseSatisfied = false;

        // inspect each literal
        for (int lit : clause) {
            litAssign val = assignment[abs(lit)];
            if (val == litAssign::UNASSIGNED) {
                unassignedCount++;
                lastUnassignedLit = lit;
                if (unassignedCount > 1) {
                    break;
                }
            }
            else {
                // if the literal is true, the clause is satisfied and thus not unit
                bool litVal = (val == litAssign::TRUE);
                if ((lit > 0 && litVal) || (lit < 0 && !litVal)) {
                    clauseSatisfied = true;
                    break;
                }
            }
        }

        if (clauseSatisfied) {
            continue;
        }
        // return unassigned literal found in unit clause
        if (unassignedCount == 1) {
            return lastUnassignedLit;
        }
    }
    return 0;
}

/**
 * Returns the first pure literal if found.
 * 
 * Arguments:
 *  - formula: the CNF formula
 * Returns: the pure literal if found, otherwise 0
 */
int findPureLiteral_parallel(const Formula &formula, Assignment &assignment) {
    unordered_map<int, pair<bool,bool>> seen;
    seen.reserve(formula.size() * 2);

    // collect polarity info on all literals
    for (const auto &clause : formula) {
        for (int lit : clause) {
            int var = std::abs(lit);
            if (assignment[var] != litAssign::UNASSIGNED)
                continue;

            auto &pol = seen[var];
            if (lit > 0) pol.first = true;   // positive occurrence
            else        pol.second = true;   // negative occurrence
        }
    }

    // find literals that only appeared in one polarity
    for (auto &kv : seen) {
        int var       = kv.first;
        bool posSeen  = kv.second.first;
        bool negSeen  = kv.second.second;
        if (posSeen ^ negSeen) {
            return posSeen ? var : -var;
        }
    }

    return 0;  // no unassigned pure literal found
}

/**
 * Choose a literal to assign within the formula.
 * Heuristic: select the first literal from the first non-satisifed clause.
 * 
 * Arguments:
 *  - formula: the CNF formula
 * Returns: the literal to assign, otherwise 0
 */
int chooseLiteral_parallel(const Formula &formula, Assignment &assignment) {
    for (const auto &clause : formula) {
        bool satisfied = false;

        for (int lit : clause) {
            litAssign val = assignment[abs(lit)];

            // check if the clause is satisfied
            if (val != litAssign::UNASSIGNED) {
                bool litVal = (val == litAssign::TRUE);
                if ((lit > 0 && litVal) || (lit < 0 && !litVal)) {
                    satisfied = true;
                    break;
                }
            }
        }
        if (satisfied) 
            continue;

        // clause is not yet satisfied, pick its first unassigned literal
        for (int lit : clause) {
            if (assignment[abs(lit)] == litAssign::UNASSIGNED) {
                return lit;
            }
        }
    }
    return 0; // all clauses satisfied or no unassigned literals left
}

/**
 * Structure for thread‑pool thread data
 *  - pool: the thread pool the worker belongs to
 *  - formula: the CNF formula
 *  - assignment: the literal assignments
 *  - result: satisfiability of the formula
 *  - finished: signals that result has been written
 */
struct DPLLThreadData {
    ThreadPool* pool;
    const Formula &formula;
    Assignment* assignment;
    int depth;
    bool result;
    atomic<bool> finished{false};
};

/**
 * 
 */
static void* dpllTaskWrapper(void* arg) {
    auto *d = static_cast<DPLLThreadData*>(arg);
    d->result = dpll_parallel(d->formula, *d->assignment, *d->pool, d->depth);
    d->finished.store(true, memory_order_release);
    atomic_notify_one(&d->finished);
    return nullptr;
}

/**
 * Attempts to solve a SAT formula in CNF using fork–join parallelism.
 * 
 * Arguments:
 *  - formula: the CNF formula
 *  - assignment: the satisfying assignment (if found)
 * Returns:
 *  - true if satisfiable, false otherwise
 */
bool dpll_parallel(const Formula &formula, Assignment &assignment, ThreadPool &pool, int depth) {
    /* TODO: I tried disabling the unit clause elimination and pure literal elimination
            after noticing that the unit clause + pure times scale super-linearly
            relative to the number of threads.
    */

    if (depth < MAX_DEPTH) {
        // --- Unit Propagation ---
        auto u0 = now();
        int unitLiteral = findUnitClause_parallel(formula, assignment);
        while (unitLiteral != 0) {
            // assign literal
            assignment[abs(unitLiteral)] = (unitLiteral > 0) ? litAssign::TRUE : litAssign::FALSE;
            
            unitLiteral = findUnitClause(formula, assignment);
        }
        auto u1 = now();
        totalUnitNs.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(u1 - u0).count(),
        std::memory_order_relaxed
        );

        // --- Pure Literal Elimination ---
        auto p0 = now();
        int pureLiteral = findPureLiteral(formula, assignment);
        while (pureLiteral != 0) {
            // assign literal
            assignment[abs(pureLiteral)] = (pureLiteral > 0) ? litAssign::TRUE : litAssign::FALSE;

            pureLiteral = findPureLiteral(formula, assignment);
        }
        auto p1 = now();
        totalPureNs.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(p1 - p0).count(),
            std::memory_order_relaxed
        );
    }

    // --- Stopping Condition ---
    bool satisfied;
    if (dpllFinished(formula, assignment, &satisfied)) {
        return satisfied;
    }

    // choose literal to assign
    int literal = chooseLiteral_parallel(formula, assignment);

    // check how many tasks are running
    size_t queued = pool.queuedTasks;
    size_t active = pool.activeTasks;
    size_t workers = pool.workerCount();

    // if there is at least one worker free, spawn a task
    if (depth > MIN_DEPTH && queued + active < workers) {
        // prepare positive branch
        auto tcp0 = now();
        auto *assignmentCopy = new Assignment(assignment);
        auto tcp1 = now();
        totalCopyNs.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(tcp1 - tcp0).count(),
            std::memory_order_relaxed
          );

        (*assignmentCopy)[abs(literal)] = (literal > 0) ? litAssign::TRUE : litAssign::FALSE;
        auto *posData = new DPLLThreadData{ &pool, formula, assignmentCopy, depth++, false };

        // submit positive branch as a task
        ThreadPoolTask posTask{ &dpllTaskWrapper, posData };
        auto s0 = now();
        threadPoolSubmit(&pool, posTask);
        auto s1 = now();
        totalSubmitNs.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(s1 - s0).count(),
          std::memory_order_relaxed
        );

        // do negative branch in this thread
        assignment[abs(literal)] = !(literal > 0) ? litAssign::TRUE : litAssign::FALSE;
        auto tn0 = now();
        bool negResult = dpll_parallel(formula, assignment, pool, depth++);
        auto tn1 = now();
        totalWorkNs.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(tn1 - tn1).count(),
            std::memory_order_relaxed
          );

        // check if negative branch succeeded
        if (negResult) {
            return true;
        }

        // wait for positive branch to finish
        auto w0 = now();
        posData->finished.wait(false, memory_order_acquire);

        /* TODO: the below could help performance in theory?
            instead of idle waiting for the spawned task to finish,
            we could instead try to grab work from the queue and do it ourself
        */

        // while (true) {
        //     auto expected = false;
        //     if (posData->finished.load()) break;
        //     usleep(100);
          
        //     ThreadPoolTask task;
        //     bool haveWork = false;
        //     pthread_mutex_lock(&pool.lock);
        //     if (pool.count > 0) {
        //         task = pool.tasksBuffer[pool.head];
        //         pool.head = (pool.head + 1) % pool.capacity;
        //         pool.count--;
        //         pool.activeTasks++;
        //         haveWork = true;
        //     }
        //     pthread_mutex_unlock(&pool.lock);
        
        //     if (haveWork) {
        //         task.function(task.arg);
        //         pthread_mutex_lock(&pool.lock);
        //         pool.activeTasks--;
        //         pthread_mutex_unlock(&pool.lock);
        //     }
        // }

        auto w1 = now();
        totalSpinNs.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(w1 - w0).count(),
            std::memory_order_relaxed
          );
        cerr << "submitted task finished\n";

        // check if positive branch succeeded
        if (posData->result) {
            assignment = *(posData->assignment);
            delete posData->assignment;
            delete posData;
            return true;
        }
        delete posData->assignment;
        delete posData;

        return false;
    }

    // otherwise, do both tasks in this thread
    else {
        cerr << "running sequential fallback...\n";

        // assign the literal to true
        auto c0 = now();
        auto *assignmentCopy = new Assignment(assignment);
        auto c1 = now();
        totalCopyNs.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(c1 - c0).count(),
            std::memory_order_relaxed
          );

        
        (*assignmentCopy)[abs(literal)] = (literal > 0) ? litAssign::TRUE : litAssign::FALSE;

        auto n0 = now();
        if (dpll_parallel(formula, *assignmentCopy, pool, depth++)) {
            auto n1 = now();
            totalWorkNs.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(n1 - n0).count(),
                std::memory_order_relaxed
              );

            assignment = *assignmentCopy;
            cerr << "sequential fallback finished\n";
            return true;
        }
        auto n1 = now();
        totalWorkNs.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(n1 - n0).count(),
            std::memory_order_relaxed
          );

        // assign the literal to false
        assignment[abs(literal)] = !(literal > 0) ? litAssign::TRUE : litAssign::FALSE;

        auto p0 = now();
        if (dpll_parallel(formula, assignment, pool, depth++)) {
            auto p1 = now();
            totalWorkNs.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(p1 - p0).count(),
                std::memory_order_relaxed
              );

            cerr << "sequential fallback finished\n";
            return true;
        }
        auto p1 = now();
        totalWorkNs.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(p1 - p0).count(),
            std::memory_order_relaxed
          );

        cerr << "sequential fallback finished\n";
        return false;
    }
}

#pragma pop_macro("cerr")