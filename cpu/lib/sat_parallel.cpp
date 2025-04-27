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
#include <mutex>
#include <condition_variable>


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

atomic<bool> foundSat{false};
atomic<int> tasksInFlight{0};
mutex satMutex;
condition_variable satCv;
Assignment sat_assignment;

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
    bool result;
    atomic<bool> finished{false};
};

/**
 * 
 */
static void* dpllTaskWrapper(void* arg) {
    auto *d = static_cast<DPLLThreadData*>(arg);
    dpll_spawn(d->formula, *d->assignment, *d->pool);
    d->finished.store(true, memory_order_release);
    if (tasksInFlight.fetch_sub(1, memory_order_acq_rel) == 1) {
        sat_assignment = *(d->assignment);
        satCv.notify_all();
    }
    delete d->assignment;
    delete d;
    return nullptr;
}

void submitTask(ThreadPool &pool, ThreadPoolTask t) {
    tasksInFlight.fetch_add(1, memory_order_relaxed);
    threadPoolSubmit(&pool, t);
}

void dpll_spawn(const Formula &formula, Assignment &assignment, ThreadPool &pool) {
    // bail if SAT already found
    if (foundSat.load(std::memory_order_relaxed)) return;

    // --- Stopping Condition ---
    bool satisfied;
    if (dpllFinished_parallel(formula, assignment, &satisfied)) {
        if (satisfied) {
            foundSat.store(true, std::memory_order_relaxed);
            satCv.notify_all();
            cerr << "statisfied!" << endl;
        }
        cerr << "finished!" << endl;
        return;
    }

    cerr << "forking..." << endl;

    // --- Unit Propagation ---
    auto u0 = now();
    int unitLiteral = findUnitClause_parallel(formula, assignment);
    while (unitLiteral != 0) {
        // assign literal
        assignment[abs(unitLiteral)] = (unitLiteral > 0) ? litAssign::TRUE : litAssign::FALSE;
        
        unitLiteral = findUnitClause_parallel(formula, assignment);
    }
    auto u1 = now();
    totalUnitNs.fetch_add(
    std::chrono::duration_cast<std::chrono::nanoseconds>(u1 - u0).count(),
    std::memory_order_relaxed
    );

    // --- Pure Literal Elimination ---
    auto p0 = now();
    int pureLiteral = findPureLiteral_parallel(formula, assignment);
    while (pureLiteral != 0) {
        // assign literal
        assignment[abs(pureLiteral)] = (pureLiteral > 0) ? litAssign::TRUE : litAssign::FALSE;

        pureLiteral = findPureLiteral_parallel(formula, assignment);
    }
    auto p1 = now();
    totalPureNs.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(p1 - p0).count(),
        std::memory_order_relaxed
    );

    // choose literal to assign
    int literal = chooseLiteral_parallel(formula, assignment);

    // prepare positive branch
    auto tcp0 = now();
    auto *assignmentPos = new Assignment(assignment);
    auto tcp1 = now();
    totalCopyNs.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(tcp1 - tcp0).count(),
        std::memory_order_relaxed
        );

    (*assignmentPos)[abs(literal)] = (literal > 0) ? litAssign::TRUE : litAssign::FALSE;
    auto *posData = new DPLLThreadData{ &pool, formula, assignmentPos, false };

    // submit positive branch as a task
    ThreadPoolTask posTask{ &dpllTaskWrapper, posData };
    auto sp0 = now();
    submitTask(pool, posTask);
    auto sp1 = now();
    totalSubmitNs.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(sp1 - sp0).count(),
        std::memory_order_relaxed
    );

    // prepare negative branch
    auto tcn0 = now();
    auto *assignmentNeg = new Assignment(assignment);
    auto tcn1 = now();
    totalCopyNs.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(tcn1 - tcn0).count(),
        std::memory_order_relaxed
    );

    (*assignmentNeg)[abs(literal)] = !(literal > 0) ? litAssign::TRUE : litAssign::FALSE;
    auto *negData = new DPLLThreadData{ &pool, formula, assignmentNeg, false };

    // submit negative branch as a task
    ThreadPoolTask negTask{ &dpllTaskWrapper, negData };
    auto sn0 = now();
    submitTask(pool, negTask);
    auto sn1 = now();
    totalSubmitNs.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(sn1 - sn0).count(),
        std::memory_order_relaxed
    );

    return;
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
bool dpll_parallel(const Formula &formula, Assignment &assignment, int numThreads) {
    bool sat = false;

    // create thread pool
    ThreadPool pool;
    threadPoolInit(&pool, numThreads);

    // submit the first task
    auto *rootData = new DPLLThreadData{ &pool, formula, new Assignment(assignment), false };
    submitTask(pool, ThreadPoolTask{ &dpllTaskWrapper, rootData });

    // wait for signal that worker found SAT or search space exhausted
    unique_lock<mutex> lk(satMutex);
    satCv.wait(lk, []{ 
        return foundSat.load(memory_order_relaxed)
            || tasksInFlight.load(std::memory_order_relaxed) == 0; 
    });
    if (foundSat.load()) {
        assignment = sat_assignment;
        sat = true;
    }

    // clean up thread pool
    threadPoolDestroy(&pool);

    return sat;
}

#pragma pop_macro("cerr")