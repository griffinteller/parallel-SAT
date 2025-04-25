#include <sat.h>
#include <pool.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <atomic>
#include <unistd.h>

using namespace std;

#pragma push_macro("cerr")
#undef cerr
#define cerr if (false) std::cerr

using Clause = vector<int>; // positive literal -> true, negative literal -> false
using Formula = vector<Clause>;

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
    Formula formula;
    Assignment assignment;
    bool result;
    atomic<bool> finished{false};
};

/**
 * 
 */
static void* dpllTaskWrapper(void* arg) {
    auto *d = static_cast<DPLLThreadData*>(arg);
    d->result = dpll_parallel(d->formula, d->assignment, *d->pool);
    d->finished.store(true);
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
bool dpll_parallel(Formula formula, Assignment &assignment, ThreadPool &pool) {
    // --- Unit Propagation ---
    int unitLiteral = findUnitClause_parallel(formula, assignment);
    while (unitLiteral != 0) {
        // assign literal
        assignment[abs(unitLiteral)] = (unitLiteral > 0) ? litAssign::TRUE : litAssign::FALSE;
        
        unitLiteral = findUnitClause(formula, assignment);
    }

    // --- Pure Literal Elimination ---
    int pureLiteral = findPureLiteral(formula, assignment);
    while (pureLiteral != 0) {
        // assign literal
        assignment[abs(pureLiteral)] = (pureLiteral > 0) ? litAssign::TRUE : litAssign::FALSE;

        pureLiteral = findPureLiteral(formula, assignment);
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
    if (queued + active < workers) {
        // prepare positive branch
        DPLLThreadData posData{ &pool, formula, assignment, false };
        posData.assignment[abs(literal)] = (literal > 0) ? litAssign::TRUE : litAssign::FALSE;

        // submit positive branch as a task
        ThreadPoolTask posTask;
        posTask.function = &dpllTaskWrapper;
        posTask.arg = &posData;
        threadPoolSubmit(&pool, posTask);
        cerr << "submitting task to the pool...\n";

        // do negative branch in this thread
        Assignment negAssign = assignment;
        negAssign[abs(literal)] = !(literal > 0) ? litAssign::TRUE : litAssign::FALSE;
        bool negResult = dpll_parallel(formula, negAssign, pool);

        // wait for positive branch to finish
        while (!posData.finished.load()) {
            sched_yield();
        }
        cerr << "submitted task finished\n";

        // check if positive branch succeeded
        if (posData.result) {
            assignment = posData.assignment;
            return true;
        }
        // check if negative branch succeeded
        if (negResult) {
            assignment = negAssign;
            return true;
        }
        return false;
    }

    // otherwise, do both tasks in this thread
    else {
        cerr << "running sequential fallback...\n";

        // do positive branch
        {
            auto assignmentCopy = assignment;
            assignmentCopy[abs(literal)] = (literal > 0) ? litAssign::TRUE : litAssign::FALSE;
            if (dpll_parallel(formula, assignmentCopy, pool)) {
                assignment = assignmentCopy;
                return true;
                cerr << "sequential fallback finished\n";
            }
        }

        // do negative branch
        {
            auto assignmentCopy = assignment;
            assignmentCopy[abs(literal)] = !(literal > 0) ? litAssign::TRUE : litAssign::FALSE;
            if (dpll_parallel(formula, assignmentCopy, pool)) {
                assignment = assignmentCopy;
                cerr << "sequential fallback finished\n";
                return true;
            }
        }
        cerr << "sequential fallback finished\n";
        return false;
    }
}

#pragma pop_macro("cerr")