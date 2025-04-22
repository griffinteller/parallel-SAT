#include <sat.h>
#include <pool.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <atomic>
#include <unistd.h>

using namespace std;

using Clause = vector<int>; // positive literal -> true, negative literal -> false
using Formula = vector<Clause>;

/**
 * Propogate literal assignment to formula, returns new formula.
 * 
 * Arguments:
 *  - literal: the assigned literal
 *  - formula: the CNF formula
 * Returns: the new formula with literal assigned
 */
Formula propagateLiteral_parallel(int literal, const Formula &formula) {
    Formula newFormula;

    for (const auto &clause : formula) {
        bool clauseSatisfied = false;
        Clause newClause;

        for (int lit : clause) {
            if (lit == literal) {
                // if the clause is satisfied, we are done
                clauseSatisfied = true;
                break;
            } else if (lit == -literal) {
                // skip negated literals
                continue;
            } else {
                // add existing literal to clause
                newClause.push_back(lit);
            }
        }
        if (!clauseSatisfied) {
            // add unsatisfied clauses to formula
            newFormula.push_back(newClause);
        }
    }
    return newFormula;
}

/**
 * Returns the literal contained in the first unit clause found.
 * 
 * Arguments:
 *  - formula: the CNF formula
 * Returns: literal contained within unit clause, otherwise 0
 */
int findUnitClause_parallel(const Formula &formula) {
    for (const auto &clause : formula) {
        if (clause.size() == 1) {
            return clause[0];
        }
    }
    return 0; // no literal found
}

/**
 * Returns the first pure literal if found.
 * 
 * Arguments:
 *  - formula: the CNF formula
 * Returns: the pure literal if found, otherwise 0
 */
int findPureLiteral_parallel(const Formula &formula) {
    // list all literals
    unordered_map<int, bool> literalSet;
    for (const auto &clause : formula) {
        for (int lit : clause) {
            literalSet[lit] = true;
        }
    }
    // look for pure literal
    for (auto &entry : literalSet) {
        int lit = entry.first;
        if (literalSet.find(-lit) == literalSet.end()) {
            return lit;
        }
    }
    return 0; // pure literal not found
}

/**
 * Choose a literal to assign within the formula.
 * Heuristic: select the first literal from the first non-empty clause.
 * 
 * Arguments:
 *  - formula: the CNF formula
 * Returns: the literal to assign, otherwise 0
 */
int chooseLiteral_parallel(const Formula &formula) {
    for (const auto &clause : formula) {
        if (!clause.empty()) {
            return clause[0];
        }
    }
    return 0; // no literals left to assign
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
    ThreadPool pool;
    Formula formula;
    unordered_map<int, bool> assignment;
    bool result;
    atomic<bool> finished{false};
};

/**
 * 
 */
static void* dpllTaskWrapper(void* arg) {
    auto *d = static_cast<DPLLThreadData*>(arg);
    d->result = dpll_parallel(d->formula, d->assignment, d->pool);
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
bool dpll_parallel(Formula formula, unordered_map<int, bool> &assignment, ThreadPool &pool) {
    // --- Unit Propagation ---
    int unitLiteral = findUnitClause_parallel(formula);
    while (unitLiteral != 0) {
        // assign literal
        assignment[abs(unitLiteral)] = (unitLiteral > 0);
        // propogate literal
        formula = propagateLiteral_parallel(unitLiteral, formula);
        // find unit clause
        unitLiteral = findUnitClause_parallel(formula);
    }

    // --- Pure Literal Elimination ---
    int pureLiteral = findPureLiteral_parallel(formula);
    while (pureLiteral != 0) {
        // assign literal
        assignment[abs(pureLiteral)] = (pureLiteral > 0);
        // propogate literal
        formula = propagateLiteral_parallel(pureLiteral, formula);
        // find pure literal
        pureLiteral = findPureLiteral_parallel(formula);
    }

    // --- Stopping Conditions ---
    // if the formula is empty, the formula is satisfiable
    if (formula.empty()) {
        return true;
    }
    // if any clause is empty, the formula is unsatisfiable
    for (const auto &clause : formula) {
        if (clause.empty()) {
            return false;
        }
    }

    int literal = chooseLiteral_parallel(formula);

    // prepare positive branch
    DPLLThreadData posData{ pool, formula, assignment, false };
    posData.formula = propagateLiteral_parallel(literal, formula);
    posData.assignment[abs(literal)] = (literal > 0);

    // submit positive branch as a task
    ThreadPoolTask posTask;
    posTask.function = &dpllTaskWrapper;
    posTask.arg = &posData;
    threadPoolSubmit(&pool, posTask);

    // do negative branch in this thread
    unordered_map<int, bool> negAssign = assignment;
    Formula negFormula = propagateLiteral_parallel(-literal, formula);
    negAssign[abs(literal)] = !(literal > 0);
    bool negResult = dpll_parallel(negFormula, negAssign, pool);

    // wait for positive branch to finish
    while (!posData.finished.load()) {
        sched_yield();
    }

    if (posData.result) {
        assignment = posData.assignment;
        return true;
    }
    if (negResult) {
        assignment = negAssign;
        return true;
    }
    return false;
}