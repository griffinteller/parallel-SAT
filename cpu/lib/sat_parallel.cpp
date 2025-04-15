#include <sat.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <pthread.h>
#include <unistd.h>

using namespace std;

using Clause = vector<int>; // positive literal -> true, negative literal -> false
using Formula = vector<Clause>;

const int MAX_DEPTH = 3;

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
 * Structure for thread data
 *  - formula: the CNF formula
 *  - assignment: the literal assignments
 *  - result: satisfiability of the formula
 */
struct DPLLThreadData {
    Formula formula;
    unordered_map<int, bool> assignment;
    bool result;
    int depth;
};

void* dpllThread(void* arg) {
    DPLLThreadData* data = static_cast<DPLLThreadData*>(arg);
    data->result = dpll_parallel(data->formula, data->assignment, data->depth);
    return nullptr;
}

/**
 * Attempts to solve a SAT formula in CNF using fork–join parallelism with pthreads.
 * 
 * Arguments:
 *  - formula: the CNF formula
 *  - assignment: the satisfying assignment (if found)
 * Returns:
 *  - true if satisfiable, false otherwise
 */
bool dpll_parallel(Formula formula, unordered_map<int, bool> &assignment, int depth) {
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

    // --- Recursion using pthreads ---
    int literal = chooseLiteral(formula);

    // fall back to sequential algorithm if we have reached max recurse depth
    if (depth >= MAX_DEPTH) {
        // assign the literal to true
        {
            auto assignmentCopy = assignment;
            Formula formulaCopy = propagateLiteral(literal, formula);
            assignmentCopy[abs(literal)] = (literal > 0);
            if (dpll(formulaCopy, assignmentCopy)) {
                assignment = assignmentCopy;
                return true;
            }
        }
        // assign the literal to false
        {
            auto assignmentCopy = assignment;
            Formula formulaCopy = propagateLiteral(-literal, formula);
            assignmentCopy[abs(literal)] = !(literal > 0);
            if (dpll(formulaCopy, assignmentCopy)) {
                assignment = assignmentCopy;
                return true;
            }
        }
    }

    // data for thread with literal assignment true
    DPLLThreadData* pos_data = new DPLLThreadData;
    pos_data->formula = propagateLiteral_parallel(literal, formula);
    pos_data->assignment = assignment;
    pos_data->assignment[abs(literal)] = (literal > 0);
    pos_data->depth = depth + 1;

    // data for thread with literal assignment false
    DPLLThreadData* neg_data = new DPLLThreadData;
    neg_data->formula = propagateLiteral_parallel(-literal, formula);
    neg_data->assignment = assignment;
    neg_data->assignment[abs(literal)] = !(literal > 0);
    neg_data->depth = depth + 1;

    // spawn threads
    pthread_t pos_thread, neg_thread;
    int err;
    err = pthread_create(&pos_thread, nullptr, dpllThread, static_cast<void*>(pos_data));
    if(err) {
        cerr << "Error creating thread!: " << err << endl;
    }
    err = pthread_create(&neg_thread, nullptr, dpllThread, static_cast<void*>(neg_data));
    if(err) {
        cerr << "Error creating thread!: " << err << endl;
    }

    bool pos_done = false, neg_done = false;

    // wait for threads to finish
    while (!pos_done || !neg_done) {
        if (!pos_done) {
            if (pthread_tryjoin_np(pos_thread, nullptr) == 0) {
                pos_done = true;

                if (pos_data->result) {
                    // solution found, cancel the other thread
                    pthread_cancel(neg_thread);
                    pthread_join(neg_thread, nullptr);
                    assignment = pos_data->assignment;
                    delete pos_data;
                    delete neg_data;
                    return true;
                }
            }
        }
        if (!neg_done) {
            if (pthread_tryjoin_np(neg_thread, nullptr) == 0) {
                neg_done = true;

                if (neg_data->result) {
                    // solution found, cancel the other thread
                    pthread_cancel(pos_thread);
                    pthread_join(pos_thread, nullptr);
                    assignment = neg_data->assignment;
                    delete pos_data;
                    delete neg_data;
                    return true;
                }
            }
        }
        // usleep(100);
    }

    delete pos_data;
    delete neg_data;
    return false;
}