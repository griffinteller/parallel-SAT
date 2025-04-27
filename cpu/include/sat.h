#pragma once

#include <vector>
#include <unordered_map>
#include <atomic>

#include "pool.h"

using namespace std;

enum litAssign {
    UNASSIGNED  = 0,
    TRUE        = 1,
    FALSE       = 2,
};

using Clause = vector<int>; // positive literal -> true, negative literal -> false
using Formula = vector<Clause>;
using Assignment = vector<litAssign>;

extern atomic<long long> totalUnitNs, totalPureNs, totalCopyNs, totalSubmitNs, totalSpinNs, totalWorkNs;

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
bool dpllFinished(const Formula &formula, Assignment &assignment, bool* satisfied);
bool dpllFinished_parallel(const Formula &formula, Assignment &assignment, bool* satisfied);


/**
 * Returns the literal contained in the first unit clause found.
 * 
 * Arguments:
 *  - formula: the CNF formula
 * Returns: literal contained within unit clause, otherwise 0
 */
int findUnitClause(const Formula &formula, Assignment &assignment);
int findUnitClause_parallel(const Formula &formula, Assignment &assignment);

/**
 * Returns the first pure literal if found.
 * 
 * Arguments:
 *  - formula: the CNF formula
 * Returns: the pure literal if found, otherwise 0
 */
int findPureLiteral(const Formula &formula, Assignment &assignment);
int findPureLiteral_parallel(const Formula &formula, Assignment &assignment);

/**
 * Choose a literal to assign within the formula.
 * Heuristic: select the first literal from the first non-satisifed clause.
 * 
 * Arguments:
 *  - formula: the CNF formula
 * Returns: the literal to assign, otherwise 0
 */
int chooseLiteral(const Formula &formula, Assignment &assignment);
int chooseLiteral_parallel(const Formula &formula, Assignment &assignment);

/**
 * Attempts to solve a SAT formula in CNF.
 * 
 * Arguments:
 *  - formula: the CNF formula
 *  - assignment: the satisfying assignment
 * Returns:
 *  - true if satisfiable, false otherwise
 *  - satisfying assignment, if found
 */
bool dpll(const Formula &formula, Assignment &assignment);
bool dpll_parallel(const Formula &formula, Assignment &assignment, ThreadPool &pool, int depth);