#include <vector>
#include <unordered_map>

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
Formula propagateLiteral(int literal, const Formula &formula);
Formula propagateLiteral_parallel(int literal, const Formula &formula);

/**
 * Returns the literal contained in the first unit clause found.
 * 
 * Arguments:
 *  - formula: the CNF formula
 * Returns: literal contained within unit clause, otherwise 0
 */
int findUnitClause(const Formula &formula);
int findUnitClause_parallel(const Formula &formula);

/**
 * Returns the first pure literal if found.
 * 
 * Arguments:
 *  - formula: the CNF formula
 * Returns: the pure literal if found, otherwise 0
 */
int findPureLiteral(const Formula &formula);
int findPureLiteral_parallel(const Formula &formula);

/**
 * Choose a literal to assign within the formula.
 * Heuristic: select the first literal from the first non-empty clause.
 * 
 * Arguments:
 *  - formula: the CNF formula
 * Returns: the literal to assign, otherwise 0
 */
int chooseLiteral(const Formula &formula);
int chooseLiteral_parallel(const Formula &formula);

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
bool dpll(Formula formula, unordered_map<int, bool> &assignment);
bool dpll_parallel(Formula formula, unordered_map<int, bool> &assignment);