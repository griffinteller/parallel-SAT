#include <sat.h>

#include <iostream>
#include <vector>
#include <unordered_map>
using namespace std;

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
bool dpllFinished(const Formula &formula, Assignment &assignment, bool* satisfied) {
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
int findUnitClause(const Formula &formula, Assignment &assignment) {
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
int findPureLiteral(const Formula &formula, Assignment &assignment) {
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
int chooseLiteral(const Formula &formula, Assignment &assignment) {
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
 * Attempts to solve a SAT formula in CNF.
 * 
 * Arguments:
 *  - formula: the CNF formula
 *  - assignment: the satisfying assignment
 * Returns:
 *  - true if satisfiable, false otherwise
 *  - satisfying assignment, if found
 */
bool dpll(Formula formula, Assignment &assignment) {
    // --- Unit Propagation ---
    int unitLiteral = findUnitClause(formula, assignment);
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

    // --- Recursion ---
    int literal = chooseLiteral(formula, assignment);

    // assign the literal to true
    {
        auto assignmentCopy = assignment;
        assignmentCopy[abs(literal)] = (literal > 0) ? litAssign::TRUE : litAssign::FALSE;
        if (dpll(formula, assignmentCopy)) {
            assignment = assignmentCopy;
            return true;
        }
    }
    // assign the literal to false
    {
        auto assignmentCopy = assignment;
        assignmentCopy[abs(literal)] = !(literal > 0) ? litAssign::TRUE : litAssign::FALSE;
        if (dpll(formula, assignmentCopy)) {
            assignment = assignmentCopy;
            return true;
        }
    }
    return false;
}