#include <sat.h>

#include <iostream>
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
Formula propagateLiteral(int literal, const Formula &formula) {
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
int findUnitClause(const Formula &formula) {
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
int findPureLiteral(const Formula &formula) {
    // list all literals
    unordered_map<int, bool> literalSet;
    for (const auto &clause : formula) {
        for (int lit : clause) {
            // Insert literal with a dummy true value; we only care about the keys.
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
int chooseLiteral(const Formula &formula) {
    for (const auto &clause : formula) {
        if (!clause.empty()) {
            return clause[0];
        }
    }
    return 0; // no literals left to assign
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
bool dpll(Formula formula, unordered_map<int, bool> &assignment) {
    // --- Unit Propagation ---
    int unitLiteral = findUnitClause(formula);
    while (unitLiteral != 0) {
        // assign literal
        assignment[abs(unitLiteral)] = (unitLiteral > 0);
        
        // propogate literal
        formula = propagateLiteral(unitLiteral, formula);

        // find unit clause
        unitLiteral = findUnitClause(formula);
    }

    // --- Pure Literal Elimination ---
    int pureLiteral = findPureLiteral(formula);
    while (pureLiteral != 0) {
        // assign literal
        assignment[abs(pureLiteral)] = (pureLiteral > 0);

        // propogate literal
        formula = propagateLiteral(pureLiteral, formula);

        // find pure literal
        pureLiteral = findPureLiteral(formula);
    }

    // --- Stopping Conditions ---
    // if the formula is empty, the formula is satisfiable
    if (formula.empty()) {
        return true;
    }
    // if any clause is empty, the formula is not satisfiable
    for (const auto &clause : formula) {
        if (clause.empty()) {
            return false;
        }
    }

    // --- Recursion ---
    int literal = chooseLiteral(formula);

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
        assignmentCopy[abs(literal)] = false;
        if (dpll(formulaCopy, assignmentCopy)) {
            assignment = assignmentCopy;
            return true;
        }
    }
    return false;
}