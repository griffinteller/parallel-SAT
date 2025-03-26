#pragma once
#include <vector>
#include <cstdint>

namespace sat {

typedef int32_t LiteralId ;
typedef std::vector<LiteralId> Clause;
typedef std::vector<Clause> Formula;
typedef std::vector<LiteralId> Assignment;

struct ClauseState {
    int32_t satisfied;
    int32_t unsatisfied;
};

struct ClauseNode {
    int32_t clauseId;
    ClauseNode* next;
};

struct SatState {
    Formula formula;
    std::vector<ClauseNode> adjLists;
    std::vector<ClauseState> clauseStates;
    std::vector<LiteralId> assignment;
};

FormulaState initFormulaState(const Formula& formula);

}
