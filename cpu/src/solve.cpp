#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <chrono>

#include "sat.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char **argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <benchmark_file_path>" << endl;
        return 1;
    }

    string file_path = argv[1];
    ifstream infile(file_path);
    if (!infile) {
        cerr << "Error: Unable to open file " << file_path << endl;
        return 1;
    }

    Formula formula;
    string line;
    bool pLineFound = false;

    while(getline(infile, line)) {
        if (line.empty() || line[0] == 'c') {
            continue;
        }
        if (line[0] == 'p') {
            pLineFound = true;
            continue;
        }
        if (pLineFound) {
            istringstream iss(line);
            int lit;
            Clause clause;
            while (iss >> lit) {
                if (lit == 0) {
                    break;
                }
                clause.push_back(lit);
            }
            if (!clause.empty()) {
                formula.push_back(clause);
            }
        }
    }
    infile.close();

    unordered_map<int, bool> assignment;

    auto start = steady_clock::now();
    bool result = dpll(formula, assignment);
    auto end = steady_clock::now();

    duration<double> elapsed = end - start;

    cout << "The formula is " << (result ? "SATISFIABLE." : "UNSATISFIABLE.") << endl;
    if (result) {
        cout << "Satisfying assignment:" << endl;
        for (const auto &entry : assignment) {
            cout << "   Variable " << entry.first << " = " 
                 << (entry.second ? "true" : "false") << endl;
        }
    }

    bool expected;
    if (file_path.find("uuf") != string::npos) {
        expected = false;
    }
    else {
        expected = true;
    }

    if (result == expected) {
        cout << "TEST PASSED" << endl;
    } else {
        cout << "TEST FAILED" << endl;
    }
    cout << "Elapsed time: " << elapsed.count() << " seconds" << endl;

    return 0;
}