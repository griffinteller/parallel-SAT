#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <chrono>

#include "sat.h"
#include "pool.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char **argv) {
    bool parallel = false;
    int numThreads = 0;
    int argIndex = 1;
    
    // usage: solver [-P] <benchmark_file_path>
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " [-P <numThreads>] <benchmark_file_path>" << endl;
        return 1;
    }
    if (string(argv[argIndex]) == "-P") {
        parallel = true;
        argIndex++;
        if (argc <= argIndex) {
            cerr << "Error: Number of threads missing after -P" << endl;
            return 1;
        }
        numThreads = std::atoi(argv[argIndex]);
        argIndex++;
    }
    if (argc <= argIndex) {
        cerr << "Error: Benchmark file path missing" << endl;
        return 1;
    }

    string file_path = argv[argIndex];
    ifstream infile(file_path);
    if (!infile) {
        cerr << "Error: Unable to open file " << file_path << endl;
        return 1;
    }

    // parse CNF file to generate formula
    Formula formula;
    string line;
    bool pLineFound = false;
    while (getline(infile, line)) {
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
            vector<int> clause;
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

    // create thread pool
    ThreadPool pool;
    if (parallel) {
        threadPoolInit(&pool, numThreads);
    }

    unordered_map<int, bool> assignment;

    auto start = steady_clock::now();
    bool result;
    
    if (parallel) {
        cout << "Running parallel algorithm..." << endl;
        result = dpll_parallel(formula, assignment, pool);
    } else {
        cout << "Running sequential algorithm..." << endl;
        result = dpll(formula, assignment);
    }

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