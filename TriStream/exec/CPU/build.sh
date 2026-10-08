#!/bin/bash
# TriStream CPU (TriSnap): build from exec/CPU
g++ -std=c++17 -O3 -march=native -fopenmp -I../../include main.cpp ../../src/CPU/*.cpp -o tristream_cpu
