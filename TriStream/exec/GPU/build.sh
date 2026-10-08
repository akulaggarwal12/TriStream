#!/bin/bash
# TriStream GPU (TriWarp): build from exec/GPU
nvcc -std=c++17 -O3 -arch=${ARCH:-sm_86} -Xcompiler -fopenmp -I../../include main.cu ../../src/GPU/*.cu ../../src/GPU/*.cpp -o tristream -lgomp
