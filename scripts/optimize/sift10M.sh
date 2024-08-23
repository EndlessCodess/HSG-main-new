#!/bin/bash

cmake --build build/clang --config Release --target optimize
# cmake --build build/gcc --config Debug --target optimize

data=sift10M
numactl --cpunodebind=1 --localalloc \
    ./binary/release/optimize \
    ./data/${data}/bigann_base.bvecs \
    ./data/${data}/bigann_query.bvecs \
    ./data/${data}/gnd/idx_10M.ivecs \
    ./data/${data}/reference_answer \
    ${data} \
    16 4 100 50 10
