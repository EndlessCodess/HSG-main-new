#!/bin/bash

cmake --build build/clang --config Release --target hnsw

data=sift10M
numactl --cpunodebind=0 --localalloc \
    ./binary/release/hnsw \
    ./data/${data}/bigann_base.bvecs \
    ./data/${data}/bigann_query.bvecs \
    ./data/${data}/gnd/idx_10M.ivecs \
    ./data/${data}/reference_answer \
    ${data}

for M in 4 8 12 16 24 36 48 64 96
do
    for ef in 500
    do
        cat result/hnsw/${data}-${M}-${ef}.txt | grep hit >> result/hnsw/hnsw-${data}.txt
    done
done
