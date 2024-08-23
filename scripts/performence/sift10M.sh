#!/bin/bash

cmake --build build/clang --config Release --target performence

data=sift10M

LL=(32)
CR=(4)
K=10
BM=(50)

numactl --cpunodebind=0 --localalloc \
    ./binary/release/performence \
    ./data/${data}/bigann_base.bvecs \
    ./data/${data}/bigann_query.bvecs \
    ./data/${data}/gnd/idx_10M.ivecs \
    ./data/${data}/reference_answer \
    ${data} \
    "${LL[*]}" \
    "${CR[*]}" \
    "${BM[*]}" \
    ${K}


for l in ${LL[*]}
do
    for c in ${CR[*]}
    do
        for b in ${BM[*]}
        do
            cat result/HSG/${data}-${l}-${c}-${b}.txt | grep hit >> result/HSG/HSG-${data}.txt
        done
    done
done
