#!/bin/bash

cmake --build build/clang --config Release --target performence

data=fashion-mnist

LL=(8)
CR=(4)
BM=(50)
K=10

numactl --cpunodebind=0 --localalloc \
    ./binary/release/performence \
    ./data/${data}/train \
    ./data/${data}/test \
    ./data/${data}/neighbors \
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
