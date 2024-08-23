#!/bin/bash

cmake --build build/clang --config Release --target delete
# cmake --build build/gcc --config Debug --target delete

data=fashion-mnist
numactl --cpunodebind=0 --localalloc \
    ./binary/release/delete \
    ./data/${data}/train \
    ./data/${data}/test \
    ./data/${data}/neighbors \
    ./data/${data}/reference_answer \
    ${data} \
    8 4 50 50 10 \
    ./data/${data}/delete2500irrelevant.binary \
    ./data/${data}/delete25000relevant.binary
