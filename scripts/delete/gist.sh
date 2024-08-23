#!/bin/bash

cmake --build build/clang --config Release --target delete
# cmake --build build/gcc --config Debug --target delete

data=gist
numactl --cpunodebind=0 --localalloc \
    ./binary/release/delete \
    ./data/${data}/train \
    ./data/${data}/test \
    ./data/${data}/neighbors \
    ./data/${data}/reference_answer \
    ${data} \
    8 6 100 200 10 \
    ./data/${data}/delete250000irrelevant.binary \
    ./data/${data}/delete25000relevant.binary
