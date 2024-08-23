#!/bin/bash

cmake --build build/clang --config Release --target delete
# cmake --build build/gcc --config Debug --target delete

data=sift10M
numactl --cpunodebind=1 --localalloc \
    ./binary/release/delete \
    ./data/${data}/bigann_base.bvecs \
    ./data/${data}/bigann_query.bvecs \
    ./data/${data}/gnd/idx_10M.ivecs \
    ./data/${data}/reference_answer \
    ${data} \
    16 4 100 50 10 \
    ./data/${data}/delete2500000irrelevant.binary \
    ./data/${data}/delete500relevant.binary
