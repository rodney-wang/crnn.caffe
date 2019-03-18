#!/bin/bash
GPU_ID=0,1
cur_date=`date +%Y-%m-%d-%H-%M-%S`
log_file_name=./examples/lprnet/log/${cur_date}
./build/tools/caffe train \
    -solver ./examples/lprnet/solver.prototxt \
    -gpu $GPU_ID 2>&1 | tee -a ${log_file_name}
