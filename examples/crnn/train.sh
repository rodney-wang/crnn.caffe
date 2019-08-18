#!/bin/bash
GPU_ID=1,2 
cur_date=`date +%Y-%m-%d-%H-%M-%S`
log_file_name=./examples/crnn/log/${cur_date}
/root/3rd/warpctc-caffe/build/tools/caffe train \
    -solver ./examples/crnn/solver.prototxt \
    -gpu $GPU_ID | tee -a ${log_file_name} 
    #-weights /ssd/wfei/models/crnn_caffe/wanda/crnn_wanda_fresh_v2.2.caffemodel 2>&1 | tee -a ${log_file_name} 
#    -weights /mnt/soulfs2/wfei/models/crnn_caffe/wanda/crnn_wanda_v2.0.caffemodel 2>&1 | tee -a ${log_file_name}
