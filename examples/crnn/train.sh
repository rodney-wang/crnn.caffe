#!/bin/bash
GPU_ID=1
cur_date=`date +%Y-%m-%d-%H-%M-%S`
log_file_name=./examples/crnn/log/${cur_date}
/root/3rd/warpctc-caffe/build/tools/caffe train \
    -solver ./examples/crnn/solver.prototxt \
    -gpu $GPU_ID 2>&1 | tee -a ${log_file_name}

#    -weights /mnt/soulfs2/wfei/code/crnn.caffe/examples/crnn/model/crnn_captcha_iter_10000.caffemodel \
