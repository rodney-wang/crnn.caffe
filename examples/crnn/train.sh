#!/bin/bash
#GPU_ID=3,4,5 
GPU_ID=6 
cur_date=`date +%Y-%m-%d-%H-%M-%S`
log_file_name=./examples/crnn/log/${cur_date}
/root/3rd/warpctc-caffe/build/tools/caffe train \
    -solver ./examples/crnn/solver.prototxt \
    -gpu $GPU_ID \
    -weights /mnt/soulfs2/wfei/models/crnn_caffe/wanda/crnn_wanda_v2.0.caffemodel 2>&1 | tee -a ${log_file_name}
#    -weights /mnt/soulfs2/wfei/code/crnn.caffe/examples/crnn/model/crnn_captcha_iter_10000.caffemodel \
