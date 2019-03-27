#!/bin/bash
GPU_ID=2,3,4 
cur_date=`date +%Y-%m-%d-%H-%M-%S`
log_file_name=./examples/crnn/log/${cur_date}
/root/3rd/warpctc-caffe/build/tools/caffe train \
    -solver ./examples/crnn/solver.prototxt \
    -gpu $GPU_ID \
    -weights /mnt/soulfs2/wfei/code/crnn.caffe/examples/crnn/model_aug2/crnn_aug_iter_32000.caffemodel 2>&1 | tee -a ${log_file_name}
#    -weights /mnt/soulfs2/wfei/code/crnn.caffe/examples/crnn/model/crnn_captcha_iter_10000.caffemodel \
