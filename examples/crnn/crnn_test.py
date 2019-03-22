# -*- coding=utf-8 -*-
import sys
#sys.path.append('~/Desktop/crnn.caffe/python')
import caffe
import json
from PIL import Image
import numpy as np

model_file = './model/crnn_plate_iter_12000.caffemodel'
deploy_file = 'deploy.prototxt'
test_img = '/mnt/soulfs2/wfei/data/plate.jpg'

# set device
caffe.set_device(4)
caffe.set_mode_gpu()

# load model
net = caffe.Net(deploy_file, model_file, caffe.TEST)

# load test img
IMAGE_WIDTH, IMAGE_HEIGHT = 96, 32
img = caffe.io.load_image(test_img, color=False) #load as grayscale
img = caffe.io.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
in_ = np.transpose(img, (2, 0, 1))

# 执行上面设置的图片预处理操作，并将图片载入到blob中
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_

# run net
net.forward()

# get result
res = net.blobs['probs'].data

print('result shape is:', res.shape)

# 取出标签文档
char_dict = json.load(open('../lprnet/utils/carplate.json', 'r'))
char_set =  {v: k for k, v in char_dict.iteritems()}
#char_set[73]='-'
# 取出最多可能的label标签
for i in range(24):
    data = res[i, :, :]
    index = np.argmax(data)
    #print(index, data[0, index])
    print char_set[index].encode('utf8')
