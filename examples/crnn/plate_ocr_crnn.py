#coding=utf-8
import numpy as np
import json
import caffe
#import cv2

class PlateOCR:

    def __init__(self, caffemodel='/mnt/soulfs2/wfei/code/crnn.caffe/examples/crnn/model/crnn_plate_iter_12000.caffemodel'):
        char_dict = json.load(open('/mnt/soulfs2/wfei/code/crnn.caffe/examples/lprnet/utils/carplate.json', 'r'))
        self.char_set = {v: k for k, v in char_dict.iteritems()}
        self.net = self._init_det(caffemodel)

    def _init_det(self, caffemodel):
        GPU_ID = 0
        caffe.set_mode_gpu()
        caffe.set_device(GPU_ID)

        model_fold = '/mnt/soulfs2/wfei/code/crnn.caffe/examples/crnn/'
        prototxt = model_fold + 'deploy.prototxt'
        #caffemodel = model_fold + 'model/crnn_plate_iter_120000.caffemodel'
        net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        return net

    def __call__(self, img):
        """ Plate OCR with CRNN
         results: chars
        """
        net = self.net
        # load test img
        IMAGE_WIDTH, IMAGE_HEIGHT = 96, 32
        img = caffe.io.load_image(img, color=False)  # load as grayscale
        img = caffe.io.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
        in_ = np.transpose(img, (2, 0, 1))

        # shape for input (data blob is N x C x H x W), set data
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_
        # run net
        net.forward()

        # get result
        res = net.blobs['probs'].data
        chars, score = self.decoder(res)
        return chars, score

    def decoder(self, res):
        score = 0
        chars = "" 
        for i in range(24):
            data = res[i, :, :]
            index = np.argmax(data)
            if i == 0:
                prev = index
                if index != 73:
                    chars += self.char_set[index].encode('utf8')
                    score += data[0, index]
                continue
            if index == 73:
                prev = -1
            else:
                if index != prev:
                    chars += self.char_set[index].encode('utf8')
                    score += data[0, index]
                prev = index

           #print index, data[0, index]
            #chars.append(self.char_set[index].encode('utf8'))

        return chars, score

if __name__ == '__main__':

    test_img = '/mnt/soulfs2/wfei/data/plate.jpg'
    pocr = PlateOCR()
    chars, score = pocr(test_img)
    print chars, score


