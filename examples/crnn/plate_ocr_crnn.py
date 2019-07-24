#coding=utf-8
import numpy as np
import json
import argparse
import caffe
#import cv2

class PlateOCR:

    def __init__(self, caffemodel='/ssd/wfei/code/crnn.caffe/examples/crnn/model_hk/crnn_hkonly_iter_40000.caffemodel'):
        char_dict = json.load(open('./utils/carplate.json', 'r'))
        self.char_set = {v: k for k, v in char_dict.iteritems()}
        self.net = self._init_det(caffemodel)

    def _init_det(self, caffemodel):
        gpu = True
        if gpu:
            GPU_ID = 0
            caffe.set_mode_gpu()
            caffe.set_device(GPU_ID)
        else:
            caffe.set_mode_cpu()
        #model_fold = '/mnt/soulfs2/wfei/code/crnn.caffe/examples/crnn/'
        model_fold = '/ssd/wfei/code/crnn.caffe/examples/crnn/'
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
        scores =[]
        chars = "" 
        for i in range(24):
            data = res[i, :, :]
            index = np.argmax(data)
            if i == 0:
                prev = index
                if index != 33:
                    chars += self.char_set[index].encode('utf8')
                    score += data[0, index]
                    scores.append( data[0, index] )
                continue
            if index == 33:
                prev = -1
            else:
                if index != prev:
                    chars += self.char_set[index].encode('utf8')
                    score += data[0, index]
                    scores.append( data[0, index] )
                prev = index

        clen = len(chars.decode('utf8'))
        if clen != 0:
            score = 10*score/clen
        """
        score = np.min(scores)        
        print scores
        """
        return chars, score

def parse_args():
    parser = argparse.ArgumentParser(description='Plate OCR with CRNN caffe')
    parser.add_argument('-i', '--img_path',
                        default='/mnt/soulfs2/wfei/tmp/car_sample_plate.jpg',
                        type=str, help='Input test image dir')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    test_img = '/mnt/soulfs2/wfei/tmp/car_sample_plate.jpg'
    pocr = PlateOCR(caffemodel='/mnt/soulfs2/wfei/models/crnn_caffe/k11/crnn_k11_energy_v1.5.caffemodel')
    #pocr = PlateOCR(caffemodel='/mnt/soulfs2/wfei/code/crnn.caffe/examples/crnn/model_k11/crnn_k11_energy_iter_17500.caffemodel')
    chars, score = pocr(args.img_path)
    print chars, score


