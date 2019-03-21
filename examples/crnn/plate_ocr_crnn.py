#coding=utf-8
import numpy as np
import caffe
#import skimage
import cv2

class PlateOCR:

    def __init__(self):
        self.net = self._init_det()

    def _init_det(self):
        GPU_ID = 0
        caffe.set_mode_gpu()
        caffe.set_device(GPU_ID)

        model_fold = '/mnt/soulfs2/wfei/code/crnn.caffe/examples/crnn/'
        prototxt = model_fold + 'deploy.prototxt'
        caffemodel = model_fold + 'model/crnn_plate_iter_60000.caffemodel'
        net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        return net

    def __call__(self, img):
        """ Plate OCR with CRNN
        Input
        -----
        net: network
        img: image

        Output
        ------
        results: chars
        """
        net = self.net
        width = 96
        height = 32
        X = cv2.resize(img, (width, height))
        X = np.transpose(X, (2, 0, 1))
        X = X.reshape(1, 1, height, width)
        net.blobs['data'].data[:] = X[:]
        out = net.forward()
        prediction = net.blobs['probs'].data
        print "prediction shape:", prediction.shape
        print "prediction[0] shape:", prediction[0].shape
        print prediction
        prediction = prediction[0]

        return prediction

if __name__ == '__main__':

    fname = ''
    img = caffe.io.imread
    ssd_net = PlateDet()
    batch_plate_det(args.img_dir, args.out_json, args.csv_file)
    #parse_csv(csv_file)


