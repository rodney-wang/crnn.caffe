# -*- coding=utf-8 -*-
import argparse
from plate_ocr_crnn import PlateOCR


parser = argparse.ArgumentParser(description='Plate Segmentation')
parser.add_argument('--img', default='/mnt/soulfs2/wfei/data/plate.jpg',
                    type=str, help='Input image')
parser.add_argument('--model',
                    default='/mnt/soulfs2/wfei/code/crnn.caffe/examples/crnn/model_k11/crnn_k11_energy_iter_7500.caffemodel',
                    type=str, help='Caffe model path')
args = parser.parse_args()

pocr = PlateOCR(args.model)
chars, score = pocr(args.img)
print chars, score
