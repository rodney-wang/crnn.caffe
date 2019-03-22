import cv2
import os
import glob
import time
import argparse
import numpy as np
from plate_ocr_crnn import PlateOCR

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

"""
 Process all plates in benchmark
"""

def run_crnn_and_write_result(plate_file, out_dir, pocr):
    plate_file = plate_file.strip()

    if not os.path.exists(plate_file.encode('utf-8')):
        print('File does not exist')
        return
    chars, score = pocr(plate_file)

    if score >=0 and len(chars) != 0:
        fname = os.path.basename(plate_file).split('_plate.png')[0]
        fname = fname.replace('.jpg', '.txt')

        out_file = os.path.join(out_dir, fname)

        out_str = ' '.join([chars, str(score)])
        print(fname, out_str.encode('utf-8'))
        # with open(out_file, 'w', encoding='utf-8') as ff:
        with open(out_file, 'w') as ff:
            ff.write(out_str.encode('utf-8'))
    return True


def batch_benchmark(img_dir, out_dir, model_path):
    fnames = glob.glob(os.path.join(img_dir, '*.png'))
    fnames = sorted(fnames)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    pocr = PlateOCR()
    start_time = time.time()
    for plate_file in fnames:
        run_crnn_and_write_result(plate_file, out_dir, pocr)
    print("--- %s seconds ---" % (time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser(description='Plate Segmentation')
    parser.add_argument('--img_dir', default='/ssd/wfei/data/testing_data/k11_plates_v1.2',
                        type=str, help='Input test image dir')
    parser.add_argument('--out_dir', default='/ssd/wfei/data/testing_data/k11_crnn_caffe_v1.0',
                        type=str, help='Output image dir')
    parser.add_argument('--model', default='/mnt/soulfs2/wfei/code/crnn.caffe/examples/crnn/model/crnn_plate_iter_120000.caffemodel',
                        type=str, help='Caffe model path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    batch_benchmark(args.img_dir, args.out_dir, args.model)
