import os
import glob
import time
import argparse
from eval_generic import eval
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
        fname = os.path.basename(plate_file).split('_plate')[0]
        fname = fname.replace('.jpg', '.txt')

        out_file = os.path.join(out_dir, fname)

        out_str = ' '.join([chars, str(score)])
        print fname, out_str
        # with open(out_file, 'w', encoding='utf-8') as ff:
        with open(out_file, 'w') as ff:
            #ff.write(out_str.encode('utf-8'))
            ff.write(out_str)
    return True


def crnn_eval_end2end(ocrtxt_file, out_dir, model_path, skip):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    pocr = PlateOCR(model_path)
    start_time = time.time()
    for line in open(ocrtxt_file, 'r'):
        plate_file, label = line.split(';')
        run_crnn_and_write_result(plate_file, out_dir, pocr)
    print("--- %s seconds ---" % (time.time() - start_time))
    eval(ocrtxt_file, out_dir, skip)

def parse_args():
    parser = argparse.ArgumentParser(description='Plate Segmentation')
    parser.add_argument('--ocr_txt', default='/ssd/wfei/data/plate_for_label/energy_cars/energy_plates_ocrlabel_test_217.txt',
                        type=str, help='Input test image dir')
    parser.add_argument('--out_dir', default='/ssd/wfei/data/ocr_results/energy_plates_res',
                        type=str, help='Output image dir')
    parser.add_argument('--model', default='/mnt/soulfs2/wfei/code/crnn.caffe/examples/crnn/model_k11/crnn_k11_energy_v1.3.caffemodel', type=str, help='Caffe model path')
    parser.add_argument('--skip', default=0, type=int, help='Skip any characters in evaluation')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    crnn_eval_end2end(args.ocr_txt, args.out_dir, args.model, args.skip)
