#! /usr/bin/env python3
from preprocessing import parse_annotation, parse_annotation_mat_box, parse_annotation_mat_point
from generator import YoloGenerator, RetinaGenerator, CAMGenerator
from frontend import YOLO, RetinaNet
import argparse
import keras
import json
import os, cv2
import numpy as np
from Inception_Models import inceptionV3_coronary_model
from retina_utils.eval import evaluate_cam
from keras import activations
from vis.utils import utils

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    default='config.json',
    help='path to configuration file')

argparser.add_argument(
    '-i',
    '--iou',
    default=0.5,
    help='IOU threshold',
    type=float)

argparser.add_argument(
    '-d',
    '--device',
    default='0',
    help='Using GPU device')
'''
Evaluation function for CAM and classification model;
Calculate: 
    Average Precision @IOU=0.2/0.5
    Recall (Sensitivity) @IOU=0.2/0.5
    Recall (Sensitivity) for 1 stenosis @IOU=0.2/0.5  
    Precision (Specificity) @IOU=0.2/0.5
    Mean square error
'''

def _main_(args):
    config_path = args.conf
    device = args.device
    #keras.backend.tensorflow_backend.set_session(get_session())

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    batch_size = config['train']['batch_size']
    if 'multiple' in config['valid']:
        is_multi = config['valid']['multiple']
    else:
        is_multi = True
    ##########################
    #   Parse the annotations 
    ##########################

    # parse annotations of the training set
    if is_multi:
        valid_imgs, valid_labels = parse_annotation_mat_box(config['valid']['valid_annot_folder'],
                                                            config['valid']['valid_image_folder'],
                                                            config['model']['labels'],
                                                            config['model']['threshold_min_wh'])
    else:
        valid_imgs, valid_labels = parse_annotation_mat_point(config['valid']['valid_annot_folder'],
                                                            config['valid']['valid_image_folder'],
                                                            config['model']['labels'],
                                                            config['model']['threshold_min_wh'])
    if len(config['model']['labels']) > 0:
        overlap_labels = set(config['model']['labels']).intersection(set(valid_labels.keys()))

        print('Seen labels:\t', valid_labels)
        print('Given labels:\t', config['model']['labels'])
        print('Overlap labels:\t', overlap_labels)

        if len(overlap_labels) < len(config['model']['labels']):
            print('Warning: Some labels have no annotations.')
    else:
        print('No labels are provided. Train on all seen labels.')
        config['model']['labels'] = valid_labels.keys()

    #########################
    #   Evaluate the network
    #########################

    print("calculing mAP for iou threshold = {}".format(args.iou))

    generator_config = {
        'IMAGE_H': config['model']['input_size'],
        'IMAGE_W': config['model']['input_size'],
        'BOX': 0,
        'LABELS': config['model']['labels'],
        'CLASS': len(config['model']['labels']),
        'ANCHORS': None,
        'BATCH_SIZE': batch_size,
        'TRUE_BOX_BUFFER': 10,
    }
    valid_generator = CAMGenerator(valid_imgs,
                                      generator_config,
                                      jitter=False,
                                      shuffle=False)

    if not os.path.exists(config['valid']['valid_output_folder']):
        os.mkdir(config['valid']['valid_output_folder'])
    out_valid_path = config['valid']['valid_output_folder']
    if not os.path.exists(out_valid_path):
        os.mkdir(out_valid_path)
    if config['model']['backend'] == "CAM":
        num_classes = 3
        pred_output = np.zeros((1, 1, num_classes))
        img_input = np.zeros((512, 512, 3))
        img_input = np.expand_dims(img_input, axis=0)
        model = inceptionV3_coronary_model(img_input, pred_output)
    else:
        print("Error model type!")
    #########################################
    #   Load the pretrained weights (if any)
    #########################################

    if os.path.exists(config['valid']['pretrained_weights']):
        print("Loading pre-trained weights in \t", config['valid']['pretrained_weights'])
        model.load_weights(config['valid']['pretrained_weights'])
    else:
        raise Exception("No pretrained weights found.")
    layer_idx = len(model.layers) - 1
    # swap with softmax with linear classifier for the reasons mentioned above
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)
    ap_0_5, ar_0_5, ar1_0_5, prec_0_5, mse_0_5 = evaluate_cam(generator=valid_generator,
                                                     model=model,
                                                       iou_threshold=0.5,
                                                       score_threshold=0.75,
                                                       max_detections=100,
                                                       save_path=out_valid_path)
    ap_0_2, ar_0_2, ar1_0_2, prec_0_2, mse_0_2 = evaluate_cam(generator=valid_generator,
                                                     model=model,
                                                       iou_threshold=0.1,
                                                       score_threshold=0.75,
                                                       max_detections=100,
                                                       save_path=None)

    eval_acc_result_txt = os.path.join(out_valid_path, 'evaluation_numerics.txt')
    with open(eval_acc_result_txt, 'w') as f:
        print('ap_0_2: {:.4f}'.format(ap_0_2), file=f)
        print('ap_0_5: {:.4f}'.format(ap_0_5), file=f)
        print('recall_0_2: {:.4f}'.format(ar_0_2), file=f)
        print('recall_0_5: {:.4f}'.format(ar_0_5), file=f)
        print('recall1_0_2: {:.4f}'.format(ar1_0_2), file=f)
        print('recall1_0_5: {:.4f}'.format(ar1_0_5), file=f)
        print('precision_0_2: {:.4f}'.format(prec_0_2), file=f)
        print('precision_0_5: {:.4f}'.format(prec_0_5), file=f)
        print('MSE_0_2: {:.4f}'.format(mse_0_2), file=f)
        print('MSE_0_5: {:.4f}'.format(mse_0_5), file=f)

if __name__ == '__main__':
    _args = argparser.parse_args()
    _main_(_args)
