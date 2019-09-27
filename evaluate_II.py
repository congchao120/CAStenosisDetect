#! /usr/bin/env python3
from preprocessing import parse_annotation, parse_annotation_mat_box
from generator import YoloGenerator_II
from frontend_II import YOLO_II
import argparse
import keras
import json
import os, cv2
import numpy as np
from utils import draw_maps_boxes

'''
Evaluation function for frontend_II model (focal loss for feature map)
'''

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

def _main_(args):
    config_path = args.conf
    device = args.device
    #keras.backend.tensorflow_backend.set_session(get_session())

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    batch_size = config['train']['batch_size']
    eval_all_imgs = False

    ##########################
    #   Parse the annotations 
    ##########################

    # parse annotations of the training set
    train_imgs, train_labels = parse_annotation_mat_box(config['train']['train_annot_folder'],
                                                        config['train']['train_image_folder'],
                                                        config['model']['labels'],
                                                        config['model']['threshold_min_wh'])

    # parse annotations of the validation set, if any.
    if os.path.exists(config['valid']['valid_annot_folder']):
        valid_imgs, valid_labels = parse_annotation_mat_box(config['valid']['valid_annot_folder'],
                                                        config['valid']['valid_image_folder'],
                                                        config['model']['labels'],
                                                        config['model']['threshold_min_wh'])
    else:
        if eval_all_imgs:
            valid_imgs = train_imgs
            valid_labels = train_labels
        else:
            train_valid_split = int(0.8 * len(train_imgs))
            np.random.seed(0)
            np.random.shuffle(train_imgs)
            valid_imgs = train_imgs[train_valid_split:]
            valid_labels = train_labels
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

    is_binary = False
    if config['train']['loss_type'] == 'focal_II' or config['train']['loss_type'] == 'focal_III':
        is_binary = True
    ########################
    #   Construct the model
    ########################

    yolo = YOLO_II(backend             = config['model']['backend'],
                    input_size          = config['model']['input_size'],
                    labels              = config['model']['labels'],
                    max_box_per_image   = config['model']['max_box_per_image'],
                    feature_extractor_weights   = config['train']['feature_extrctor_weights'])

    #########################################
    #   Load the pretrained weights (if any) 
    #########################################

    if os.path.exists(config['valid']['pretrained_weights']):
        print("Loading pre-trained weights in \t", config['valid']['pretrained_weights'])
        yolo.load_weights(config['valid']['pretrained_weights'])
    else:
        raise Exception("No pretrained weights found.")

    #########################
    #   Evaluate the network
    #########################

    print("calculing mAP for iou threshold = {}".format(args.iou))
    generator_config = {
        'IMAGE_H': yolo.input_size,
        'IMAGE_W': yolo.input_size,
        'GRID_H': yolo.grid_h,
        'GRID_W': yolo.grid_w,
        'LABELS': yolo.labels,
        'CLASS': len(yolo.labels),
        'BATCH_SIZE': batch_size
    }

    valid_generator = YoloGenerator_II(valid_imgs,
                                       generator_config,
                                       norm=yolo.feature_extractor.normalize,
                                       jitter=False,
                                       shuffle=False)

    ############################################
    # Compute mAP on the validation set
    ############################################

    average_precision_0_1 = yolo.evaluate_stenose(valid_generator, obj_threshold=0.1)
    average_precision_0_2 = yolo.evaluate_stenose(valid_generator, obj_threshold=0.2)
    average_precision_0_5 = yolo.evaluate_stenose(valid_generator, obj_threshold=0.5)
    average_precision = yolo.evaluate_stenose_ap(valid_generator)
    recall_obj_0_5, recall_img_0_5 = yolo.evaluate_stenose_rec(valid_generator)
    recall_obj_0_2, recall_img_0_2 = yolo.evaluate_stenose_rec(valid_generator, obj_threshold=0.2, iou_threshold=0.2)
    f_score_0_5, iou_0_5 = yolo.evaluate_stenose_f1_iou(valid_generator, obj_threshold=0.5)
    f_score_0_2, iou_0_2 = yolo.evaluate_stenose_f1_iou(valid_generator, obj_threshold=0.2)

    # print evaluation
    print('ap_0_1: {:.4f}'.format(average_precision_0_1))
    print('ap_0_2: {:.4f}'.format(average_precision_0_2))
    print('ap_0_5: {:.4f}'.format(average_precision_0_5))
    print(yolo.labels[1], '{:.4f}'.format(average_precision))
    print('recall_obj_0_5: {:.4f}'.format(recall_obj_0_5))
    print('recall_img_0_5: {:.4f}'.format(recall_img_0_5))
    print('recall_obj_0_2: {:.4f}'.format(recall_obj_0_2))
    print('recall_img_0_2: {:.4f}'.format(recall_img_0_2))


    # predict samples
    if not os.path.exists(config['valid']['valid_output_folder']):
        os.mkdir(config['valid']['valid_output_folder'])
    eval_acc_result_txt = os.path.join(config['valid']['valid_output_folder'], 'evaluation_numerics.txt')
    with open(eval_acc_result_txt, 'w') as f:
        print('ap_0_1: {:.4f}'.format(average_precision_0_1), file=f)
        print('ap_0_2: {:.4f}'.format(average_precision_0_2), file=f)
        print('ap_0_5: {:.4f}'.format(average_precision_0_5), file=f)
        print(yolo.labels[1], '{:.4f}'.format(average_precision), file=f)
        print('recall_obj_0_5: {:.4f}'.format(recall_obj_0_5), file=f)
        print('recall_img_0_5: {:.4f}'.format(recall_img_0_5), file=f)
        print('recall_obj_0_2: {:.4f}'.format(recall_obj_0_2), file=f)
        print('recall_img_0_2: {:.4f}'.format(recall_img_0_2), file=f)
        print('F1_score_0_5: %2.4f' % (f_score_0_5), file=f)
        print('F1_score_0_2: %2.4f' % (f_score_0_2), file=f)
        print('iou_0_5: %2.4f' % (iou_0_5), file=f)
        print('iou_0_2: %2.4f' % (iou_0_2), file=f)

    for i in range(valid_generator.size()):
        sample_img = valid_generator.load_image(i)
        #sample_img = cv2.resize(sample_img, (yolo.input_size, yolo.input_size))
        sample_img_file = valid_generator.images[i]['filename']
        pred_maps = yolo.predict(sample_img)
        true_boxes = valid_generator.load_annotation(i).tolist()
        sample_img = draw_maps_boxes(sample_img, pred_maps, config['model']['labels'], true_boxes, top_k=5)

        save_path = os.path.join(config['valid']['valid_output_folder'], os.path.basename(sample_img_file))
        cv2.imwrite(save_path, sample_img)

if __name__ == '__main__':
    _args = argparser.parse_args()
    _main_(_args)
