#! /usr/bin/env python3
from preprocessing import parse_annotation, parse_annotation_mat_box, parse_annotation_mat_point
from generator import YoloGenerator, RetinaGenerator
from frontend import YOLO, RetinaNet
import argparse
import keras
import json
import os, cv2
import numpy as np
from utils import draw_boxes

'''
Evaluation function for yolo model and retina model;
Calculate: 
    Average Precision @IOU=0.2/0.5
    Recall (Sensitivity) @IOU=0.2/0.5
    Recall (Sensitivity) for 1 stenosis @IOU=0.2/0.5  
    Precision (Specificity) @IOU=0.2/0.5
    Mean square error
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
    weights_path = config['train']['pretrained_weights']
    batch_size = config['train']['batch_size']
    eval_all_imgs = False
    if 'multiple' in config['valid']:
        is_multi = config['valid']['multiple']
    else:
        is_multi = True
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
    else:
        if eval_all_imgs:
            valid_imgs = train_imgs
            valid_labels = train_labels
        else:
            if config['valid']['valid_fold'] > 0 and config['valid']['valid_fold'] <= 4:
                valid_fold = config['valid']['valid_fold']
            else:
                valid_fold = 1
            valid_start = int((valid_fold - 1) / 4 * len(train_imgs))
            valid_end = int(valid_fold / 4 * len(train_imgs))
            np.random.seed(0)
            np.random.shuffle(train_imgs)

            valid_imgs = train_imgs[valid_start:valid_end]
            train_imgs = train_imgs[0:valid_start] + train_imgs[valid_end:]
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

    if config['model']['backend'] == "Retina":
        model = RetinaNet(
                        backend             = config['model']['backend'],
                        input_size          = config['model']['input_size'],
                        labels              = config['model']['labels'],
                        max_box_per_image   = config['model']['max_box_per_image'],
                        anchors             = config['model']['anchors'],
                        label_wts           = config['model']['label_wt'],
                        feature_extractor_weights   = config['train']['feature_extrctor_weights'],
                        feature_trainable   = config['train']['feature_trainable'])
    else:
        model = YOLO(backend             = config['model']['backend'],
                    input_size          = config['model']['input_size'],
                    labels              = config['model']['labels'],
                    max_box_per_image   = config['model']['max_box_per_image'],
                    anchors             = config['model']['anchors'],
                    label_wts           = config['model']['label_wt'],
                    feature_extractor_weights   = config['train']['feature_extrctor_weights'],
                    feature_trainable   = config['train']['feature_trainable'])
    #########################################
    #   Load the pretrained weights (if any) 
    #########################################

    if os.path.exists(config['valid']['pretrained_weights']):
        print("Loading pre-trained weights in \t", config['valid']['pretrained_weights'])
        model.load_weights(config['valid']['pretrained_weights'])
    else:
        raise Exception("No pretrained weights found.")

    #########################
    #   Evaluate the network
    #########################

    print("calculing mAP for iou threshold = {}".format(args.iou))

    if config['model']['backend'] == "Retina":
        generator_config = {
            'IMAGE_H': model.input_size,
            'IMAGE_W': model.input_size,
            'BOX': model.nb_box,
            'LABELS': model.labels,
            'CLASS': len(model.labels),
            'ANCHORS': model.anchors,
            'BATCH_SIZE': batch_size,
            'TRUE_BOX_BUFFER': model.max_box_per_image,
        }

        valid_generator = RetinaGenerator(valid_imgs,
                                          generator_config,
                                          norm=model.normalize,
                                          jitter=False,
                                          shuffle=False)

        if not os.path.exists(config['valid']['valid_output_folder']):
            os.mkdir(config['valid']['valid_output_folder'])
        out_valid_path = os.path.join(config['valid']['valid_output_folder'], str(config['valid']['valid_fold']))
        if not os.path.exists(out_valid_path):
            os.mkdir(out_valid_path)
        ap_0_5, ar_0_5, ar1_0_5, prec_0_5, mse_0_5 = model.evaluate(generator=valid_generator,
                                            iou_threshold=0.5,
                                            score_threshold=0.35,
                                            max_detections=100,
                                            save_path=out_valid_path)
        ap_0_2, ar_0_2, ar1_0_2, prec_0_2, mse_0_2 = model.evaluate(generator=valid_generator,
                                            iou_threshold=0.2,
                                            score_threshold=0.35,
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
    else:
        generator_config = {
            'IMAGE_H': model.input_size,
            'IMAGE_W': model.input_size,
            'GRID_H': model.grid_h,
            'GRID_W': model.grid_w,
            'BOX': model.nb_box,
            'LABELS': model.labels,
            'CLASS': len(model.labels),
            'ANCHORS': model.anchors,
            'BATCH_SIZE': batch_size,
            'TRUE_BOX_BUFFER': model.max_box_per_image,
        }
        valid_generator = YoloGenerator(valid_imgs,
                                        generator_config,
                                        norm=model.feature_extractor.normalize,
                                        jitter=False,
                                        shuffle=False)

        ############################################
        # Compute mAP on the validation set
        ############################################
        average_precision_0_5, recall_rate_0_5 = model.evaluate_stenose(valid_generator, 0.5, is_binary)
        average_precision_0_2, recall_rate_0_2 = model.evaluate_stenose(valid_generator, 0.2, is_binary)
        # print evaluation
        print('ap_0_5: {:.4f}'.format(average_precision_0_5))
        print('recall_0_5: {:.4f}'.format(recall_rate_0_5))
        print('ap_0_2: {:.4f}'.format(average_precision_0_2))
        print('recall_0_2: {:.4f}'.format(recall_rate_0_2))
        if not os.path.exists(config['valid']['valid_output_folder']):
            os.mkdir(config['valid']['valid_output_folder'])
        out_valid_path = os.path.join(config['valid']['valid_output_folder'], str(config['valid']['valid_fold']))
        if not os.path.exists(out_valid_path):
            os.mkdir(out_valid_path)
        eval_acc_result_txt = os.path.join(out_valid_path, 'evaluation_numerics.txt')
        with open(eval_acc_result_txt, 'w') as f:
            print('ap_0_2: {:.4f}'.format(average_precision_0_2), file=f)
            print('ap_0_5: {:.4f}'.format(average_precision_0_5), file=f)
            print('recall_0_5: {:.4f}'.format(recall_rate_0_5), file=f)
            print('recall_0_2: {:.4f}'.format(recall_rate_0_2), file=f)

        # predict samples
        for i in range(valid_generator.size()):
            sample_img = valid_generator.load_image(i)
            #sample_img = cv2.resize(sample_img, (yolo.input_size, yolo.input_size))
            sample_img_file = valid_generator.images[i]['filename']
            pred_boxes = model.predict(sample_img, is_binary)
            true_boxes = valid_generator.load_annotation(i).tolist()
            sample_img = draw_boxes(sample_img, pred_boxes, config['model']['labels'], true_boxes, top_k=5)

            save_path = os.path.join(out_valid_path, os.path.basename(sample_img_file))
            cv2.imwrite(save_path, sample_img)

if __name__ == '__main__':
    _args = argparser.parse_args()
    _main_(_args)
