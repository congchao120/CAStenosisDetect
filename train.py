#! /usr/bin/env python

import argparse
import os
import numpy as np
from preprocessing import parse_annotation, parse_annotation_mat_box
from frontend import YOLO, RetinaNet
import json


argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-d',
    '--device',
    default='0',
    help='Using GPU device')

def _main_(args):
    config_path = args.conf
    device = args.device
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    ###############################
    #   Parse the annotations 
    ###############################

    # parse annotations of the training set
    train_imgs, train_labels = parse_annotation_mat_box(config['train']['train_annot_folder'],
                                                        config['train']['train_image_folder'],
                                                        config['model']['labels'],
                                                        config['model']['threshold_min_wh'])

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(config['valid']['valid_annot_folder']):
        valid_imgs, valid_labels = parse_annotation_mat_box(config['valid']['valid_annot_folder'],
                                                            config['valid']['valid_image_folder'],
                                                            config['model']['labels'],
                                                            config['model']['threshold_min_wh'])
    else:
        if config['valid']['valid_fold'] > 0 and config['valid']['valid_fold'] <= 4:
            valid_fold = config['valid']['valid_fold']
        else:
            valid_fold = 1
        valid_start = int((valid_fold - 1)/4*len(train_imgs))
        valid_end = int(valid_fold/4*len(train_imgs))
        np.random.seed(0)
        np.random.shuffle(train_imgs)

        valid_imgs = train_imgs[valid_start:valid_end]
        train_imgs = train_imgs[0:valid_start] + train_imgs[valid_end:]

    if len(config['model']['labels']) > 0:
        overlap_labels = set(config['model']['labels']).intersection(set(train_labels.keys()))

        print('Seen labels:\t', train_labels)
        print('Given labels:\t', config['model']['labels'])
        print('Overlap labels:\t', overlap_labels)           

        if len(overlap_labels) < len(config['model']['labels']):
            print('Warning: Some labels have no annotations.')
    else:
        print('No labels are provided. Train on all seen labels.')
        config['model']['labels'] = train_labels.keys()
        
    ###############################
    #   Construct the model 
    ###############################
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

    ###############################
    #   Load the pretrained weights (if any) 
    ###############################    

    if os.path.exists(config['train']['pretrained_weights']):
        print("Loading pre-trained weights in", config['train']['pretrained_weights'])
        model.load_weights(config['train']['pretrained_weights'])

    ###############################
    #   Start the training process 
    ###############################

    model.train(train_imgs         = train_imgs,
               valid_imgs         = valid_imgs,
               train_times        = config['train']['train_times'],
               valid_times        = config['valid']['valid_times'],
               nb_epochs          = config['train']['nb_epochs'], 
               learning_rate      = config['train']['learning_rate'], 
               batch_size         = config['train']['batch_size'],
               warmup_epochs      = config['train']['warmup_epochs'],
               loss_type          = config['train']['loss_type'],
               object_scale       = config['train']['object_scale'],
               no_object_scale    = config['train']['no_object_scale'],
               coord_scale        = config['train']['coord_scale'],
               class_scale        = config['train']['class_scale'],
               focal_gamma        = config['train']['focal_gamma'],
               focal_alpha        = config['train']['focal_alpha'],
               l2_sigma           = config['train']['l2_sigma'],
               saved_weights_name = config['train']['saved_weights_name'],
               debug              = config['train']['debug'],
               save_best_only     = config['train']['save_best_only'],
               valid_path         = config['valid']['valid_output_folder'])



if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
