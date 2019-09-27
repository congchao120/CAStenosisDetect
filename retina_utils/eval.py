"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from .anchors import compute_overlap, predbox_transform
from .visualization import draw_detections, draw_annotations

import math
import keras
import numpy as np
import os
from predict_grad_cam import pred_cam_bbox
import cv2
import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."
from retina_utils.anchors import (
    anchor_targets_bbox,
    anchors_for_shape,
    guess_shapes
)

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    num_stenose = 1
    all_detections = [[None for i in range(num_stenose) if generator.has_label(i)] for j in range(len(generator.images))]

    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        raw_image    = generator.load_image(i)
        image        = generator.norm(raw_image.copy())
        scale = generator.config['IMAGE_H']/image.shape[0]
        image = cv2.resize(image, (generator.config['IMAGE_H'], generator.config['IMAGE_W']))
        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        pred_res = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes, scores = pred_res
        labels = np.argmax(scores, axis=2)
        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]
        anchors = anchors_for_shape(image.shape, shapes_callback=guess_shapes)
        # select those scores
        if len(indices) != 0:
            scores = scores[0][indices]
            labels = labels[0][indices]
            anchors = anchors[indices]
            # find the order with which to sort the scores
            scores_sort = np.argsort(-scores)[:max_detections]

            # select detections
            boxes      = boxes[0, indices, :]
            image_scores     = scores[:, scores_sort[:,0]][:, 0]
            image_labels     = labels
            image_boxes = predbox_transform(anchors, boxes)
        else:
            image_scores = []
            image_labels = []
            image_boxes = []
        #image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            file_name = os.path.basename(generator.images[i]['filename'])
            draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name, score_threshold=score_threshold)
            draw_annotations(raw_image, generator.load_annotation(i), label_to_name=generator.label_to_name)
            cv2.imwrite(os.path.join(save_path, file_name), raw_image)

        # copy detections to all_detections
        if len(image_boxes) == 0:
            all_detections[i] = []
        else:
            all_detections[i] = np.c_[image_boxes.transpose(), image_scores.transpose()]


    return all_detections



def _get_detections_cam(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    num_stenose = 1
    all_detections = [[None for i in range(num_stenose) if generator.has_label(i)] for j in range(len(generator.images))]

    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        raw_image    = generator.load_image(i)
        image        = generator.norm(raw_image.copy())
        scale = generator.config['IMAGE_H']/image.shape[0]
        image = cv2.resize(image, (generator.config['IMAGE_H'], generator.config['IMAGE_W']))
        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))
        layer_idx = len(model.layers) - 1
        # run network
        pred = model.predict(np.expand_dims(image, axis=0))
        idx_pred = int(np.argmax(pred[0]))
        all_boxes = pred_cam_bbox(model, layer_idx, filter_indices=1,  # 1 for stenosis
                                  seed_input=image, score_threshold=score_threshold,  # relu and guided don't work
                                  penultimate_layer_idx=310  # 310 is concatenation before global average pooling
                                  )
        if len(all_boxes) != 0:
            scores = all_boxes[:,4]
            labels = all_boxes[:,4]
            # correct boxes for image scale
            all_boxes /= scale
        else:
            scores = np.zeros((0,))
            labels = np.zeros((0,))



        # select indices which have a score above the threshold
        indices = np.where(scores[:] > score_threshold)[0]

        # select those scores
        if len(indices) != 0:
            scores = scores[indices]
            labels = labels[indices]
            # find the order with which to sort the scores
            scores_sort = np.argsort(-scores)[:max_detections]

            # select detections
            image_scores     = scores
            image_labels     = labels

            image_boxes = all_boxes[indices,0:4]
        else:
            image_scores = np.zeros((1))
            image_labels = np.zeros((1))
            image_boxes = np.zeros((1,4))
        #image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
        image_boxes = image_boxes.transpose((1,0))
        if save_path is not None:
            file_name = os.path.basename(generator.images[i]['filename'])
            draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name, score_threshold=score_threshold)
            draw_annotations(raw_image, generator.load_annotation(i), label_to_name=generator.label_to_name)
            cv2.imwrite(os.path.join(save_path, file_name), raw_image)

        # copy detections to all_detections
        if image_boxes.shape[1] == 0:
            all_detections[i] = []
        else:
            all_detections[i] = np.c_[image_boxes.transpose(), image_scores.transpose()]


    return all_detections

def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]

    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    num_stenose = 1
    all_annotations = [[None for i in range(num_stenose)] for j in range(len(generator.images))]

    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotation(i)
        # copy detections to all_annotations
        all_annotations[i]= annotations.copy()

    return all_annotations


def evaluate(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections     = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations    = _get_annotations(generator)
    average_precisions = {}

    # all_detections = pickle.load(open('all_detections.pkl', 'rb'))
    # all_annotations = pickle.load(open('all_annotations.pkl', 'rb'))
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))

    # process detections and annotations
    false_positives = np.zeros((0,))
    true_positives = np.zeros((0,))
    scores = np.zeros((0,))
    num_annotations = 0.0
    num_detections = 0.0
    meet_true_positive = False
    num_detected_annotations = 0.0
    num_rec_1 = 0
    total_sqr_error = 0.0
    for i in range(generator.size()):
        meet_true_positive = False
        detections = all_detections[i]
        annotations = all_annotations[i]
        num_annotations += annotations.shape[0]
        num_detections += len(detections)
        detected_annotations = []
        if len(detections) == 0:
            continue
        for j in range(detections.shape[0]):
            d = detections[j, :]
            if annotations.shape[1] == 0:
                continue

            overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
            assigned_annotation = np.argmax(overlaps, axis=1)
            max_overlap = overlaps[0, assigned_annotation]

            if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                false_positives = np.append(false_positives, 0)
                true_positives = np.append(true_positives, 1)
                detected_annotations.append(assigned_annotation)
                scores = np.append(scores, d[4])
                meet_true_positive = True
                num_detected_annotations += 1
            elif max_overlap < iou_threshold:
                false_positives = np.append(false_positives, 1)
                true_positives = np.append(true_positives, 0)
                scores = np.append(scores, d[4])
            elif max_overlap >= iou_threshold:
                num_detected_annotations += 1

            center_x_det = (d[0] + d[2])/2.0
            center_y_det = (d[1] + d[3])/2.0
            dist_x = ((annotations[:, 0] + annotations[:, 2]) / 2.0 - center_x_det) * ((annotations[:, 0] + annotations[:, 2]) / 2.0 - center_x_det)
            dist_y = ((annotations[:, 1] + annotations[:, 3]) / 2.0 - center_y_det) * ((annotations[:, 1] + annotations[:, 3]) / 2.0 - center_y_det)
            sqr_error = math.sqrt(min(dist_x + dist_y))
            total_sqr_error += sqr_error
        if meet_true_positive:
            num_rec_1 = num_rec_1 + 1

    # no annotations -> AP for this class is 0 (is this correct?)
    if num_annotations == 0:
        average_precisions= 0, 0

    # sort by score
    indices = np.argsort(-scores)
    false_positives = false_positives[indices]
    true_positives = true_positives[indices]

    # compute false positives and true positives
    false_positives_cumsum = np.cumsum(false_positives)
    true_positives_cumsum = np.cumsum(true_positives)

    # compute recall and precision
    recall = true_positives_cumsum / num_annotations
    precision = true_positives_cumsum / np.maximum(true_positives_cumsum + false_positives_cumsum, np.finfo(np.float64).eps)

    # compute average precision
    average_precision = _compute_ap(recall, precision)
    average_recall = np.sum(true_positives) / num_annotations
    average_recall_1 = num_rec_1 / generator.size()
    precision = num_detected_annotations / num_detections
    mean_sqr_error = total_sqr_error / num_detections
    return average_precision, average_recall, average_recall_1, precision, mean_sqr_error



def evaluate_cam(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections     = _get_detections_cam(generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations    = _get_annotations(generator)
    average_precisions = {}

    # all_detections = pickle.load(open('all_detections.pkl', 'rb'))
    # all_annotations = pickle.load(open('all_annotations.pkl', 'rb'))
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))

    # process detections and annotations
    false_positives = np.zeros((0,))
    true_positives = np.zeros((0,))
    scores = np.zeros((0,))
    num_annotations = 0.0
    num_detections = 0.0
    meet_true_positive = False
    num_detected_annotations = 0.0
    num_rec_1 = 0
    total_sqr_error = 0.0
    for i in range(generator.size()):
        meet_true_positive = False
        detections = all_detections[i]
        annotations = all_annotations[i]
        if annotations.shape[1] == 0:
            continue
        num_annotations += annotations.shape[0]
        num_detections += len(detections)
        detected_annotations = []
        if len(detections) == 0:
            #num_rec_1 = num_rec_1 + 1
            continue
        for j in range(detections.shape[0]):
            d = detections[j, :]

            overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
            assigned_annotation = np.argmax(overlaps, axis=1)
            max_overlap = overlaps[0, assigned_annotation]
            center_x_det = (d[0] + d[2])/2.0
            center_y_det = (d[1] + d[3])/2.0
            dist_x = ((annotations[:, 0] + annotations[:, 2]) / 2.0 - center_x_det) * ((annotations[:, 0] + annotations[:, 2]) / 2.0 - center_x_det)
            dist_y = ((annotations[:, 1] + annotations[:, 3]) / 2.0 - center_y_det) * ((annotations[:, 1] + annotations[:, 3]) / 2.0 - center_y_det)
            sqr_error = math.sqrt(min(dist_x + dist_y))
            total_sqr_error += sqr_error
            if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                false_positives = np.append(false_positives, 0)
                true_positives = np.append(true_positives, 1)
                detected_annotations.append(assigned_annotation)
                scores = np.append(scores, d[4])
                meet_true_positive = True
                num_detected_annotations += 1
            elif max_overlap < iou_threshold:
                false_positives = np.append(false_positives, 1)
                true_positives = np.append(true_positives, 0)
                scores = np.append(scores, d[4])
            elif max_overlap >= iou_threshold:
                num_detected_annotations += 1

        if meet_true_positive:
            num_rec_1 = num_rec_1 + 1

    # no annotations -> AP for this class is 0 (is this correct?)
    if num_annotations == 0:
        average_precisions= 0, 0

    # sort by score
    indices = np.argsort(-scores)
    false_positives = false_positives[indices]
    true_positives = true_positives[indices]

    # compute false positives and true positives
    false_positives_cumsum = np.cumsum(false_positives)
    true_positives_cumsum = np.cumsum(true_positives)

    # compute recall and precision
    recall = true_positives_cumsum / num_annotations
    precision = true_positives_cumsum / np.maximum(true_positives_cumsum + false_positives_cumsum, np.finfo(np.float64).eps)

    # compute average precision
    average_precision = _compute_ap(recall, precision)
    average_recall = np.sum(true_positives) / num_annotations
    average_recall_1 = num_rec_1 / generator.size()
    precision = num_detected_annotations / num_detections
    mean_sqr_error = total_sqr_error / num_detections
    return average_precision, average_recall, average_recall_1, precision, mean_sqr_error
