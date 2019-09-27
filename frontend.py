from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
import tensorflow as tf
import numpy as np
import os
import cv2
from utils import decode_netout, compute_overlap, compute_ap, decode_stenoses, decode_stenoses_binary
from keras.applications.mobilenet import MobileNet
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam, RMSprop
from generator import YoloGenerator, RetinaGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from backend import TinyYoloFeature, FullYoloFeature, MobileNetFeature, SqueezeNetFeature, Inception3Feature, VGG16Feature, ResNet50Feature, InceptCoronaryFeature, UpsampleLike
import keras
from Inception_Models import inceptionV3_coronary_model
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import initializers
from retina_utils.eval import evaluate as evaluate_retina

'''
Using loss_xy, loss_wh, loss_confidence, loss_classification as loss function;
This is a typical yolo_9000 model
'''

class YOLO(object):
    def __init__(self, backend,
                       input_size, 
                       labels,
                       max_box_per_image,
                       anchors,
                       label_wts=None,
                       feature_extractor_weights = None,
                       feature_trainable = True):

        self.input_size = input_size
        self.backend = backend
        self.labels   = list(labels)
        self.nb_class = len(self.labels)
        if self.nb_class == 1:
            self.nb_class = 2
        self.nb_box   = len(anchors)//2
        if label_wts == None:
            self.class_wt = np.ones(self.nb_class, dtype='float32')
        else:
            self.class_wt = label_wts
        self.anchors  = anchors

        self.max_box_per_image = max_box_per_image

        ##########################
        # Make the model
        ##########################

        # make the feature extractor layers
        input_image     = Input(shape=(self.input_size, self.input_size, 3))
        self.true_boxes = Input(shape=(1, 1, 1, max_box_per_image , 4))


        if backend == 'Inception3':
            self.feature_extractor = Inception3Feature(self.input_size)  
        elif backend == 'SqueezeNet':
            self.feature_extractor = SqueezeNetFeature(self.input_size)        
        elif backend == 'MobileNet':
            self.feature_extractor = MobileNetFeature(self.input_size)
        elif backend == 'Full Yolo':
            self.feature_extractor = FullYoloFeature(self.input_size)
        elif backend == 'Tiny Yolo':
            self.feature_extractor = TinyYoloFeature(self.input_size)
        elif backend == 'VGG16':
            self.feature_extractor = VGG16Feature(self.input_size)
        elif backend == 'ResNet50':
            self.feature_extractor = ResNet50Feature(self.input_size)
        elif backend == 'Coronary':
            self.feature_extractor = InceptCoronaryFeature(self.input_size, feature_extractor_weights)
        else:
            raise Exception('Architecture not supported! ')

        self.feature_extractor.feature_extractor.trainable = feature_trainable
        print(self.feature_extractor.get_output_shape())    
        self.grid_h, self.grid_w = self.feature_extractor.get_output_shape()        
        features = self.feature_extractor.extract(input_image)            

        # make the object detection layer
        output = Conv2D(self.nb_box * (4 + 1 + self.nb_class), 
                        (1,1), strides=(1,1), 
                        padding='same', 
                        name='DetectionLayer', 
                        kernel_initializer='lecun_normal')(features)
        output = Reshape((self.grid_h, self.grid_w, self.nb_box, 4 + 1 + self.nb_class))(output)
        output = Lambda(lambda args: args[0])([output, self.true_boxes])

        self.model = Model([input_image, self.true_boxes], output)

        
        # initialize the weights of the detection layer
        layer = self.model.layers[-4]
        weights = layer.get_weights()

        new_kernel = np.random.normal(size=weights[0].shape)/(self.grid_h*self.grid_w)
        new_bias   = np.random.normal(size=weights[1].shape)/(self.grid_h*self.grid_w)

        layer.set_weights([new_kernel, new_bias])

        # print a summary of the whole model
        self.model.summary()

    def custom_loss(self, y_true, y_pred):
        mask_shape = tf.shape(y_true)[:4]
        
        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(self.grid_w), [self.grid_h]), (1, self.grid_h, self.grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))

        cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [self.batch_size, 1, 1, self.nb_box, 1])
        
        coord_mask = tf.zeros(mask_shape)
        conf_mask  = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)
        
        seen = tf.Variable(0.)
        total_recall = tf.Variable(0.)
        
        """
        Adjust prediction
        """
        ### adjust x and y      
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
        
        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.anchors, [1,1,1,self.nb_box,2])
        
        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])
        
        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]
        
        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:2] # relative position to the containing cell
        
        ### adjust w and h
        true_box_wh = y_true[..., 2:4] # number of cells accross, horizontally and vertically
        
        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins    = true_box_xy - true_wh_half
        true_maxes   = true_box_xy + true_wh_half
        
        pred_wh_half = pred_box_wh / 2.
        pred_mins    = pred_box_xy - pred_wh_half
        pred_maxes   = pred_box_xy + pred_wh_half       
        
        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)
        
        true_box_conf = iou_scores * y_true[..., 4]
        
        ### adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)
        
        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.coord_scale
        
        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = self.true_boxes[..., 0:2]
        true_wh = self.true_boxes[..., 2:4]
        
        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half
        
        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)
        
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half    
        
        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * self.no_object_scale
        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * self.object_scale
        # refer to https://github.com/experiencor/keras-yolo2/issues/353
        conf_mask_neg = tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * self.no_object_scale
        conf_mask_pos = y_true[..., 4] * self.object_scale
        nb_conf_box_neg = tf.reduce_sum(tf.to_float(conf_mask_neg > 0.0))
        nb_conf_box_pos = tf.reduce_sum(tf.to_float(conf_mask_pos > 0.0))

        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * tf.gather(self.class_wt, true_box_class) * self.class_scale       
        
        """
        Warm-up training
        """
        no_boxes_mask = tf.to_float(coord_mask < self.coord_scale/2.)
        seen = tf.assign_add(seen, 1.)
        
        true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, self.warmup_batches+1), 
                              lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask, 
                                       true_box_wh + tf.ones_like(true_box_wh) * \
                                       np.reshape(self.anchors, [1,1,1,self.nb_box,2]) * \
                                       no_boxes_mask, 
                                       tf.ones_like(coord_mask)],
                              lambda: [true_box_xy, 
                                       true_box_wh,
                                       coord_mask])
        
        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))
        
        loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        #loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
        # refer to https://github.com/experiencor/keras-yolo2/issues/353
        loss_conf_neg = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask_neg) / (
                    nb_conf_box_neg + 1e-6) / 2.
        loss_conf_pos = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask_pos) / (
                    nb_conf_box_pos + 1e-6) / 2.
        loss_conf = loss_conf_neg + loss_conf_pos
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
        
        loss = tf.cond(tf.less(seen, self.warmup_batches+1), 
                      lambda: loss_xy + loss_wh + loss_conf + loss_class + 10,
                      lambda: loss_xy + loss_wh + loss_conf + loss_class)
        
        if self.debug:
            nb_true_box = tf.reduce_sum(y_true[..., 4])
            nb_true = tf.reduce_sum(tf.to_float(true_box_conf > 0.2))
            nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.5))
            
            current_recall = nb_pred_box/(nb_true_box + 1e-6)
            total_recall = tf.assign_add(total_recall, current_recall) 

            loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
            loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
            loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
            loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
            loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
            loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
            loss = tf.Print(loss, [total_recall/seen], message='Average Recall \t', summarize=1000)
        
        return loss

    def stenosis_loss(self, y_true, y_pred):
        mask_shape = tf.shape(y_true)[:4]

        coord_mask = tf.zeros(mask_shape)
        conf_mask = tf.zeros(mask_shape)

        seen = tf.Variable(0.)
        total_recall = tf.Variable(0.)

        """
        Adjust prediction
        """
        ### adjust x and y
        pred_box_xy = tf.sigmoid(y_pred[..., :2])

        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.anchors, [1, 1, 1, self.nb_box, 2])

        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])

        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]

        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:2]  # relative position to the containing cell

        ### adjust w and h
        true_box_wh = y_true[..., 2:4]  # number of cells accross, horizontally and vertically

        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins = true_box_xy - true_wh_half
        true_maxes = true_box_xy + true_wh_half

        pred_wh_half = pred_box_wh / 2.
        pred_mins = pred_box_xy - pred_wh_half
        pred_maxes = pred_box_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        true_box_conf = iou_scores * y_true[..., 4]

        ### adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)

        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.coord_scale

        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = self.true_boxes[..., 0:2]
        true_wh = self.true_boxes[..., 2:4]

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        # refer to https://github.com/experiencor/keras-yolo2/issues/353
        conf_mask_neg = tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * self.no_object_scale
        conf_mask_pos = y_true[..., 4] * self.object_scale
        nb_conf_box_neg = tf.reduce_sum(tf.to_float(conf_mask_neg > 0.0))
        nb_conf_box_pos = tf.reduce_sum(tf.to_float(conf_mask_pos > 0.0))

        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * tf.gather(self.class_wt, true_box_class) * self.class_scale

        """
        Warm-up training
        """
        no_boxes_mask = tf.to_float(coord_mask < self.coord_scale / 2.)
        seen = tf.assign_add(seen, 1.)

        #true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, self.warmup_batches + 1),
        #                                               lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
        #                                                        true_box_wh + tf.ones_like(true_box_wh) * \
        #                                                        np.reshape(self.anchors, [1, 1, 1, self.nb_box, 2]) * \
        #                                                        no_boxes_mask,
        #                                                        tf.ones_like(coord_mask)],
        #                                               lambda: [true_box_xy,
        #                                                        true_box_wh,
        #                                                        coord_mask])

        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

        loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf_neg = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask_neg) / (
                    nb_conf_box_neg + 1e-6) / 2.
        loss_conf_pos = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask_pos) / (
                    nb_conf_box_pos + 1e-6) / 2.
        loss_conf =  loss_conf_pos + loss_conf_neg
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

        #loss = tf.cond(tf.less(seen, self.warmup_batches + 1),
        #               lambda: loss_xy + loss_wh + loss_conf + loss_class + 10,
        #               lambda: loss_xy + loss_wh + loss_conf + loss_class)

        loss = tf.cond(tf.less(seen, self.warmup_batches + 1),
                       lambda: loss_xy + loss_wh + loss_conf + loss_class  + 10,
                       lambda: loss_xy + loss_wh + loss_conf + loss_class )

        if self.debug:
            nb_true_box = tf.reduce_sum(y_true[..., 4])
            nb_true = tf.reduce_sum(tf.to_float(true_box_conf > 0.5))
            nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.5))

            current_recall = nb_pred_box / (nb_true_box + 1e-6)
            total_recall = tf.assign_add(total_recall, current_recall)

            loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
            loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
            loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
            loss = tf.Print(loss, [loss_conf_pos], message='Loss Conf + \t', summarize=1000)
            loss = tf.Print(loss, [loss_conf_neg], message='Loss Conf - \t', summarize=1000)
            loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
            loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
            loss = tf.Print(loss, [total_recall / seen], message='Average Recall \t', summarize=1000)

        return loss


    def stenosis_focal_loss(self, y_true, y_pred):
        mask_shape = tf.shape(y_true)[:4]

        coord_mask = tf.zeros(mask_shape)
        conf_mask = tf.zeros(mask_shape)

        seen = tf.Variable(0.)
        total_recall = tf.Variable(0.)
        total_precision = tf.Variable(0.)
        """
        Adjust prediction
        """
        ### adjust x and y
        pred_box_xy = tf.sigmoid(y_pred[..., :2])

        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.anchors, [1, 1, 1, self.nb_box, 2])

        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])

        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]

        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:2]  # relative position to the containing cell

        ### adjust w and h
        true_box_wh = y_true[..., 2:4]  # number of cells accross, horizontally and vertically

        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins = true_box_xy - true_wh_half
        true_maxes = true_box_xy + true_wh_half

        pred_wh_half = pred_box_wh / 2.
        pred_mins = pred_box_xy - pred_wh_half
        pred_maxes = pred_box_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        true_box_conf = iou_scores * y_true[..., 4]

        ### adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)

        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.coord_scale

        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = self.true_boxes[..., 0:2]
        true_wh = self.true_boxes[..., 2:4]

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        # refer to https://github.com/experiencor/keras-yolo2/issues/353
        conf_mask_neg = tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * self.no_object_scale
        conf_mask_pos = y_true[..., 4] * self.object_scale


        ### adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)
        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * tf.gather(self.class_wt, true_box_class) * self.class_scale
        true_box_class = y_true[..., 5:]
        alpha = 0.25
        gamma = 2.0


        true_box_class_labels = true_box_class[..., :-1]
        pred_box_class_labels = pred_box_class[..., :-1]
        anchor_state = true_box_class[...,  -1]  # -1 for ignore, -3 for background, -2 for object

        # filter out "ignore" anchors
        indices = tf.where(tf.not_equal(anchor_state, 1))
        true_box_class_labels = tf.gather_nd(true_box_class_labels, indices)
        pred_box_class_labels = tf.gather_nd(pred_box_class_labels, indices)

        true_box_class_labels = true_box_class[..., -2]#  -2 for object
        pred_box_class_labels = pred_box_class[..., -2]#  -2 for object

        # compute the focal loss
        alpha_factor = tf.ones_like(true_box_class_labels) * alpha
        alpha_factor = tf.where(tf.equal(true_box_class_labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(tf.equal(true_box_class_labels, 1), 1 - pred_box_class_labels, pred_box_class_labels)
        focal_weight = alpha_factor * focal_weight ** gamma

        seen = tf.assign_add(seen, 1.)
        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
        nb_conf_box_neg = tf.reduce_sum(tf.to_float(conf_mask_neg > 0.0))
        nb_conf_box_pos = tf.reduce_sum(tf.to_float(conf_mask_pos > 0.0))

        loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf_neg = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask_neg) / (
                    nb_conf_box_neg + 1e-6) / 2.
        loss_conf_pos = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask_pos) / (
                    nb_conf_box_pos + 1e-6) / 2.
        loss_conf =  loss_conf_pos + loss_conf_neg
        cls_loss = focal_weight * tf.keras.backend.binary_crossentropy(true_box_class_labels, pred_box_class_labels)
        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(tf.equal(true_box_class_labels, 1))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.keras.backend.floatx())
        normalizer = tf.maximum(tf.keras.backend.cast_to_floatx(1.0), normalizer)
        loss_focal_class = tf.keras.backend.sum(cls_loss) / (normalizer + 1e-10)



        #loss = tf.cond(tf.less(seen, self.warmup_batches + 1),
        #               lambda: loss_xy + loss_wh + loss_conf + loss_class + 10,
        #               lambda: loss_xy + loss_wh + loss_conf + loss_class)

        loss = loss_xy + loss_wh + loss_focal_class

        if self.debug:
            nb_true_box = tf.reduce_sum(y_true[..., 4])
            nb_pred_box = tf.reduce_sum(tf.to_float(pred_box_conf > 0.5))
            nb_true = tf.reduce_sum(tf.to_float(true_box_conf > 0.5))
            nb_truepred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.5))

            current_recall = nb_truepred_box / (nb_true_box + 1e-6)
            current_precision = nb_truepred_box / (nb_pred_box + 1e-6)
            total_recall = tf.assign_add(total_recall, current_recall)
            total_precision = tf.assign_add(total_precision, current_precision)

            loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
            loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
            loss = tf.Print(loss, [loss_focal_class], message='Loss Focal Class \t', summarize=1000)
            #loss = tf.Print(loss, [loss_focal_class], message='Loss Focal Class \t', summarize=1000)
            #loss = tf.Print(loss, [loss_conf_pos], message='Loss Conf + \t', summarize=1000)
            #loss = tf.Print(loss, [loss_conf_neg], message='Loss Conf - \t', summarize=1000)
            loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
            loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
            loss = tf.Print(loss, [total_recall / seen], message='Average Recall \t', summarize=1000)
            loss = tf.Print(loss, [current_precision], message='Current Precision \t', summarize=1000)
            loss = tf.Print(loss, [total_precision / seen], message='Average Precision \t', summarize=1000)
        return loss

    def stenosis_focal_loss_II(self, y_true, y_pred):
        mask_shape = tf.shape(y_true)[:4]

        seen = tf.Variable(0.)
        total_recall = tf.Variable(0.)
        total_precision = tf.Variable(0.)
        """
        Adjust prediction
        """
        ### adjust x and y
        pred_box_xy = tf.sigmoid(y_pred[..., :2])

        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.anchors, [1, 1, 1, self.nb_box, 2])

        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])

        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]

        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:2]  # relative position to the containing cell

        ### adjust w and h
        true_box_wh = y_true[..., 2:4]  # number of cells accross, horizontally and vertically

        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins = true_box_xy - true_wh_half
        true_maxes = true_box_xy + true_wh_half

        pred_wh_half = pred_box_wh / 2.
        pred_mins = pred_box_xy - pred_wh_half
        pred_maxes = pred_box_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        true_box_conf = iou_scores * y_true[..., 4]

        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.coord_scale

        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = self.true_boxes[..., 0:2]
        true_wh = self.true_boxes[..., 2:4]

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        alpha = self.focal_alpha
        gamma = self.focal_gamma

        true_box_class_labels = y_true[..., 4]  # 4 confidence as stenosis
        pred_box_class_labels = tf.sigmoid(y_pred[..., 4]) # 4 confidence as stenosis

        # compute the focal loss
        alpha_factor = tf.ones_like(true_box_class_labels) * alpha
        alpha_factor = tf.where(tf.equal(true_box_class_labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(tf.equal(true_box_class_labels, 1), 1 - pred_box_class_labels, pred_box_class_labels)
        focal_weight = alpha_factor * focal_weight ** gamma

        seen = tf.assign_add(seen, 1.)
        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))

        loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        cls_loss = focal_weight * tf.keras.backend.binary_crossentropy(true_box_class_labels, pred_box_class_labels)
        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(tf.equal(true_box_class_labels, 1))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.keras.backend.floatx())
        normalizer = tf.maximum(tf.keras.backend.cast_to_floatx(1.0), normalizer)
        loss_focal_class = tf.keras.backend.sum(cls_loss) / (normalizer + 1e-10)

        loss = loss_xy + loss_wh + loss_focal_class

        if self.debug:
            nb_true_box = tf.reduce_sum(y_true[..., 4])
            nb_pred_box = tf.reduce_sum(tf.to_float(pred_box_conf > 0.5))
            nb_truepred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.5))

            current_recall = nb_truepred_box / (nb_true_box + 1e-6)
            current_precision = nb_truepred_box / (nb_pred_box + 1e-6)
            total_recall = tf.assign_add(total_recall, current_recall)
            total_precision = tf.assign_add(total_precision, current_precision)

            loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
            loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
            loss = tf.Print(loss, [loss_focal_class], message='Loss Focal Class \t', summarize=1000)
            loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
            loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
            loss = tf.Print(loss, [total_recall / seen], message='Average Recall \t', summarize=1000)
            loss = tf.Print(loss, [current_precision], message='Current Precision \t', summarize=1000)
            loss = tf.Print(loss, [total_precision / seen], message='Average Precision \t', summarize=1000)
        return loss

    def stenosis_focal_loss_III(self, y_true, y_pred):
        mask_shape = tf.shape(y_true)[:4]

        coord_mask = tf.zeros(mask_shape)
        conf_mask = tf.zeros(mask_shape)

        seen = tf.Variable(0.)
        total_recall = tf.Variable(0.)
        total_precision = tf.Variable(0.)
        """
        Adjust prediction
        """
        ### adjust x and y
        pred_box_xy = tf.sigmoid(y_pred[..., :2])

        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.anchors, [1, 1, 1, self.nb_box, 2])

        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])

        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]

        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:2]  # relative position to the containing cell

        ### adjust w and h
        true_box_wh = y_true[..., 2:4]  # number of cells accross, horizontally and vertically

        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins = true_box_xy - true_wh_half
        true_maxes = true_box_xy + true_wh_half

        pred_wh_half = pred_box_wh / 2.
        pred_mins = pred_box_xy - pred_wh_half
        pred_maxes = pred_box_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        true_box_conf = iou_scores * y_true[..., 4]

        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.coord_scale

        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = self.true_boxes[..., 0:2]
        true_wh = self.true_boxes[..., 2:4]

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)

        ### adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)
        alpha = self.focal_alpha
        gamma = self.focal_gamma
        eps = 1e-12

        true_box_class_labels = y_true[..., 4]  # 4 confidence as stenosis
        pred_box_class_labels = y_pred[..., 4] # 4 confidence as stenosis
        pred_box_class_labels = K.clip(pred_box_class_labels, eps, 1. - eps)
        # compute the focal loss
        pt_1 = tf.where(tf.equal(true_box_class_labels, 1.0), pred_box_class_labels, tf.ones_like(pred_box_class_labels))
        pt_0 = tf.where(tf.equal(true_box_class_labels, 0.0), pred_box_class_labels, tf.zeros_like(pred_box_class_labels))

        seen = tf.assign_add(seen, 1.)
        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))

        loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.

        loss_focal_class = -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + eps)) - \
                           K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + eps))

        # loss = tf.cond(tf.less(seen, self.warmup_batches + 1),
        #               lambda: loss_xy + loss_wh + loss_conf + loss_class + 10,
        #               lambda: loss_xy + loss_wh + loss_conf + loss_class)

        loss = loss_xy + loss_wh + loss_focal_class

        if self.debug:
            nb_true_box = tf.reduce_sum(y_true[..., 4])
            nb_pred_box = tf.reduce_sum(tf.to_float(pred_box_conf > 0.5))
            nb_truepred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.5))

            current_recall = nb_truepred_box / (nb_true_box + 1e-6)
            current_precision = nb_truepred_box / (nb_pred_box + 1e-6)
            total_recall = tf.assign_add(total_recall, current_recall)
            total_precision = tf.assign_add(total_precision, current_precision)

            loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
            loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
            loss = tf.Print(loss, [loss_focal_class], message='Loss Focal Class \t', summarize=1000)

            loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
            loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
            loss = tf.Print(loss, [total_recall / seen], message='Average Recall \t', summarize=1000)
            loss = tf.Print(loss, [current_precision], message='Current Precision \t', summarize=1000)
            loss = tf.Print(loss, [total_precision / seen], message='Average Precision \t', summarize=1000)
        return loss

    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)
        #self.model.layers[1].layers[1].load_weights(weight_path)

    def train(self, train_imgs,     # the list of images to train the model
                    valid_imgs,     # the list of images used to validate the model
                    train_times,    # the number of time to repeat the training set, often used for small datasets
                    valid_times,    # the number of times to repeat the validation set, often used for small datasets
                    nb_epochs,      # number of epoches
                    learning_rate,  # the learning rate
                    batch_size,     # the size of the batch
                    warmup_epochs,  # number of initial batches to let the model familiarize with the new dataset
                    loss_type,
                    object_scale,
                    no_object_scale,
                    coord_scale,
                    class_scale,
                    focal_gamma=2.0,
                    focal_alpha=0.25,
                    saved_weights_name='best_weights.h5',
                    debug=False,
                    save_best_only=True,
                    valid_path=None):

        self.batch_size = batch_size

        self.object_scale    = object_scale
        self.no_object_scale = no_object_scale
        self.coord_scale     = coord_scale
        self.class_scale     = class_scale

        self.debug = debug
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        ############################################
        # Make train and validation generators
        ############################################

        generator_config = {
            'IMAGE_H'         : self.input_size, 
            'IMAGE_W'         : self.input_size,
            'GRID_H'          : self.grid_h,  
            'GRID_W'          : self.grid_w,
            'BOX'             : self.nb_box,
            'LABELS'          : self.labels,
            'CLASS'           : len(self.labels),
            'ANCHORS'         : self.anchors,
            'BATCH_SIZE'      : self.batch_size,
            'TRUE_BOX_BUFFER' : self.max_box_per_image,
        }    

        train_generator = YoloGenerator(train_imgs,
                                        generator_config,
                                        norm=self.feature_extractor.normalize)
        valid_generator = YoloGenerator(valid_imgs,
                                        generator_config,
                                        norm=self.feature_extractor.normalize,
                                        jitter=False)
                                     
        self.warmup_batches  = warmup_epochs * (train_times*len(train_generator) + valid_times*len(valid_generator))   

        ############################################
        # Compile the model
        ############################################
        loss_func = self.custom_loss
        if loss_type == "stenosis":
            loss_func = self.stenosis_loss
        elif loss_type == "focal":
            loss_func = self.stenosis_focal_loss
        elif loss_type == "focal_II":
            loss_func = self.stenosis_focal_loss_II
        elif loss_type == "focal_III":
            loss_func = self.stenosis_focal_loss_III
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss=loss_func, optimizer=optimizer)

        ############################################
        # Make a few callbacks
        ############################################
        num_patience = int(nb_epochs/4)
        early_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=1e-08,
                           patience=num_patience,
                           mode='min', 
                           verbose=1)
        checkpoint = ModelCheckpoint(saved_weights_name, 
                                     monitor='val_loss', 
                                     verbose=1, 
                                     save_best_only=save_best_only,
                                     mode='min', 
                                     period=1)
        tensorboard = TensorBoard(log_dir=os.path.expanduser('~/logs/'), 
                                  histogram_freq=0, 
                                  #write_batch_performance=True,
                                  write_graph=True, 
                                  write_images=False)

        ############################################
        # Start the training process
        ############################################        
        call_back_func = [early_stop, checkpoint, tensorboard]
        if not save_best_only:
            call_back_func = [checkpoint, tensorboard]
        self.model.fit_generator(generator        = train_generator, 
                                 steps_per_epoch  = len(train_generator) * train_times, 
                                 epochs           = warmup_epochs + nb_epochs, 
                                 verbose          = 2 if debug else 1,
                                 validation_data  = valid_generator,
                                 validation_steps = len(valid_generator) * valid_times,
                                 callbacks        = call_back_func,
                                 workers          = 3,
                                 max_queue_size   = 8,
                                 shuffle          = False)

        ############################################
        # Compute mAP on the validation set
        ############################################
        is_binary = False
        if loss_type == 'focal_II' or loss_type == 'focal_III':
            is_binary = True
        average_precision_0_5, recall_rate_0_5 = self.evaluate_stenose(valid_generator, 0.5, is_binary)
        average_precision_0_2, recall_rate_0_2 = self.evaluate_stenose(valid_generator, 0.2, is_binary)
        # print evaluation
        print('ap_0_5: {:.4f}'.format(average_precision_0_5))
        print('recall_0_5: {:.4f}'.format(recall_rate_0_5))
        print('ap_0_2: {:.4f}'.format(average_precision_0_2))
        print('recall_0_2: {:.4f}'.format(recall_rate_0_2))



    def evaluate(self, 
                 generator, 
                 iou_threshold,
                 score_threshold,
                 max_detections=100,
                 save_path=None):
        """ Evaluate a given dataset using a given model.
        code originally from https://github.com/fizyr/keras-retinanet

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
        all_detections     = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
        all_annotations    = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

        for i in range(generator.size()):
            raw_image = generator.load_image(i)
            raw_height, raw_width, raw_channels = raw_image.shape

            # make the boxes and the labels
            pred_boxes  = self.predict(raw_image)

            
            score = np.array([box.score for box in pred_boxes])
            pred_labels = np.array([box.label for box in pred_boxes])        
            
            if len(pred_boxes) > 0:
                pred_boxes = np.array([[box.xmin*raw_width, box.ymin*raw_height, box.xmax*raw_width, box.ymax*raw_height, box.score] for box in pred_boxes])
            else:
                pred_boxes = np.array([[]])  
            
            # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes  = pred_boxes[score_sort]
            
            # copy detections to all_detections
            for label in range(generator.num_classes()):
                all_detections[i][label] = []
                if any(pred_labels == label):
                    all_detections[i][label] = pred_boxes[pred_labels == label, :]
                
            annotations = generator.load_annotation(i)

            # copy detections to all_annotations
            for label in range(generator.num_classes()):
                all_annotations[i][label] = []
                if annotations.size == 0:
                    continue
                if any(annotations[:, 4] == label):
                    all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
                
        # compute mAP by comparing all detections and all annotations
        average_precisions = {}
        
        for label in range(generator.num_classes()):
            false_positives = np.zeros((0,))
            true_positives  = np.zeros((0,))
            scores          = np.zeros((0,))
            num_annotations = 0.0

            for i in range(generator.size()):
                detections           = all_detections[i][label]
                annotations          = all_annotations[i][label]
                num_annotations     += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)
                        continue

                    overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap         = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives  = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            # sort by score
            indices         = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives  = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives  = np.cumsum(true_positives)

            # compute recall and precision
            recall    = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision  = compute_ap(recall, precision)  
            average_precisions[label] = average_precision

        return average_precisions    

    def predict(self, image, is_binary=False):
        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = self.feature_extractor.normalize(image)

        #input_image = image[:,:,::-1]
        input_image = np.expand_dims(image, 0)
        dummy_array = np.zeros((1,1,1,1,self.max_box_per_image,4))

        netout = self.model.predict([input_image, dummy_array])[0]
        if is_binary:
            boxes = decode_stenoses_binary(netout, self.anchors, self.nb_class)
        else:
            boxes  = decode_stenoses(netout, self.anchors, self.nb_class)

        return boxes

    def evaluate_stenose(self,
                 generator,
                 iou_threshold=0.5,
                 is_binary=False):
        """ Evaluate a given dataset using a given model.
        code originally from https://github.com/fizyr/keras-retinanet

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
        # Returns
            A dict mapping class names to mAP scores.
        """
        # gather all detections and annotations
        all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
        all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

        for i in range(generator.size()):
            raw_image = generator.load_image(i)
            raw_image = cv2.resize(raw_image, (self.input_size, self.input_size))
            raw_height, raw_width, raw_channels = raw_image.shape

            # make the boxes and the labels
            pred_boxes = self.predict(raw_image, is_binary)

            score = np.array([box.score for box in pred_boxes])
            pred_labels = np.array([box.label for box in pred_boxes])

            if len(pred_boxes) > 0:
                pred_boxes = np.array([[box.xmin * raw_width, box.ymin * raw_height, box.xmax * raw_width,
                                        box.ymax * raw_height, box.score] for box in pred_boxes])
            else:
                pred_boxes = np.array([[]])

                # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes = pred_boxes[score_sort]

            # copy detections to all_detections
            for label in range(generator.num_classes()):
                all_detections[i][label] = []
                if any(pred_labels == label):
                    all_detections[i][label] = pred_boxes[pred_labels == label, :]

            annotations = generator.load_annotation(i)

            # copy detections to all_annotations
            for label in range(generator.num_classes()):
                all_annotations[i][label] = []
                if annotations.size == 0:
                    continue
                if any(annotations[:, 4] == label):
                    all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        # compute AP by comparing label 1 detections and label 1 annotations
        label = 1
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            if len(annotations) == 0:
                continue
            else:
                num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        cumsum_false_positives = np.cumsum(false_positives)
        cumsum_true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = cumsum_true_positives / num_annotations
        precision = cumsum_true_positives / np.maximum(cumsum_true_positives + cumsum_false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = compute_ap(recall, precision)
        recall_rate = np.sum(true_positives)/num_annotations

        return average_precision, recall_rate

    def evaluate_f1_iou(self,
                 generator,
                 iou_threshold=0.5,
                 is_binary=False):
        """ Evaluate a given dataset using a given model.
        code originally from https://github.com/fizyr/keras-retinanet

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
        # Returns
            A dict mapping class names to mAP scores.
        """
        # gather all detections and annotations
        all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
        all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

        for i in range(generator.size()):
            raw_image = generator.load_image(i)
            raw_image = cv2.resize(raw_image, (self.input_size, self.input_size))
            raw_height, raw_width, raw_channels = raw_image.shape

            # make the boxes and the labels
            pred_boxes = self.predict(raw_image, is_binary)

            score = np.array([box.score for box in pred_boxes])
            pred_labels = np.array([box.label for box in pred_boxes])

            if len(pred_boxes) > 0:
                pred_boxes = np.array([[box.xmin * raw_width, box.ymin * raw_height, box.xmax * raw_width,
                                        box.ymax * raw_height, box.score] for box in pred_boxes])
            else:
                pred_boxes = np.array([[]])

                # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes = pred_boxes[score_sort]

            # copy detections to all_detections
            for label in range(generator.num_classes()):
                all_detections[i][label] = []
                if any(pred_labels == label):
                    all_detections[i][label] = pred_boxes[pred_labels == label, :]

            annotations = generator.load_annotation(i)

            # copy detections to all_annotations
            for label in range(generator.num_classes()):
                all_annotations[i][label] = []
                if annotations.size == 0:
                    continue
                if any(annotations[:, 4] == label):
                    all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        # compute AP by comparing label 1 detections and label 1 annotations
        label = 1
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            if len(annotations) == 0:
                continue
            else:
                num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = compute_ap(recall, precision)

        return average_precision



class RetinaNet(object):
    """docstring for ClassName"""
    def __init__(self, backend,
                       input_size,
                       labels,
                       max_box_per_image,
                       anchors,
                       label_wts=None,
                       feature_extractor_weights = None,
                       feature_trainable = True):
        self.input_size = input_size
        self.backend = backend
        self.labels = list(labels)
        self.nb_class = len(self.labels)
        if self.nb_class == 1:
            self.nb_class = 2
        self.nb_box = len(anchors) // 2
        if label_wts == None:
            self.class_wt = np.ones(self.nb_class, dtype='float32')
        else:
            self.class_wt = label_wts
        self.anchors = anchors

        self.max_box_per_image = max_box_per_image

        ##########################
        # Make the model
        ##########################

        # make the feature extractor layers
        input_image = Input(shape=(self.input_size, self.input_size, 3))
        self.true_boxes = Input(shape=(1, 1, 1, max_box_per_image, 4))

        img_input = np.zeros((512, 512, 3))
        img_input = np.expand_dims(img_input, axis=0)
        pred_output = np.zeros((1, 1, 3))
        num_classes = self.nb_class
        coronary_model = inceptionV3_coronary_model(img_input, pred_output)
        if feature_extractor_weights != None and os.path.exists(feature_extractor_weights):
            coronary_model.load_weights(feature_extractor_weights)
        elif feature_extractor_weights != None and not os.path.exists(feature_extractor_weights):
            print("Cannot find feature extractor weight file from: ", feature_extractor_weights)

        submodels = self.default_submodels(num_classes, self.nb_box)
        C9_layer = 310
        C8_layer = 228
        C3_layer = 86
        C9 = coronary_model.layers[C9_layer].output
        C8 = coronary_model.layers[C8_layer].output
        C3 = coronary_model.layers[C3_layer].output
        # compute pyramid features as per https://arxiv.org/abs/1708.02002
        features = self.__create_pyramid_features(C3, C8, C9)
        pyramids = self.__build_pyramid(submodels, features)
        self.model = Model(coronary_model.layers[0].input, pyramids)
        # print a summary of the whole model
        self.model.summary()

    def normalize(self, image):
        preprocess_img = preprocess_input(image.astype('float64'))
        return preprocess_img

    def __create_pyramid_features(self, C3, C4, C5, feature_size=256):
        """ Creates the FPN layers on top of the backbone features.

        Args
            C3           : Feature stage C3 from the backbone.
            C4           : Feature stage C4 from the backbone.
            C5           : Feature stage C5 from the backbone.
            feature_size : The feature size to use for the resulting feature levels.

        Returns
            A list of feature levels [P3, P4, P5, P6, P7].
        """
        # upsample C5 to get P5 from the FPN paper
        P5           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
        P5_upsampled = UpsampleLike(name='P5_upsampled')([P5, C4])
        P5           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

        # add P5 elementwise to C4
        P4           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
        P4           = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
        P4_upsampled = UpsampleLike(name='P4_upsampled')([P4, C3])
        P4           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

        # add P4 elementwise to C3
        P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
        P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
        P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
        P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

        return [P3, P4, P5, P6, P7]

    def __build_model_pyramid(self, name, model, features):
        """ Applies a single submodel to each FPN level.

        Args
            name     : Name of the submodel.
            model    : The submodel to evaluate.
            features : The FPN features.

        Returns
            A tensor containing the response from the submodel on the FPN features.
        """
        return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])

    def __build_pyramid(self, models, features):
        """ Applies all submodels to each FPN level.

        Args
            models   : List of sumodels to run on each pyramid level (by default only regression, classifcation).
            features : The FPN features.

        Returns
            A list of tensors, one for each submodel.
        """
        return [self.__build_model_pyramid(n, m, features) for n, m in models]

    def default_submodels(self, num_classes, num_anchors):
        """ Create a list of default submodels used for object detection.

        The default submodels contains a regression submodel and a classification submodel.

        Args
            num_classes : Number of classes to use.
            num_anchors : Number of base anchors.

        Returns
            A list of tuple, where the first element is the name of the submodel and the second element is the submodel itself.
        """
        return [
            ('regression', self.default_regression_model(4, num_anchors)),
            ('classification', self.default_classification_model(num_classes, num_anchors))
        ]

    def default_regression_model(self, num_values, num_anchors, pyramid_feature_size=256, regression_feature_size=256,
                                 name='regression_submodel'):
        """ Creates the default regression submodel.

        Args
            num_values              : Number of values to regress.
            num_anchors             : Number of anchors to regress for each feature level.
            pyramid_feature_size    : The number of filters to expect from the feature pyramid levels.
            regression_feature_size : The number of filters to use in the layers in the regression submodel.
            name                    : The name of the submodel.

        Returns
            A keras.models.Model that predicts regression values for each anchor.
        """
        # All new conv layers except the final one in the
        # RetinaNet (classification) subnets are initialized
        # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'kernel_initializer': keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            'bias_initializer': 'zeros'
        }

        if keras.backend.image_data_format() == 'channels_first':
            inputs = keras.layers.Input(shape=(pyramid_feature_size, None, None))
        else:
            inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
        outputs = inputs
        for i in range(4):
            outputs = keras.layers.Conv2D(
                filters=regression_feature_size,
                activation='relu',
                name='pyramid_regression_{}'.format(i),
                **options
            )(outputs)

        outputs = keras.layers.Conv2D(num_anchors * num_values, name='pyramid_regression', **options)(outputs)
        if keras.backend.image_data_format() == 'channels_first':
            outputs = keras.layers.Permute((2, 3, 1), name='pyramid_regression_permute')(outputs)
        outputs = keras.layers.Reshape((-1, num_values), name='pyramid_regression_reshape')(outputs)

        return keras.models.Model(inputs=inputs, outputs=outputs, name=name)

    def default_classification_model(
            self,
            num_classes,
            num_anchors,
            pyramid_feature_size=256,
            prior_probability=0.01,
            classification_feature_size=256,
            name='classification_submodel'
    ):
        """ Creates the default regression submodel.

        Args
            num_classes                 : Number of classes to predict a score for at each feature level.
            num_anchors                 : Number of anchors to predict classification scores for at each feature level.
            pyramid_feature_size        : The number of filters to expect from the feature pyramid levels.
            classification_feature_size : The number of filters to use in the layers in the classification submodel.
            name                        : The name of the submodel.

        Returns
            A keras.models.Model that predicts classes for each anchor.
        """
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
        }

        if keras.backend.image_data_format() == 'channels_first':
            inputs = keras.layers.Input(shape=(pyramid_feature_size, None, None))
        else:
            inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
        outputs = inputs
        for i in range(4):
            outputs = keras.layers.Conv2D(
                filters=classification_feature_size,
                activation='relu',
                name='pyramid_classification_{}'.format(i),
                kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
                bias_initializer='zeros',
                **options
            )(outputs)

        outputs = keras.layers.Conv2D(
            filters=num_classes * num_anchors,
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer=initializers.PriorProbability(probability=prior_probability),
            name='pyramid_classification',
            **options
        )(outputs)

        # reshape output and apply sigmoid
        if keras.backend.image_data_format() == 'channels_first':
            outputs = keras.layers.Permute((2, 3, 1), name='pyramid_classification_permute')(outputs)
        outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
        outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

        return keras.models.Model(inputs=inputs, outputs=outputs, name=name)

    def stenosis_focal_loss(self, y_true, y_pred):

        """
        Determine the masks
        """

        alpha = self.focal_alpha
        gamma = self.focal_gamma

        labels = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
        classification = y_pred

        # filter out "ignore" anchors
        indices = tf.where(keras.backend.not_equal(anchor_state, -1))
        labels = tf.gather_nd(labels, indices)
        classification = tf.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = tf.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        loss = keras.backend.sum(cls_loss) / normalizer

        if self.debug:
            loss = tf.Print(loss, [loss], message='Current focal loss \t', summarize=1000)
        return loss

    def stenosis_focal_loss_II(self, y_true, y_pred):
        alpha = self.focal_alpha
        gamma = self.focal_gamma
        labels = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
        classification = y_pred

        # filter out "ignore" anchors
        indices = tf.where(keras.backend.not_equal(anchor_state, -1))
        labels = tf.gather_nd(labels, indices)
        classification = tf.gather_nd(classification, indices)
        true_box_class_labels = labels[..., 1]
        pred_box_class_labels = classification[...,1]
        # compute the focal loss
        alpha_factor = tf.ones_like(true_box_class_labels) * alpha
        alpha_factor = tf.where(tf.equal(true_box_class_labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(tf.equal(true_box_class_labels, 1), 1 - pred_box_class_labels, pred_box_class_labels)
        focal_weight = alpha_factor * focal_weight ** gamma

        """
        Finalize the loss
        """
        cls_loss = focal_weight * tf.keras.backend.binary_crossentropy(true_box_class_labels, pred_box_class_labels)
        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(tf.equal(true_box_class_labels, 1))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.keras.backend.floatx())
        normalizer = tf.maximum(tf.keras.backend.cast_to_floatx(1.0), normalizer)
        loss_focal_class = tf.keras.backend.sum(cls_loss) / (normalizer + 1e-10)

        loss = loss_focal_class

        if self.debug:
            loss = tf.Print(loss, [loss_focal_class], message='Loss Focal Class \t', summarize=1000)

        return loss

    def stenosis_l2_loss(self, y_true, y_pred):
        sigma = self.l2_sigma
        sigma_squared = sigma ** 2
        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices           = tf.where(keras.backend.equal(anchor_state, 1))
        regression        = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        loss =  keras.backend.sum(regression_loss) / normalizer * self.coord_scale

        if self.debug:
            loss = tf.Print(loss, [loss], message='L2 Loss \t', summarize=1000)
        return loss

    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)
        # self.model.layers[1].layers[1].load_weights(weight_path)

    def train(self, train_imgs,  # the list of images to train the model
              valid_imgs,  # the list of images used to validate the model
              train_times,  # the number of time to repeat the training set, often used for small datasets
              valid_times,  # the number of times to repeat the validation set, often used for small datasets
              nb_epochs,  # number of epoches
              learning_rate,  # the learning rate
              batch_size,  # the size of the batch
              warmup_epochs,  # number of initial batches to let the model familiarize with the new dataset
              loss_type,
              object_scale,
              no_object_scale,
              coord_scale,
              class_scale,
              focal_gamma=2.0,
              focal_alpha=0.25,
              l2_sigma=3.0,
              saved_weights_name='best_weights.h5',
              debug=False,
              save_best_only=True,
              valid_path=None):

        self.batch_size = batch_size

        self.object_scale = object_scale
        self.no_object_scale = no_object_scale
        self.coord_scale = coord_scale
        self.class_scale = class_scale

        self.debug = debug
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.l2_sigma = l2_sigma
        ############################################
        # Make train and validation generators
        ############################################

        generator_config = {
            'IMAGE_H': self.input_size,
            'IMAGE_W': self.input_size,
            'BOX': self.nb_box,
            'LABELS': self.labels,
            'CLASS': len(self.labels),
            'ANCHORS': self.anchors,
            'BATCH_SIZE': self.batch_size,
            'TRUE_BOX_BUFFER': self.max_box_per_image,
        }

        train_generator = RetinaGenerator(train_imgs,
                                          generator_config,
                                          norm=self.normalize)
        valid_generator = RetinaGenerator(valid_imgs,
                                          generator_config,
                                          norm=self.normalize,
                                          jitter=False)

        self.warmup_batches = warmup_epochs * (train_times * len(train_generator) + valid_times * len(valid_generator))

        ############################################
        # Compile the model
        ############################################
        loss_func = {
            'regression'    : self.stenosis_l2_loss,
            'classification': self.stenosis_focal_loss
        }

        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        self.model.compile(loss=loss_func, optimizer=optimizer)

        ############################################
        # Make a few callbacks
        ############################################
        num_patience = int(nb_epochs / 4)
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=1e-08,
                                   patience=num_patience,
                                   mode='min',
                                   verbose=1)
        checkpoint = ModelCheckpoint(saved_weights_name,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=save_best_only,
                                     mode='min',
                                     period=1)
        tensorboard = TensorBoard(log_dir=os.path.expanduser('~/logs/'),
                                  histogram_freq=0,
                                  # write_batch_performance=True,
                                  write_graph=True,
                                  write_images=False)

        ############################################
        # Start the training process
        ############################################
        call_back_func = [early_stop, checkpoint, tensorboard]
        if not save_best_only:
            call_back_func = [checkpoint, tensorboard]
        self.model.fit_generator(generator=train_generator,
                                 steps_per_epoch=len(train_generator) * train_times,
                                 epochs=warmup_epochs + nb_epochs,
                                 verbose=2 if debug else 1,
                                 validation_data=valid_generator,
                                 validation_steps=len(valid_generator) * valid_times,
                                 callbacks=call_back_func,
                                 workers=3,
                                 max_queue_size=8,
                                 shuffle=False)

        # load the best version of the model
        self.model.load_weights(saved_weights_name)

        self.evaluate(
            generator=valid_generator,
            iou_threshold=0.5,
            score_threshold=0.5,
            max_detections=100,
            save_path=valid_path)



    def evaluate(self,
                 generator,
                 iou_threshold,
                 score_threshold,
                 max_detections=100,
                 save_path=None):
        """ Evaluate a given dataset using a given model.
        code originally from https://github.com/fizyr/keras-retinanet

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
        eval_res = evaluate_retina(generator,
                                             self.model,
                                             iou_threshold=iou_threshold,
                                             score_threshold=score_threshold,
                                             max_detections=max_detections,
                                             save_path=save_path)
        return eval_res

    def predict(self, image, is_binary=False):
        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = self.normalize(image)

        # input_image = image[:,:,::-1]
        input_image = np.expand_dims(image, 0)
        dummy_array = np.zeros((1, 1, 1, 1, self.max_box_per_image, 4))

        netout = self.model.predict([input_image, dummy_array])[0]
        if is_binary:
            boxes = decode_stenoses_binary(netout, self.anchors, self.nb_class)
        else:
            boxes = decode_stenoses(netout, self.anchors, self.nb_class)

        return boxes

    def evaluate_stenose(self,
                         generator,
                         iou_threshold=0.5,
                         is_binary=False):
        """ Evaluate a given dataset using a given model.
        code originally from https://github.com/fizyr/keras-retinanet

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
        # Returns
            A dict mapping class names to mAP scores.
        """
        # gather all detections and annotations
        all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
        all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

        for i in range(generator.size()):
            raw_image = generator.load_image(i)
            raw_image = cv2.resize(raw_image, (self.input_size, self.input_size))
            raw_height, raw_width, raw_channels = raw_image.shape

            # make the boxes and the labels
            pred_boxes = self.predict(raw_image, is_binary)

            score = np.array([box.score for box in pred_boxes])
            pred_labels = np.array([box.label for box in pred_boxes])

            if len(pred_boxes) > 0:
                pred_boxes = np.array([[box.xmin * raw_width, box.ymin * raw_height, box.xmax * raw_width,
                                        box.ymax * raw_height, box.score] for box in pred_boxes])
            else:
                pred_boxes = np.array([[]])

                # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes = pred_boxes[score_sort]

            # copy detections to all_detections
            for label in range(generator.num_classes()):
                all_detections[i][label] = []
                if any(pred_labels == label):
                    all_detections[i][label] = pred_boxes[pred_labels == label, :]

            annotations = generator.load_annotation(i)

            # copy detections to all_annotations
            for label in range(generator.num_classes()):
                all_annotations[i][label] = []
                if annotations.size == 0:
                    continue
                if any(annotations[:, 4] == label):
                    all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        # compute AP by comparing label 1 detections and label 1 annotations
        label = 1
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            if len(annotations) == 0:
                continue
            else:
                num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        cumsum_false_positives = np.cumsum(false_positives)
        cumsum_true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = cumsum_true_positives / num_annotations
        precision = cumsum_true_positives / np.maximum(cumsum_true_positives + cumsum_false_positives,
                                                       np.finfo(np.float64).eps)

        # compute average precision
        average_precision = compute_ap(recall, precision)
        recall_rate = np.sum(true_positives) / num_annotations

        return average_precision, recall_rate

    def evaluate_f1_iou(self,
                        generator,
                        iou_threshold=0.5,
                        is_binary=False):
        """ Evaluate a given dataset using a given model.
        code originally from https://github.com/fizyr/keras-retinanet

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
        # Returns
            A dict mapping class names to mAP scores.
        """
        # gather all detections and annotations
        all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
        all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

        for i in range(generator.size()):
            raw_image = generator.load_image(i)
            raw_image = cv2.resize(raw_image, (self.input_size, self.input_size))
            raw_height, raw_width, raw_channels = raw_image.shape

            # make the boxes and the labels
            pred_boxes = self.predict(raw_image, is_binary)

            score = np.array([box.score for box in pred_boxes])
            pred_labels = np.array([box.label for box in pred_boxes])

            if len(pred_boxes) > 0:
                pred_boxes = np.array([[box.xmin * raw_width, box.ymin * raw_height, box.xmax * raw_width,
                                        box.ymax * raw_height, box.score] for box in pred_boxes])
            else:
                pred_boxes = np.array([[]])

                # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes = pred_boxes[score_sort]

            # copy detections to all_detections
            for label in range(generator.num_classes()):
                all_detections[i][label] = []
                if any(pred_labels == label):
                    all_detections[i][label] = pred_boxes[pred_labels == label, :]

            annotations = generator.load_annotation(i)

            # copy detections to all_annotations
            for label in range(generator.num_classes()):
                all_annotations[i][label] = []
                if annotations.size == 0:
                    continue
                if any(annotations[:, 4] == label):
                    all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        # compute AP by comparing label 1 detections and label 1 annotations
        label = 1
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            if len(annotations) == 0:
                continue
            else:
                num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = compute_ap(recall, precision)

        return average_precision
