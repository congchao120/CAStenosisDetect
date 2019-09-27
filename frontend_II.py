from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
import tensorflow as tf
import numpy as np
import os
import cv2
from utils import decode_featuremap, compute_overlap, compute_ap
from keras.applications.mobilenet import MobileNet
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam, RMSprop
from generator import YoloGenerator_II
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from backend import TinyYoloFeature, FullYoloFeature, MobileNetFeature, SqueezeNetFeature, Inception3Feature, VGG16Feature, ResNet50Feature, InceptCoronaryFeature

'''
Only use focal loss on final feature map as loss function
'''
class YOLO_II(object):
    def __init__(self, backend,
                       input_size, 
                       labels,
                       max_box_per_image,
                       label_wts=None,
                       feature_extractor_weights = None,
                       feature_trainable = True):

        self.input_size = input_size
        
        self.labels   = list(labels)
        self.nb_class = len(self.labels)
        if self.nb_class == 1:
            self.nb_class = 2
        if label_wts == None:
            self.class_wt = np.ones(self.nb_class, dtype='float32')
        else:
            self.class_wt = label_wts

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
        output = Conv2D(1,
                        (1,1), strides=(1,1), 
                        padding='same', 
                        name='DetectionLayer', 
                        kernel_initializer='lecun_normal')(features)
        output = Reshape((self.grid_h, self.grid_w, 1))(output)
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

    def featuremap_focal_loss(self, y_true, y_pred):
        seen = tf.Variable(0.)
        total_recall = tf.Variable(0.)
        total_precision = tf.Variable(0.)

        alpha = self.focal_alpha
        gamma = self.focal_gamma

        true_box_class_labels = y_true  # confidence as stenosis
        pred_box_class_labels = tf.sigmoid(y_pred) # confidence as stenosis

        # compute the focal loss
        alpha_factor = tf.ones_like(true_box_class_labels) * alpha
        alpha_factor = tf.where(tf.equal(true_box_class_labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(tf.equal(true_box_class_labels, 1), 1 - pred_box_class_labels, pred_box_class_labels)
        focal_weight = alpha_factor * focal_weight ** gamma

        seen = tf.assign_add(seen, 1.)
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
            nb_true_box = tf.reduce_sum(y_true)
            nb_pred_box = tf.reduce_sum(tf.to_float(pred_box_class_labels > 0.5))
            nb_truepred_box = tf.reduce_sum(tf.to_float(y_true > 0.5) * tf.to_float(pred_box_class_labels > 0.5))

            current_recall = nb_truepred_box / (nb_true_box + 1e-6)
            current_precision = nb_truepred_box / (nb_pred_box + 1e-6)
            total_recall = tf.assign_add(total_recall, current_recall)
            total_precision = tf.assign_add(total_precision, current_precision)

            loss = tf.Print(loss, [loss_focal_class], message='Loss Focal Class \t', summarize=1000)
            loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
            loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
            loss = tf.Print(loss, [total_recall / seen], message='Average Recall \t', summarize=1000)
            loss = tf.Print(loss, [current_precision], message='Current Precision \t', summarize=1000)
            loss = tf.Print(loss, [total_precision / seen], message='Average Precision \t', summarize=1000)
        return loss

    def f1_metric(self, y_true, y_pred):
        true_box_class_labels = y_true  # confidence as stenosis
        pred_box_class_labels = tf.sigmoid(y_pred) # confidence as stenosis
        nb_true_box = tf.reduce_sum(true_box_class_labels)
        nb_pred_box = tf.reduce_sum(tf.to_float(pred_box_class_labels > 0.5))
        nb_tp_box = tf.reduce_sum(tf.to_float(true_box_class_labels > 0.5) * tf.to_float(pred_box_class_labels > 0.5))
        f1_score = 2 * nb_tp_box / (nb_true_box + nb_pred_box + 1e-6)

        return f1_score

    def IOU_metric(self, y_true, y_pred):
        true_box_class_labels = y_true  # confidence as stenosis
        pred_box_class_labels = tf.sigmoid(y_pred)  # confidence as stenosis
        nb_intersect_box = tf.reduce_sum(tf.to_float(true_box_class_labels > 0.5) * tf.to_float(pred_box_class_labels > 0.5))
        nb_union_box = tf.reduce_sum(tf.to_float(tf.logical_or(true_box_class_labels > 0.5, pred_box_class_labels > 0.5)))
        IOU = nb_intersect_box / (nb_union_box + 1e-6)

        return IOU

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
                    focal_gamma=2.0,
                    focal_alpha=0.25,
                    saved_weights_name='best_weights.h5',
                    debug=False,
                    save_best_only=True):

        self.batch_size = batch_size

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
            'LABELS'          : self.labels,
            'CLASS'           : len(self.labels),
            'BATCH_SIZE'      : self.batch_size,
            'TRUE_BOX_BUFFER' : self.max_box_per_image,
        }    

        train_generator = YoloGenerator_II(train_imgs,
                                           generator_config,
                                           norm=self.feature_extractor.normalize)
        valid_generator = YoloGenerator_II(valid_imgs,
                                           generator_config,
                                           norm=self.feature_extractor.normalize,
                                           jitter=False)
                                     
        self.warmup_batches  = warmup_epochs * (train_times*len(train_generator) + valid_times*len(valid_generator))   

        ############################################
        # Compile the model
        ############################################
        loss_func = self.featuremap_focal_loss
        if loss_type == "stenosis":
            loss_func = self.featuremap_focal_loss
        elif loss_type == "focal":
            loss_func = self.featuremap_focal_loss
        elif loss_type == "focal_II":
            loss_func = self.featuremap_focal_loss
        elif loss_type == "focal_III":
            loss_func = self.featuremap_focal_loss
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        metrics = [self.f1_metric, self.IOU_metric]
        self.model.compile(loss=loss_func, optimizer=optimizer, metrics=metrics)

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
                                     monitor='val_f1_metric',
                                     verbose=1, 
                                     save_best_only=save_best_only,
                                     mode='max',
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
        average_precision_0_1 = self.evaluate_stenose(valid_generator, obj_threshold=0.1)
        average_precision_0_2 = self.evaluate_stenose(valid_generator, obj_threshold=0.2)
        average_precision_0_5 = self.evaluate_stenose(valid_generator, obj_threshold=0.5)
        average_precision = self.evaluate_stenose_ap(valid_generator)
        recall_obj_0_5, recall_img_0_5 = self.evaluate_stenose_rec(valid_generator)
        recall_obj_0_2, recall_img_0_2 = self.evaluate_stenose_rec(valid_generator, obj_threshold=0.2, iou_threshold=0.2)
        # print evaluation
        print('ap_0_1: {:.4f}'.format(average_precision_0_1))
        print('ap_0_2: {:.4f}'.format(average_precision_0_2))
        print('ap_0_5: {:.4f}'.format(average_precision_0_5))
        print(self.labels[1], '{:.4f}'.format(average_precision))
        print('recall_obj_0_5: {:.4f}'.format(recall_obj_0_5))
        print('recall_img_0_5: {:.4f}'.format(recall_img_0_5))
        print('recall_obj_0_2: {:.4f}'.format(recall_obj_0_2))
        print('recall_img_0_2: {:.4f}'.format(recall_img_0_2))



    def evaluate(self, 
                 generator, 
                 iou_threshold):
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
            pred_maps  = self.predict(raw_image)

            
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

    def predict(self, image):
        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = self.feature_extractor.normalize(image)

        #input_image = image[:,:,::-1]
        input_image = np.expand_dims(image, 0)
        dummy_array = np.zeros((1,1,1,1,self.max_box_per_image,4))

        netout = self.model.predict([input_image, dummy_array])[0]
        pred_map = decode_featuremap(netout)

        return pred_map

    def evaluate_stenose(self,
                 generator,
                 obj_threshold=0.1,
                 iou_threshold=0.5):
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
        all_annotation_fms = np.zeros((generator.size(), self.grid_h, self.grid_w))
        all_prediction_fms = np.zeros((generator.size(), self.grid_h, self.grid_w))
        all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

        for i in range(generator.size()):
            raw_image = generator.load_image(i)
            raw_image = cv2.resize(raw_image, (self.input_size, self.input_size))

            # make the boxes and the labels
            pred_maps = self.predict(raw_image)
            pred_masks = (pred_maps > obj_threshold)[...,0].astype(float)
            all_prediction_fms[i] = pred_masks
            annotations = generator.load_annotation(i)
            true_masks = generator.load_true_featuremap(i)
            all_annotation_fms[i] =  true_masks

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
            annot_fm = all_annotation_fms[i]
            pred_fm = all_prediction_fms[i]
            score_fm = pred_maps[...,0]
            #element-wise product
            intersect = annot_fm*pred_fm
            union = np.logical_or(annot_fm, pred_fm).astype(float)
            if np.sum(annot_fm) == 0:
                continue
            else:
                num_annotations += np.sum(annot_fm)

            fp = 0
            tp = 0
            for w in range(self.grid_w):
                for h in range(self.grid_h):
                    if pred_fm[w, h] == 1:
                        score = score_fm[w, h]
                        if intersect[w, h] == 1:
                            fp = 0
                            tp = 1
                        else:
                            fp = 1
                            tp = 0
                        scores = np.append(scores, score)
                        false_positives = np.append(false_positives, fp)
                        true_positives = np.append(true_positives, tp)

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



    def evaluate_stenose_ap(self,
                 generator,
                 obj_threshold=0.1,
                 iou_threshold=0.5):
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
        all_annotation_fms = np.zeros((generator.size(), self.grid_h, self.grid_w))
        all_prediction_fms = np.zeros((generator.size(), self.grid_h, self.grid_w))
        all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

        for i in range(generator.size()):
            raw_image = generator.load_image(i)
            raw_image = cv2.resize(raw_image, (self.input_size, self.input_size))

            # make the boxes and the labels
            pred_maps = self.predict(raw_image)
            pred_masks = (pred_maps > 0)[...,0].astype(float)
            all_prediction_fms[i] = pred_masks
            annotations = generator.load_annotation(i)
            true_masks = generator.load_true_featuremap(i)
            all_annotation_fms[i] =  true_masks

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
            annot_fm = all_annotation_fms[i]
            pred_fm = all_prediction_fms[i]
            score_fm = pred_maps[...,0]
            #element-wise product
            intersect = annot_fm*pred_fm
            union = np.logical_or(annot_fm, pred_fm).astype(float)
            if np.sum(annot_fm) == 0:
                continue
            else:
                num_annotations += np.sum(annot_fm)

            fp = 0
            tp = 0
            for w in range(self.grid_w):
                for h in range(self.grid_h):
                    if pred_fm[w, h] == 1:
                        score = score_fm[w, h]
                        if intersect[w, h] == 1:
                            fp = 0
                            tp = 1
                        else:
                            fp = 1
                            tp = 0
                        scores = np.append(scores, score)
                        false_positives = np.append(false_positives, fp)
                        true_positives = np.append(true_positives, tp)

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        cum_sum_false_positives = np.cumsum(false_positives)
        cum_sum_true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = cum_sum_true_positives / num_annotations
        precision = cum_sum_true_positives / np.maximum(cum_sum_true_positives + cum_sum_false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = compute_ap(recall, precision)

        return average_precision


    def evaluate_stenose_rec(self,
                 generator,
                 obj_threshold=0.5,
                 iou_threshold=0.5):
        """ Evaluate a given dataset using a given model, calculate recall in 2 different ways: per obj (> obj_threshold) or per image (> iou_threshold)
        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            obj_threshold   : The threshold used to consider when a detection is positive or negative.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
        # Returns
            A dict mapping class names to mAP scores.
        """
        # gather all detections and annotations
        all_annotation_fms = np.zeros((generator.size(), self.grid_h, self.grid_w))
        all_prediction_fms = np.zeros((generator.size(), self.grid_h, self.grid_w))

        for i in range(generator.size()):
            raw_image = generator.load_image(i)
            raw_image = cv2.resize(raw_image, (self.input_size, self.input_size))

            # make the boxes and the labels
            pred_maps = self.predict(raw_image)
            pred_masks = (pred_maps > obj_threshold)[...,0].astype(float)
            all_prediction_fms[i] = pred_masks
            annotations = generator.load_annotation(i)
            true_masks = generator.load_true_featuremap(i)
            all_annotation_fms[i] =  true_masks

        # compute AP by comparing label 1 detections and label 1 annotations
        num_obj_annotations = 0.0
        num_obj_tp = 0.0
        num_img_tp = 0.0
        for i in range(generator.size()):
            annot_fm = all_annotation_fms[i]
            pred_fm = all_prediction_fms[i]

            #element-wise product
            intersect = annot_fm*pred_fm
            union = np.logical_or(annot_fm, pred_fm).astype(float)
            if np.sum(annot_fm) != 0:
                num_obj_annotations += np.sum(annot_fm)
                num_obj_tp += np.sum(intersect)
                if np.sum(intersect)/np.sum(union) >= iou_threshold:
                    num_img_tp += 1
            elif np.sum(annot_fm) == 0 and np.sum(pred_fm) == 0:
                num_img_tp += 1

        # compute recall/sensitivity
        recall_obj = num_obj_tp / np.maximum(num_obj_annotations, np.finfo(np.float64).eps)
        recall_img = num_img_tp / np.maximum(generator.size(), np.finfo(np.float64).eps)

        return recall_obj, recall_img

    def evaluate_stenose_f1_iou(self,
                 generator,
                 obj_threshold=0.5):
        """ Evaluate a given dataset using a given model, calculate recall in 2 different ways: per obj (> obj_threshold) or per image (> iou_threshold)
        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            obj_threshold   : The threshold used to consider when a detection is positive or negative.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
        # Returns
            A dict mapping class names to mAP scores.
        """
        # gather all detections and annotations
        all_annotation_fms = np.zeros((generator.size(), self.grid_h, self.grid_w))
        all_prediction_fms = np.zeros((generator.size(), self.grid_h, self.grid_w))

        for i in range(generator.size()):
            raw_image = generator.load_image(i)
            raw_image = cv2.resize(raw_image, (self.input_size, self.input_size))

            # make the boxes and the labels
            pred_maps = self.predict(raw_image)
            pred_masks = (pred_maps > obj_threshold)[...,0].astype(float)
            all_prediction_fms[i] = pred_masks
            annotations = generator.load_annotation(i)
            true_masks = generator.load_true_featuremap(i)
            all_annotation_fms[i] =  true_masks

        # compute AP by comparing label 1 detections and label 1 annotations
        num_obj_annotations = 0.0
        num_obj_detections = 0.0
        num_obj_tp = 0.0
        num_obj_union = 0.0
        for i in range(generator.size()):
            annot_fm = all_annotation_fms[i]
            pred_fm = all_prediction_fms[i]

            #element-wise product
            intersect = annot_fm*pred_fm
            union = np.logical_or(annot_fm, pred_fm).astype(float)
            if np.sum(annot_fm) != 0:
                num_obj_annotations += np.sum(annot_fm)
                num_obj_detections += np.sum(pred_fm)
                num_obj_tp += np.sum(intersect)
                num_obj_union += np.sum(union)
        # compute recall/sensitivity
        f1_score = num_obj_tp / np.maximum((num_obj_annotations + num_obj_detections)/2.0, np.finfo(np.float64).eps)
        iou_score = num_obj_tp / np.maximum(num_obj_union, np.finfo(np.float64).eps)

        return f1_score, iou_score