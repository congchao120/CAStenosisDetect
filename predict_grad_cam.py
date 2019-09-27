"""
=================
Animated subplots
=================

This example uses subclassing, but there is no reason that the proper function
couldn't be set up and then use FuncAnimation. The code is long, but not
really complex. The length is due solely to the fact that there are a total of
9 lines that need to be changed for the animation as well as 3 subplots that
need initial set up.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import pandas as pd
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import numpy as np
from keras import activations
from Inception_Models import inceptionV3_coronary_model
from vis.optimizer import Optimizer
from vis.losses import ActivationMaximization
import os, sys
from vis.utils import utils
import cv2
import matplotlib.cm as cm
from scipy.misc import imresize
from keras import backend as K

import matplotlib.patches as patches
import matplotlib.pyplot as plt  # showing and rendering figures
import scipy.io as scio


def calc_cam_with_losses(input_tensor, losses,
                         seed_input, penultimate_layer,
                         grad_modifier=None):
    """Generates a gradient based class activation map (CAM) by using positive gradients of `input_tensor`
    with respect to weighted `losses`.

    For details on grad-CAM, see the paper:
    [Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization]
    (https://arxiv.org/pdf/1610.02391v1.pdf).

    Unlike [class activation mapping](https://arxiv.org/pdf/1512.04150v1.pdf), which requires minor changes to
    network architecture in some instances, grad-CAM has a more general applicability.

    Compared to saliency maps, grad-CAM is class discriminative; i.e., the 'cat' explanation exclusively highlights
    cat regions and not the 'dog' region and vice-versa.

    Args:
        input_tensor: An input tensor of shape: `(samples, channels, image_dims...)` if `image_data_format=
            channels_first` or `(samples, image_dims..., channels)` if `image_data_format=channels_last`.
        losses: List of ([Loss](vis.losses#Loss), weight) tuples.
        seed_input: The model input for which activation map needs to be visualized.
        penultimate_layer: The pre-layer to `layer_idx` whose feature maps should be used to compute gradients
            with respect to filter output.
        grad_modifier: gradient modifier to use. See [grad_modifiers](vis.grad_modifiers.md). If you don't
            specify anything, gradients are unchanged (Default value = None)

    Returns:
        The heatmap image indicating the `seed_input` regions whose change would most contribute towards minimizing the
        weighted `losses`.
    """
    penultimate_output = penultimate_layer.output
    opt = Optimizer(input_tensor, losses, wrt_tensor=penultimate_output, norm_grads=False)
    _, grads, penultimate_output_value = opt.minimize(seed_input, max_iter=1, grad_modifier=grad_modifier, verbose=False)

    # For numerical stability. Very small grad values along with small penultimate_output_value can cause
    # w * penultimate_output_value to zero out, even for reasonable fp precision of float32.
    grads = grads / (np.max(grads) + K.epsilon())

    # Average pooling across all feature maps.
    # This captures the importance of feature map (channel) idx to the output.
    channel_idx = 1 if K.image_data_format() == 'channels_first' else -1
    other_axis = np.delete(np.arange(len(grads.shape)), channel_idx)
    weights = np.mean(grads, axis=tuple(other_axis))

    # Generate heatmap by computing weight * output over feature maps
    output_dims = utils.get_img_shape(penultimate_output)[2:]
    heatmap = np.zeros(shape=output_dims, dtype=K.floatx())
    for i, w in enumerate(weights):
        if channel_idx == -1:
            heatmap += w * penultimate_output_value[0, ..., i]
        else:
            heatmap += w * penultimate_output_value[0, i, ...]

    # ReLU thresholding to exclude pattern mismatch information (negative gradients).
    heatmap = np.maximum(heatmap, 0)

    return heatmap


def visualize_cam(model, layer_idx, filter_indices,
                  seed_input, penultimate_layer_idx=None,
                  backprop_modifier=None, grad_modifier=None):
    """Generates a gradient based class activation map (grad-CAM) that maximizes the outputs of
    `filter_indices` in `layer_idx`.

    Args:
        model: The `keras.models.Model` instance. The model input shape must be: `(samples, channels, image_dims...)`
            if `image_data_format=channels_first` or `(samples, image_dims..., channels)` if
            `image_data_format=channels_last`.
        layer_idx: The layer index within `model.layers` whose filters needs to be visualized.
        filter_indices: filter indices within the layer to be maximized.
            If None, all filters are visualized. (Default value = None)
            For `keras.layers.Dense` layer, `filter_idx` is interpreted as the output index.
            If you are visualizing final `keras.layers.Dense` layer, consider switching 'softmax' activation for
            'linear' using [utils.apply_modifications](vis.utils.utils#apply_modifications) for better results.
        seed_input: The input image for which activation map needs to be visualized.
        penultimate_layer_idx: The pre-layer to `layer_idx` whose feature maps should be used to compute gradients
            wrt filter output. If not provided, it is set to the nearest penultimate `Conv` or `Pooling` layer.
        backprop_modifier: backprop modifier to use. See [backprop_modifiers](vis.backprop_modifiers.md). If you don't
            specify anything, no backprop modification is applied. (Default value = None)
        grad_modifier: gradient modifier to use. See [grad_modifiers](vis.grad_modifiers.md). If you don't
            specify anything, gradients are unchanged (Default value = None)

     Example:
        If you wanted to visualize attention over 'bird' category, say output index 22 on the
        final `keras.layers.Dense` layer, then, `filter_indices = [22]`, `layer = dense_layer`.

        One could also set filter indices to more than one value. For example, `filter_indices = [22, 23]` should
        (hopefully) show attention map that corresponds to both 22, 23 output categories.

    Returns:
        The heatmap image indicating the input regions whose change would most contribute towards
        maximizing the output of `filter_indices`.
    """

    penultimate_layer = model.layers[penultimate_layer_idx]

    # `ActivationMaximization` outputs negative gradient values for increase in activations. Multiply with -1
    # so that positive gradients indicate increase instead.
    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), -1)
    ]

    heatmap = calc_cam_with_losses(model.input, losses, seed_input, penultimate_layer, grad_modifier)

    input_dims = utils.get_img_shape(model.input)[2:]
    heatmap = imresize(heatmap, input_dims, interp='nearest', mode='F')
    heatmap = utils.normalize(heatmap)
    return np.uint8(cm.jet(heatmap)[..., :3] * 255)



def pred_cam_bbox(model, layer_idx, filter_indices,
                  seed_input, penultimate_layer_idx=None,
                  score_threshold=0.5, grad_modifier=None):
    """Generates a gradient based class activation map (grad-CAM) that maximizes the outputs of
    `filter_indices` in `layer_idx`.

    Args:
        model: The `keras.models.Model` instance. The model input shape must be: `(samples, channels, image_dims...)`
            if `image_data_format=channels_first` or `(samples, image_dims..., channels)` if
            `image_data_format=channels_last`.
        layer_idx: The layer index within `model.layers` whose filters needs to be visualized.
        filter_indices: filter indices within the layer to be maximized.
            If None, all filters are visualized. (Default value = None)
            For `keras.layers.Dense` layer, `filter_idx` is interpreted as the output index.
            If you are visualizing final `keras.layers.Dense` layer, consider switching 'softmax' activation for
            'linear' using [utils.apply_modifications](vis.utils.utils#apply_modifications) for better results.
        seed_input: The input image for which activation map needs to be visualized.
        penultimate_layer_idx: The pre-layer to `layer_idx` whose feature maps should be used to compute gradients
            wrt filter output. If not provided, it is set to the nearest penultimate `Conv` or `Pooling` layer.
        backprop_modifier: backprop modifier to use. See [backprop_modifiers](vis.backprop_modifiers.md). If you don't
            specify anything, no backprop modification is applied. (Default value = None)
        grad_modifier: gradient modifier to use. See [grad_modifiers](vis.grad_modifiers.md). If you don't
            specify anything, gradients are unchanged (Default value = None)

     Example:
        If you wanted to visualize attention over 'bird' category, say output index 22 on the
        final `keras.layers.Dense` layer, then, `filter_indices = [22]`, `layer = dense_layer`.

        One could also set filter indices to more than one value. For example, `filter_indices = [22, 23]` should
        (hopefully) show attention map that corresponds to both 22, 23 output categories.

    Returns:
        The heatmap image indicating the input regions whose change would most contribute towards
        maximizing the output of `filter_indices`.
    """

    penultimate_layer = model.layers[penultimate_layer_idx]

    # `ActivationMaximization` outputs negative gradient values for increase in activations. Multiply with -1
    # so that positive gradients indicate increase instead.
    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), -1)
    ]

    heatmap = calc_cam_with_losses(model.input, losses, seed_input, penultimate_layer, grad_modifier)
    input_dims = utils.get_img_shape(model.input)[2:]
    heatmap = imresize(heatmap, input_dims, interp='bicubic', mode='F')
    heatmap = utils.normalize(heatmap) * 255
    img = np.stack((heatmap,) * 3, -1)
    img = img.astype(np.uint8)
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    activation_threshold = 255*score_threshold
    ret, binary = cv2.threshold(grayed, activation_threshold, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    '''
    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    hm_size = heatmap.shape

    all_boxes = np.zeros((0, 5))
    hm_w = input_dims[0]/hm_size[0]
    hm_h = input_dims[1]/hm_size[1]
    hm_box_offset = 1
    for i in range(hm_size[0]):
        for j in range(hm_size[1]):
            score = heatmap[i,j]
            if score > 0.8:
                box_x1 = j*hm_w + hm_box_offset
                box_y1 = i*hm_h + hm_box_offset
                box_x2 = (j+1)*hm_w - hm_box_offset
                box_y2 = (i+1)*hm_h - hm_box_offset
                box_score = score
                all_boxes = np.append(all_boxes, [box_x1, box_y1, box_x2, box_y2, box_score])
    if len(all_boxes) > 5:
        all_boxes = np.resize(all_boxes, (-1, 5))
    else:
        all_boxes = []
    '''
    all_boxes = np.zeros((0, ))
    for contour in contours:
        box_x1 = min(contour[:, 0, 0])
        box_y1 = min(contour[:, 0, 1])
        box_x2 = max(contour[:, 0, 0])
        box_y2 = max(contour[:, 0, 1])
        box_score = 1.0
        all_boxes = np.append(all_boxes, [box_x1, box_y1, box_x2, box_y2, box_score])
    if len(all_boxes) >= 5:
        all_boxes = np.reshape(all_boxes, (-1, 5))
    else:
        all_boxes = []
    return all_boxes


if __name__ == '__main__':
    JPG_PATH = 'D:\\OneDrive\\Core320_candidate_20190701\\L\\LAO_CRA\\image\\'
    OUT_PATH = 'D:\\OneDrive\\Core320_candidate_20190701\\L\\LAO_CRA\\heatmap\\'
    weight_best_path = 'E:\\hdf5\\full_coronary_model.TOTAL_LAO_CRA.0_1234_d.allin.4321.hdf5'
    num_classes = 3
    pred_output = np.zeros((1, 1, num_classes))
    img_input = np.zeros((512, 512, 3))
    img_input = np.expand_dims(img_input, axis=0)
    model = inceptionV3_coronary_model(img_input, pred_output)
    model.load_weights(weight_best_path)
    # Utility to search for layer index by name
    layer_idx = len(model.layers)-1
    # swap with softmax with linear classifier for the reasons mentioned above
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)
    img_path = JPG_PATH
    img_list = os.listdir(img_path)
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    for img in img_list:
        img_name = os.path.basename(img)
        img_file = os.path.join(img_path, img_name)
        img = utils.load_img(img_file, target_size=(512, 512))
        preprocess_img = preprocess_input(img.astype('float64'))
        pred = model.predict(np.expand_dims(preprocess_img, axis=0))
        idx_pred = int(np.argmax(pred[0]))
        if pred[0][1] >= 1.0:
            '''
            heatmap = visualize_cam(model, layer_idx, filter_indices=1,  # 1 for stenosis
                                    seed_input=preprocess_img, backprop_modifier=None,  # relu and guided don't work
                                    penultimate_layer_idx=310  # 310 is concatenation before global average pooling
                                    )
            out_img_path = os.path.join(OUT_PATH, img_name + 'r_img_' + '.jpg')
            cv2.imwrite(out_img_path, img)
            out_img_path = os.path.join(OUT_PATH, img_name + 'r_cam_' + '.jpg')
            cv2.imwrite(out_img_path, heatmap)
'''
            all_boxes = pred_cam_bbox(model, layer_idx, filter_indices=1,  # 1 for stenosis
                                    seed_input=preprocess_img, backprop_modifier=None,  # relu and guided don't work
                                    penultimate_layer_idx=310  # 310 is concatenation before global average pooling
                                    )
            fig, ax1 = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
            ax1.imshow(img)
            for box in all_boxes:
                obj_xy = (box[0], box[1])
                obj_w = box[2] - box[0]
                obj_h = box[3] - box[1]
                score = box[4]

                # Create a Rectangle patch
                rect = patches.Rectangle(obj_xy, obj_w, obj_h, linewidth=1, edgecolor='r', facecolor='none')
                # Add the patch to the Axes
                ax1.add_patch(rect)

            out_img_path = os.path.join(OUT_PATH, img_name + 'bbox' + '.jpg')
            fig.savefig(out_img_path)
        print(pred[0])



