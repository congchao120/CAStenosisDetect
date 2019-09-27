import os
import xml.etree.ElementTree as ET

import matplotlib.patches as patches
import matplotlib.pyplot as plt  # showing and rendering figures
import scipy.io as scio
from vis.utils import utils


def parse_annotation(ann_dir, img_dir, labels=[]):
    all_imgs = []
    seen_labels = {}
    
    for ann in sorted(os.listdir(ann_dir)):
        img = {'object':[]}

        tree = ET.parse(ann_dir + ann)
        
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text + '.jpg'
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1
                        
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]
                            
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]
                        
    return all_imgs, seen_labels


def parse_annotation_mat_box(mat_dir, img_dir, labels=[], thres_min_wh=10):
    all_imgs = []
    seen_labels = {}
    mat = scio.loadmat(mat_dir)
    roi_array = mat.get('ROI')
    imgname_array = mat.get('imgs')

    img_list = os.listdir(img_dir)
    print('Image number: %d'%(len(img_list)))
    _, roi_number, img_number = roi_array.shape

    if img_number != imgname_array.shape[0]:
        print('Warning: incorrect image and roi number!')

    for ii in range(img_number):
        img = {'object': []}
        img_name = imgname_array[ii]['name'][0][0]

        if not os.path.exists(os.path.join(img_dir, img_name)):
            print('Warning: missing image:' + img_name)
            continue
        img['filename'] = os.path.join(img_dir, img_name)
        _img = utils.load_img(img['filename'])
        img['width'] = _img.shape[0]
        img['height'] = _img.shape[1]
        scale_w = img['width'] / 512
        scale_h = img['height'] / 512

        for jj in range(roi_number):
            roi_attribute = roi_array[:,jj,ii]
            obj = {}
            obj['name'] = 'stenosis'
            obj['xmin'] = min(roi_attribute[::2])
            obj['xmax'] = max(roi_attribute[::2])
            obj['ymin'] = min(roi_attribute[1::2])
            obj['ymax'] = max(roi_attribute[1::2])

            if obj['xmax'] == 0 and obj['ymax'] == 0:
                break

            if obj['name'] in seen_labels:
                seen_labels[obj['name']] += 1
            else:
                seen_labels[obj['name']] = 1

            #modify width/height to be bigger than threshold
            if obj['xmax'] - obj['xmin'] < thres_min_wh * scale_w:
                obj['xmax'] = obj['xmax'] + ((thres_min_wh * scale_w - (obj['xmax'] - obj['xmin']))/2.0)
                obj['xmin'] = obj['xmin'] - ((thres_min_wh * scale_w - (obj['xmax'] - obj['xmin']))/2.0)

            if obj['ymax'] - obj['ymin'] < thres_min_wh * scale_h:
                obj['ymax'] = obj['ymax'] + ((thres_min_wh * scale_h - (obj['ymax'] - obj['ymin']))/2.0)
                obj['ymin'] = obj['ymin'] - ((thres_min_wh * scale_h - (obj['ymax'] - obj['ymin']))/2.0)

            if len(labels) > 0 and obj['name'] not in labels:
                break
            else:
                img['object'] += [obj]

        #if len(img['object']) > 0:
        all_imgs += [img]

    return all_imgs, seen_labels



def parse_annotation_mat_point(mat_dir, img_dir, labels=[], thres_min_wh=10):
    all_imgs = []
    seen_labels = {}
    mat = scio.loadmat(mat_dir)
    pos_array = mat.get('poss')
    imgname_array = mat.get('imgs')

    img_list = os.listdir(img_dir)
    print('Image number: %d'%(len(img_list)))
    _, img_number = pos_array.shape

    if img_number != imgname_array.shape[0]:
        print('Warning: incorrect image and roi number!')

    for ii in range(img_number):
        img = {'object': []}
        img_name = imgname_array[ii]['name'][0][0]

        if not os.path.exists(os.path.join(img_dir, img_name)):
            print('Warning: missing image:' + img_name)
            continue
        img['filename'] = os.path.join(img_dir, img_name)
        _img = utils.load_img(img['filename'])
        img['width'] = _img.shape[0]
        img['height'] = _img.shape[1]
        scale_w = img['width'] / 512
        scale_h = img['height'] / 512

        pos = pos_array[:, ii]
        obj = {}
        obj['name'] = 'stenosis'
        obj['xmin'] = pos[0]
        obj['xmax'] = pos[0]+1
        obj['ymin'] = pos[1]
        obj['ymax'] = pos[1]+1

        if obj['xmax'] == 0 and obj['ymax'] == 0:
            break

        if obj['name'] in seen_labels:
            seen_labels[obj['name']] += 1
        else:
            seen_labels[obj['name']] = 1

        # modify width/height to be bigger than threshold
        if obj['xmax'] - obj['xmin'] < thres_min_wh * scale_w:
            obj['xmax'] = obj['xmax'] + ((thres_min_wh * scale_w - (obj['xmax'] - obj['xmin'])) / 2.0)
            obj['xmin'] = obj['xmin'] - ((thres_min_wh * scale_w - (obj['xmax'] - obj['xmin'])) / 2.0)

        if obj['ymax'] - obj['ymin'] < thres_min_wh * scale_h:
            obj['ymax'] = obj['ymax'] + ((thres_min_wh * scale_h - (obj['ymax'] - obj['ymin'])) / 2.0)
            obj['ymin'] = obj['ymin'] - ((thres_min_wh * scale_h - (obj['ymax'] - obj['ymin'])) / 2.0)

        if len(labels) > 0 and obj['name'] not in labels:
            break
        else:
            img['object'] += [obj]


        #if len(img['object']) > 0:
        all_imgs += [img]

    return all_imgs, seen_labels

if __name__ == '__main__':
    path = 'C:\\OneDrive\\\Core320_SingleStenose\\lca_laocra\\'
    to_path = 'C:\\temp\\lca_laocra\\'
    #path = '/home/ccong3/data/Core320_candidate_20190701/R/'
    #to_path = '/home/ccong3/TEMP/Core320_candidate_20190701/R/'
    if not os.path.exists(to_path):
        os.mkdir(to_path)
    min_size = 10
    img_path = os.path.join(path, 'image')
    obj_path = os.path.join(path, 'label', 'pos.mat')
    train_imgs, train_labels = parse_annotation_mat_point(obj_path, img_path, thres_min_wh=70)
    n_imgs = len(train_imgs)
    for ii in range(n_imgs):
        sample = train_imgs[ii]
        img_path = sample.get('filename')
        img_name = os.path.basename(img_path)
        img_w = sample.get('width')
        img_h = sample.get('height')
        n_roi = len(sample.get('object'))
        img = utils.load_img(img_path, target_size=(img_w, img_h))
        fig, ax1 = plt.subplots(1,1, figsize = (6, 6), dpi = 150)
        ax1.imshow(img)
        for jj in range(n_roi):
            obj_name = sample.get('object')[jj].get('name')
            obj_xy = (sample.get('object')[jj].get('xmin'), sample.get('object')[jj].get('ymin'))
            obj_w = sample.get('object')[jj].get('xmax') - sample.get('object')[jj].get('xmin')
            obj_h = sample.get('object')[jj].get('ymax') - sample.get('object')[jj].get('ymin')

            # Create a Rectangle patch
            rect = patches.Rectangle(obj_xy, obj_w, obj_h, linewidth=1, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax1.add_patch(rect)

        fig.savefig(os.path.join(to_path, img_name + '.png'))
