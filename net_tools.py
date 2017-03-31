import os
import sys

import numpy as np
from PIL import Image
from skimage.transform import resize

import constants
from metrics import grouped_confusion_matrix
from utils import load_pixel_labels


def add_to_path(p):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_net(net_dir):
    """ Load the CNN from disk. """
    add_to_path(constants.CAFFE_PYTHON_ROOT)
    import caffe
    caffe.set_mode_gpu()
    caffe.set_device(0)
    deployfile_path = os.path.join(net_dir, 'deploy.prototxt')
    weights_path = os.path.join(net_dir, 'snapshot.caffemodel')

    return caffe.Net(deployfile_path, weights_path, caffe.TEST)


def eval_net_on_photo(pixel_labels_dir, thres_list, photo_id, saw_image_dir,
                      net):
    """
    This method generates a list of precision-recall pairs and confusion
    matrices for each threshold provided in ``thres_list`` for a specific
    photo.

    :param pixel_labels_dir: Directory which contains the SAW pixel labels for each photo.

    :param thres_list: List of shading gradient magnitude thresholds we use to
    generate points on the precision-recall curve.

    :param photo_id: ID of the photo we want to evaluate on.

    :param saw_image_dir: Directory which contains the SAW images (input to the
    CNN).

    :param net: Loaded CNN we use for smooth shading predictions.
    """
    scores = run_net_on_photo(
        saw_image_dir=saw_image_dir, net=net, photo_id=photo_id)

    # We have the following ground truth labels:
    # (0) normal/depth discontinuity non-smooth shading (NS-ND)
    # (1) shadow boundary non-smooth shading (NS-SB)
    # (2) smooth shading (S)
    # (100) no data, ignored
    y_true = load_pixel_labels(pixel_labels_dir=pixel_labels_dir, photo_id=photo_id)

    h, w = y_true.shape
    y_true = np.ravel(y_true)
    ignored_mask = y_true == 100
    # If we don't have labels for this photo (so everything is ignored), return
    # None
    if np.all(ignored_mask):
        return [None] * len(thres_list)

    # Use linear interpolation for probabilities
    scores = resize(scores, (h, w), clip=False, preserve_range=True)
    # Renormalize scores
    scores /= np.sum(scores, axis=2)[:, :, np.newaxis]
    smooth_probs = scores[:, :, 2]

    ret = []
    for thres in thres_list:
        # Predict 1: smooth, 0: non-smooth
        y_pred = (smooth_probs > thres).astype(int)
        y_pred = np.ravel(y_pred)
        ret.append(grouped_confusion_matrix(y_true[~ignored_mask], y_pred[~ignored_mask]))

    return ret


def run_net_on_photo(photo_id, saw_image_dir, net):
    """
    This runs the CNN on a specified photo.

    :param photo_id: ID of the photo we want to evaluate on.

    :param saw_image_dir: Directory which contains the SAW images (input to the
    CNN).

    :param net: Loaded CNN we use for smooth shading predictions.
    """
    img_path = os.path.join(saw_image_dir, '%s.png' % photo_id)
    pil_img = Image.open(img_path)
    # Note: Our model expects images with values in range [0, 255]
    im = np.array(pil_img).astype(float)

    data_layers = ['data0_img', 'data1_labels', 'data2_valid_mask']

    imsize = 224
    impadding = 100
    padimsize = imsize + 2*impadding
    mean_value = np.array([103.939, 116.77, 123.68], dtype=np.float32)

    # Image
    net.blobs[data_layers[0]].reshape(1, 3, padimsize, padimsize)
    # Valid mask
    net.blobs[data_layers[2]].reshape(1, 1, padimsize, padimsize)
    # Labels (dummy)
    net.blobs[data_layers[1]].reshape(1, 1, padimsize, padimsize)

    # Change color channels from RGB to BGR
    im = im[:, :, [2, 1, 0]]
    # Resize
    im = resize(im, (imsize, imsize))
    # Subtract mean
    im = im - mean_value[np.newaxis, np.newaxis, :]
    # Transpose dimensions from HxWxC to CxHxW
    im = im.transpose([2, 0, 1])
    # Pad image
    im = np.lib.pad(
        im,
        (
            (0, 0),
            (impadding, impadding),
            (impadding, impadding),
        ), 'constant', constant_values=0
    )

    valid_mask = np.ones((imsize, imsize))
    valid_mask = np.lib.pad(
        valid_mask,
        (
            (impadding, impadding),
            (impadding, impadding),
        ), 'constant', constant_values=0
    )

    net.blobs[data_layers[0]].data[0, :, :, :] = im
    net.blobs[data_layers[2]].data[...] = valid_mask

    net.forward()

    scores = np.squeeze(net.blobs['prob_antishadow'].data[...])
    return np.reshape(scores, (imsize, imsize, scores.shape[1]))
