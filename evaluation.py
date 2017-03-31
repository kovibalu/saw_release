# Functions for precision-recall curve generation

import csv
import os

import numpy as np
from scipy.ndimage.filters import maximum_filter

from metrics import get_pr_from_conf_mx, grouped_confusion_matrix
from utils import (compute_gradmag, ensuredir, gen_class_weights_str,
                   load_photo_ids_for_split, load_pixel_labels,
                   load_shading_image_arr, plot_and_save_2D_arrays,
                   progress_bar, srgb_to_rgb, to_perc)
from net_tools import eval_net_on_photo, load_net


def generate_pr(saw_image_dir, pixel_labels_dir, splits_dir, out_dir,
                dataset_split, class_weights, bl_filter_size, algo_configs,
                thres_count=200):
    """
    Generate precision-recall curves for each specified algorithm.

    :param saw_image_dir: Directory which contains the SAW images (input to the
    CNN).

    :param pixel_labels_dir: Directory which contains the SAW pixel labels for each photo.

    :param splits_dir: Directory which contains the list of photo IDs for each
    dataset split (training, validation, test).

    :param out_dir: Directory where we will save the generated PR curves.

    :param dataset_split: Dataset split we want to evaluate on. Can be "R"
    (training), "V" (validation) or "E" (test).

    :param class_weights: List of weights for the 3 classes (NS-ND, NS-SB, S).
    We used [1, 1, 2] in the paper.

    :param bl_filter_size: The size of the maximum filter used on the shading
    gradient magnitude image. We used 10 in the paper. If 0, we do not filter.

    :param algo_configs: List of baselines as (algorithm slug, predicted
    (decomposed) shading directory) pairs or ("saw_pixelnet",
    "path_to_trained_net_dir") for our trained CNN.

    :param thres_count: Number of thresholds we want to evaluate on. Check
    ``gen_pr_thres_list`` to see how we sample thresholds between 0 and 1.
    """
    bl_names_dic = {
        'baseline_reflectance': 'Constant R',
        'zhou2015_reflprior': '[Zhou et al. 2015]',
        'bell2014_densecrf': '[Bell et al. 2014]',
        'grosse2009_color_retinex': 'Color Retinex',
        'grosse2009_grayscale_retinex': 'Grayscale Retinex',
        'zhao2012_nonlocal': '[Zhao et al. 2012]',
        'garces2012_clustering': '[Garces et al. 2012]',
        'shen2011_optimization': '[Shen et al. 2011]',
        'saw_pixelnet': 'SAW Pixelnet',
    }

    rootdir = os.path.join(out_dir, gen_class_weights_str(class_weights))
    ensuredir(rootdir)
    thres_list = gen_pr_thres_list(thres_count)
    photo_ids = load_photo_ids_for_split(
        splits_dir=splits_dir, dataset_split=dataset_split
    )

    plot_arrs = []
    line_names = []

    fn = 'pr-%s' % {'R': 'train', 'V': 'val', 'E': 'test'}[dataset_split]
    title = '%s Precision-Recall' % (
        {'R': 'Training', 'V': 'Validation', 'E': 'Test'}[dataset_split],
    )

    def snap_plot():
        plot_and_save_2D_arrays(
            filename=os.path.join(rootdir, fn) + '.pdf', arrs=plot_arrs, title=title,
            xlabel='Recall', xinterval=(0, 1), ylabel='Precision',
            yinterval=(0, 1), line_names=line_names,
        )
        save_plot_arr_to_csv(
            file_path=os.path.join(rootdir, fn) + '.csv',
            thres_list=thres_list,
            arrs=plot_arrs,
            line_names=line_names,
        )

    for algo_slug, algo_dir in algo_configs:
        print 'Working on %s (path: %s)...' % (algo_slug, algo_dir)

        if algo_slug == 'saw_pixelnet':
            eval_kwargs = dict(
                saw_image_dir=saw_image_dir,
                net=load_net(net_dir=algo_dir),
            )
            rdic_list = get_precision_recall_list(
                pixel_labels_dir=pixel_labels_dir, thres_list=thres_list,
                photo_ids=photo_ids, class_weights=class_weights,
                eval_func=eval_net_on_photo, eval_kwargs=eval_kwargs,
            )
        else:
            eval_kwargs = dict(
                pred_shading_dir=algo_dir,
                bl_filter_size=bl_filter_size,
            )
            rdic_list = get_precision_recall_list(
                pixel_labels_dir=pixel_labels_dir, thres_list=thres_list,
                photo_ids=photo_ids, class_weights=class_weights,
                eval_func=eval_baseline_on_photo, eval_kwargs=eval_kwargs,
            )

        plot_arrs.append(gen_plot_arr(rdic_list))
        if algo_slug in bl_names_dic:
            line_names.append(bl_names_dic[algo_slug])
        else:
            line_names.append('%s, bfs (%s)' % (
                algo_slug, bl_filter_size
            ))
        snap_plot()


def gen_plot_arr(rdic_list):
    """ Generate a list of x, y pairs based on recall and precision for each
    threshold. """
    # x: recall, y: precision
    plot_arr = np.empty((len(rdic_list), 2))
    for i, rdic in enumerate(rdic_list):
        plot_arr[i, 0] = rdic['overall_recall']
        plot_arr[i, 1] = rdic['overall_prec']

    return plot_arr


def save_plot_arr_to_csv(file_path, thres_list, arrs, line_names):
    """ Save plot arrays (each is a list of x,y pairs) as a CSV file. """
    assert len(arrs) == len(line_names)
    assert len(thres_list) == arrs[0].shape[0]

    csv_data = []
    header = ['Threshold', 'Recall', 'Precision']

    for ln, arr in zip(line_names, arrs):
        csv_data.append([])
        csv_data.append([ln])
        csv_data.append(header)

        for thres, (recall, prec) in zip(thres_list, arr):
            csv_data.append([
                '%.3f' % thres,
                to_perc(recall),
                to_perc(prec),
            ])

    with open(file_path, 'w') as fp:
        wr = csv.writer(fp, delimiter=',')
        wr.writerows(csv_data)


def gen_pr_thres_list(thres_count):
    """ Generate a list of thresholds between 0 and 1, generating more around 0
    and 1 than in the middle. """
    zero_to_one = np.logspace(-4, 0, num=thres_count//2)
    h0 = zero_to_one / 2
    h1 = 1 - h0
    thres_list = sorted(list(h0) + list(h1))

    return thres_list


def get_precision_recall_list(pixel_labels_dir, thres_list, photo_ids,
                              class_weights, eval_func, eval_kwargs):
    """
    This method generates a list of precision-recall pairs and confusion
    matrices for each threshold provided in ``thres_list`` over all photos.

    :param pixel_labels_dir: Directory which contains the SAW pixel labels for each photo.

    :param thres_list: List of shading gradient magnitude thresholds we use to
    generate points on the precision-recall curve.

    :param photo_ids: IDs of the photos we want to evaluate on.

    :param class_weights: List of weights for the 3 classes (NS-ND, NS-SB, S).
    We used [1, 1, 2] in the paper.

    :param eval_func: Evaluation function which returns a confusion matrix for
    a given photo.

    :param eval_kwargs: Extra parameters for the evaluation fuction.

    """

    output_count = len(thres_list)
    overall_conf_mx_list = [
        np.zeros((3, 2), dtype=int)
        for _ in xrange(output_count)
    ]

    for photo_id in progress_bar(photo_ids):
        conf_mx_list = eval_func(
            pixel_labels_dir=pixel_labels_dir, thres_list=thres_list,
            photo_id=photo_id, **eval_kwargs
        )

        for i, conf_mx in enumerate(conf_mx_list):
            # If this image didn't have any labels
            if conf_mx is None:
                continue

            overall_conf_mx_list[i] += conf_mx

    ret = []
    for i in xrange(output_count):
        overall_prec, overall_recall = get_pr_from_conf_mx(
            conf_mx=overall_conf_mx_list[i], class_weights=class_weights,
        )

        ret.append(dict(
            overall_prec=overall_prec,
            overall_recall=overall_recall,
            overall_conf_mx=overall_conf_mx_list[i],
        ))

    return ret


def eval_baseline_on_photo(pixel_labels_dir, thres_list, photo_id,
                           pred_shading_dir, bl_filter_size):
    """
    This method generates a list of precision-recall pairs and confusion
    matrices for each threshold provided in ``thres_list`` for a specific
    photo.

    :param pixel_labels_dir: Directory which contains the SAW pixel labels for each photo.

    :param thres_list: List of shading gradient magnitude thresholds we use to
    generate points on the precision-recall curve.

    :param photo_id: ID of the photo we want to evaluate on.

    :param pred_shading_dir: Directory which contains the intrinsic image
    decompositions for all photos generated by a decomposition algorithm.

    :param bl_filter_size: The size of the maximum filter used on the shading
    gradient magnitude image. We used 10 in the paper. If 0, we do not filter.
    """
    shading_image_arr = load_shading_image_arr(
        pred_shading_dir=pred_shading_dir, photo_id=photo_id
    )
    shading_image_linear = srgb_to_rgb(shading_image_arr)
    shading_image_linear_grayscale = np.mean(shading_image_linear, axis=2)
    shading_gradmag = compute_gradmag(shading_image_linear_grayscale)

    if bl_filter_size:
        shading_gradmag = maximum_filter(shading_gradmag, size=bl_filter_size)

    # We have the following ground truth labels:
    # (0) normal/depth discontinuity non-smooth shading (NS-ND)
    # (1) shadow boundary non-smooth shading (NS-SB)
    # (2) smooth shading (S)
    # (100) no data, ignored
    y_true = load_pixel_labels(pixel_labels_dir=pixel_labels_dir, photo_id=photo_id)
    y_true = np.ravel(y_true)
    ignored_mask = y_true == 100

    # If we don't have labels for this photo (so everything is ignored), return
    # None
    if np.all(ignored_mask):
        return [None] * len(thres_list)

    ret = []
    for thres in thres_list:
        y_pred = (shading_gradmag < thres).astype(int)
        y_pred = np.ravel(y_pred)
        # Note: y_pred should have the same image resolution as y_true
        assert y_pred.shape == y_true.shape
        ret.append(grouped_confusion_matrix(y_true[~ignored_mask], y_pred[~ignored_mask]))

    return ret

