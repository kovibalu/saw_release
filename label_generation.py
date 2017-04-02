# Functions for pixel label generation from SAW annotations and NYUv2 depth,
# normal maps.

import os

import numpy as np
from PIL import Image
from scipy.ndimage import (binary_dilation, binary_erosion,
                           generate_binary_structure, zoom)

from utils import (compute_color_gradmag, compute_gradmag, ensuredir,
                   get_pixel_labels_dirname, load_all_photo_ids,
                   load_annotations, load_depth_normals, progress_bar,
                   render_full_complex_polygon_mask, vis_pixel_labels)


def generate_labels(saw_image_dir, saw_anno_dir, splits_dir, nyu_dataset_dir,
                    out_dir, filter_size, ignore_border=0.05,
                    depth_gradmag_thres=2.0, normal_gradmag_thres=1.5):
    """
    Generates pixel labels for each photo based on the NYUv2 dataset depths,
    normals and masks and the Shading Annotations in the Wild (SAW) dataset
    annotations.

    :param saw_image_dir: Directory which contains the SAW images (input to the
    CNN).

    :param saw_anno_dir: Directory which contains the SAW crowdsourced annotations.

    :param splits_dir: Directory which contains the list of photo IDs for each
    dataset split (training, validation, test).

    :param nyu_dataset_dir: Directory which contains the NYUv2 depth dataset.

    :param out_dir: Directory where we will save the generated pixel labels.

    :param filter_size: The size of the filter which is used for "label
    dilation" for the NS-SB and NS-ND labels. See our paper for details.

    :param ignore_border: We ignore ``ignore_border`` * max(image_width,
    image_height) pixels around the border and assign ignored labels there.

    :param depth_gradmag_thres: Depth gradient magnitude threshold (tau_depth),
    see our paper for details.

    :param normal_gradmag_thres: Normal gradient magnitude threshold (tau_normal),
    see our paper for details.
    """
    photo_ids = load_all_photo_ids(splits_dir)
    print 'Loading NYUv2 dataset...'
    depths, normals, masks = load_depth_normals(nyu_dataset_dir)

    pixlabel_dirname = get_pixel_labels_dirname(
        filter_size=filter_size, ignore_border=ignore_border,
        normal_gradmag_thres=normal_gradmag_thres,
        depth_gradmag_thres=depth_gradmag_thres,
    )
    pixlabel_dir = os.path.join(out_dir, pixlabel_dirname)
    ensuredir(pixlabel_dir)

    print 'Generating pixel labels...'
    for photo_id in progress_bar(photo_ids):
        pixel_labels = get_pixel_labels(
            saw_image_dir=saw_image_dir,
            saw_anno_dir=saw_anno_dir,
            photo_id=photo_id,
            filter_size=filter_size,
            ignore_border=ignore_border,
            masks=masks,
            depths=depths,
            depth_gradmag_thres=depth_gradmag_thres,
            normals=normals,
            normal_gradmag_thres=normal_gradmag_thres,
        )
        # Save pixel labels blended with the image for visualization
        vis_pixel_labels(
            saw_image_dir=saw_image_dir, pixlabel_dir=pixlabel_dir,
            photo_id=photo_id, pixel_labels=pixel_labels,
        )
        np.save(os.path.join(out_dir, '%s.npy' % photo_id), pixel_labels)


def get_region_masks(photo_dic, img_w, img_h):
    """
    Generates a mask for each constant shading region in the photo.
    """
    masks = []
    shapes = photo_dic['constant_shading_regions']
    square_struct = generate_binary_structure(2, 2)

    for shape in shapes:
        mask_img = render_full_complex_polygon_mask(
            vertices=shape['vertices'], triangles=shape['triangles'],
            width=img_w, height=img_h, inverted=False, mode='L',
        )
        shape_mask = np.asarray(mask_img).astype(float)
        assert shape_mask.ndim == 2

        # Erode the mask to avoid shading regions which are on the boundary
        shape_mask = binary_erosion(shape_mask, structure=square_struct, iterations=3)
        mask = shape_mask > 0.5

        # Don't use empty masks
        if np.sum(mask) == 0:
            continue

        masks.append(mask)

    return masks


def get_shadow_boundaries(photo_dic, img_w, img_h, filter_size):
    """
    Generates pixel labels for shadow boundary points based on the shadow
    boundary point annotations for the specified photo.
    """
    mask = np.zeros((img_h, img_w), dtype=bool)

    pt_data = photo_dic['shadow_boundary_points']
    h = filter_size // 2
    for pt in pt_data:
        cx, cy = int(pt['x'] * img_w), int(pt['y'] * img_h)
        mask[cy-h:cy+h+1, cx-h:cx+h+1] = True

    return mask


def get_pixel_labels(saw_image_dir, saw_anno_dir, photo_id, filter_size,
                     ignore_border, masks, depths, depth_gradmag_thres,
                     normals, normal_gradmag_thres):
    """
    Generates pixel labels for a photo based on the NYUv2 dataset depth, normal
    maps and the Shading Annotations in the Wild (SAW) annotations.
    """
    photo_dic = load_annotations(
        saw_anno_dir=saw_anno_dir, photo_id=photo_id)

    # Get image size
    img_path = os.path.join(saw_image_dir, '%s.png' % photo_id)
    img = Image.open(img_path)
    img_w, img_h = img.size

    # 0: non-smooth shading (NS-ND), 1: non-smooth shading (NS-SB), 2: smooth
    # shading (S), 100: no data
    # Fill with "no data" by default
    labels = np.full((img_h, img_w), 100, dtype=int)

    # These photos have incorrect depth in the NYUv2 dataset
    nyu_blacklist = [451]
    if photo_dic['in_nyu_dataset'] and photo_dic['nyu_idx'] not in nyu_blacklist:
        # True if we are on the border
        ignore_border_px = int(ignore_border * max(img_w, img_h))
        border_mask = np.zeros_like(labels, dtype=bool)
        border_mask[:ignore_border_px, :] = True
        border_mask[:, :ignore_border_px] = True
        border_mask[-ignore_border_px:, :] = True
        border_mask[:, -ignore_border_px:] = True

        assert photo_dic['nyu_idx'] is not None
        struct2 = generate_binary_structure(2, 2)
        valid_mask = masks[photo_dic['nyu_idx']]
        valid_mask = zoom(valid_mask, float(labels.shape[0])/valid_mask.shape[0], order=1)
        valid_mask = binary_erosion(valid_mask, structure=struct2, iterations=5)
        assert valid_mask.shape == labels.shape

        depth_map = np.transpose(depths[photo_dic['nyu_idx']], (1, 0))
        depth_map = zoom(depth_map, float(labels.shape[0])/depth_map.shape[0], order=1)
        assert depth_map.shape == labels.shape
        depth_gradmag = compute_gradmag(depth_map)
        depth_mask = np.logical_and(
            depth_gradmag >= depth_gradmag_thres, valid_mask
        )
        depth_mask = np.logical_and(depth_mask, ~border_mask)
        if filter_size:
            depth_mask = binary_dilation(
                depth_mask, structure=struct2, iterations=filter_size//2
            )
        # Pixels around depth discontinuities count as non-smooth shading
        labels[depth_mask] = 0

        normal_map = normals[photo_dic['nyu_idx']]
        zoom_scale = float(labels.shape[0])/normal_map.shape[0]
        normal_map = zoom(normal_map, (zoom_scale, zoom_scale, 1), order=1)
        assert normal_map.shape[:2] == labels.shape[:2]
        assert normal_map.shape[2] == 3
        # Normalize normals after resizing image
        normal_map /= np.linalg.norm(normal_map, axis=2)[:, :, np.newaxis]
        normal_gradmag = compute_color_gradmag(normal_map)
        normal_mask = np.logical_and(
            normal_gradmag >= normal_gradmag_thres, valid_mask
        )
        if filter_size:
            normal_mask = binary_dilation(
                normal_mask, structure=struct2, iterations=filter_size//2
            )
        normal_mask = np.logical_and(normal_mask, ~border_mask)
        # Pixels around normal discontinuities count as non-smooth shading
        labels[normal_mask] = 0

    # Our constant shading regions count as "smooth shading"
    region_masks = get_region_masks(
        photo_dic=photo_dic, img_w=img_w, img_h=img_h)
    for regm in region_masks:
        labels[regm] = 2

    # Add points on/around shadow boundaries
    osb_mask = get_shadow_boundaries(
        photo_dic=photo_dic, img_w=img_w, img_h=img_h, filter_size=filter_size)
    labels[osb_mask] = 1

    return labels
