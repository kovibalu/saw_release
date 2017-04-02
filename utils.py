# Miscellaneous helper functions

import itertools
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage.filters import sobel


###############
# Plotting utils
###############


def plot_and_save_2D_arrays(filename, arrs, title='', xlabel='', xinterval=None, ylabel='', yinterval=None, line_names=[], simplified=False):
    """ Plots multiple arrays in the same plot based on the specifications and
    saves it. """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plot_2D_arrays(arrs, title, xlabel, xinterval, ylabel, yinterval, line_names, simplified)
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()


def plot_2D_arrays(arrs, title='', xlabel='', xinterval=None, ylabel='', yinterval=None, line_names=[], simplified=False):
    """ Plots multiple arrays in the same plot based on the specifications. """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.clf()
    sns.set_style('darkgrid')
    sns.set(font_scale=1.5)
    sns.set_palette('husl', 8)

    for arr in arrs:
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                'The array should be 2D and the second dimension should be 2!'
                ' Shape: %s' % str(arr.shape)
            )

        plt.plot(arr[:, 0], arr[:, 1])

    # If simplified, we don't show text anywhere
    if not simplified:
        plt.title(title[:30])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if line_names:
            plt.legend(line_names, loc=6, bbox_to_anchor=(1, 0.5))

    if xinterval:
        plt.xlim(xinterval)
    if yinterval:
        plt.ylim(yinterval)

    plt.tight_layout()


###############
# String handling
###############


def gen_class_weights_str(class_weights):
    assert len(class_weights) == 3 or class_weights is None
    if class_weights is None:
        suffix = 'None'
    else:
        suffix = '-'.join([
            '%.2f' % cw
            for cw in class_weights
        ])

    return 'class_weights-%s' % suffix


def get_pixel_labels_dirname(filter_size=0, ignore_border=0.05,
                             normal_gradmag_thres=1.5,
                             depth_gradmag_thres=2.0):
    return 'saw_data-filter_size_%s-ignore_border_%.2f-normal_gradmag_thres_%.1f-depth_gradmag_thres_%.1f' % (
        filter_size, ignore_border, normal_gradmag_thres, depth_gradmag_thres
    )


def to_perc(x):
    return '%.2f%%' % (x * 100)


###############
# I/O
###############


def ensuredir(dirpath):
    if os.path.exists(dirpath):
        return

    try:
        os.makedirs(dirpath)
    except OSError as exc:  # Python >2.5
        import errno
        if exc.errno == errno.EEXIST and os.path.isdir(dirpath):
            pass
        else:
            raise


def load_shading_image_arr(pred_shading_dir, photo_id):
    """ Loads up the decomposed shading layer for a photo as a numpy array with
    values in [0, 1]. """

    shading_img_path = os.path.join(pred_shading_dir, '%s-s.png' % photo_id)
    if not os.path.exists(shading_img_path):
        raise ValueError('Could not find decomposed shading image at "%s"' % shading_img_path)

    shading_image = Image.open(shading_img_path)
    shading_image_arr = np.asarray(shading_image).astype(float) / 255.0
    return shading_image_arr


def load_pixel_labels(pixel_labels_dir, photo_id):
    """ Loads up the ground truth pixel labels for a photo as a numpy array. """

    pixel_labels_path = os.path.join(pixel_labels_dir, '%s.npy' % photo_id)
    if not os.path.exists(pixel_labels_path):
        raise ValueError('Could not find ground truth labels at "%s"' % pixel_labels_path)

    return np.load(pixel_labels_path)


def load_annotations(saw_anno_dir, photo_id):
    """ Loads up the ground truth SAW annotations for a photo as a dictionary. """

    json_path = os.path.join(saw_anno_dir, '%s.json' % photo_id)
    if not os.path.exists(json_path):
        raise ValueError('Could not find ground truth annotations at "%s"' % json_path)

    return json.load(open(json_path))


def load_photo_ids_for_split(splits_dir, dataset_split):
    """ Loads photo ids in a SAW dataset split. """
    split_name = {'R': 'train', 'V': 'val', 'E': 'test'}[dataset_split]
    photo_ids_path = os.path.join(splits_dir, '%s_ids.npy' % split_name)
    return np.load(photo_ids_path)


def load_all_photo_ids(splits_dir):
    """ Loads all photo ids in the SAW dataset. """
    splits = [
        list(load_photo_ids_for_split(splits_dir, dataset_split))
        for dataset_split in ['R', 'V', 'E']
    ]
    return list(itertools.chain(*splits))


def load_photo(saw_image_dir, photo_id):
    """ Loads a photo from disk. """
    img_path = os.path.join(saw_image_dir, '%s.png' % photo_id)
    return Image.open(img_path)


def load_depth_normals(nyu_dataset_dir):
    """ Loads depth, normals and masks which specify valid pixels for all
    photos in the NYUv2 dataset. The correspondance between the image index of
    in these arrays and the photo ID is given by the ``nyu_idx`` attribute of
    the photo in the SAW annotation JSON files. """
    import h5py
    mat_filename = os.path.join(nyu_dataset_dir, 'nyu_depth_v2_labeled.mat')
    depths = h5py.File(mat_filename)['depths']
    normals_filepath = os.path.join(nyu_dataset_dir, 'normals.npy')
    normals = np.load(normals_filepath)
    masks_filepath = os.path.join(nyu_dataset_dir, 'masks.npy')
    masks = np.load(masks_filepath)

    return depths, normals, masks


def vis_pixel_labels(saw_image_dir, pixlabel_dir, photo_id, pixel_labels):
    """ Saves an image which visualizes the SAW pixel labels as colored pixels
    blended with the original photo. """
    img = load_photo(saw_image_dir, photo_id)
    w, h = img.size
    # Save image with labels overlayed
    label_img = np.full((h, w, 4), 0.0, dtype=float)
    # 0: depth/normal discontinuities (red)
    # 1: shadow boundaries (cyan)
    # 2: constant shading regions (green)
    alpha = 0.6
    colors = np.array([
        [1, 0, 0, alpha],
        [0, 1, 1, alpha],
        [0, 1, 0, alpha],
    ])
    for l in xrange(colors.shape[0]):
        label_img[pixel_labels == l, :] = colors[l]

    pil_label_img = numpy_to_pil(label_img)
    img.paste(pil_label_img, (0, 0), pil_label_img)
    img.save(os.path.join(pixlabel_dir, '%s_labels.png' % photo_id))


###############
# Image processing
###############


def compute_gradmag(image_arr):
    """ Compute gradient magnitude image of a 2D (grayscale) image. """
    assert image_arr.ndim == 2
    dy = sobel(image_arr, axis=0)
    dx = sobel(image_arr, axis=1)
    return np.hypot(dx, dy)


def compute_color_gradmag(image_arr):
    """ Compute average gradient magnitude of a 3D image (2D image with
    multiple channels). """
    if image_arr.ndim != 3 or image_arr.shape[2] != 3:
        raise ValueError('The image should have 3 channels!')

    dy = sobel(image_arr, axis=0)
    dx = sobel(image_arr, axis=1)
    return np.mean(np.hypot(dx, dy), axis=2)


def srgb_to_rgb(srgb):
    """ Convert an image from sRGB to linear RGB.

    :param srgb: numpy array in range (0.0 to 1.0)
    """
    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret


def pil_to_numpy(pil):
    """ Convert an 8bit PIL image (0 to 255) to a floating-point numpy array
    (0.0 to 1.0). """
    return np.asarray(pil).astype(float) / 255.0


def numpy_to_pil(img):
    """ Convert a floating point numpy array (0.0 to 1.0) to an 8bit PIL image
    (0 to 255). """
    return Image.fromarray(
        np.clip(img * 255, 0, 255).astype(np.uint8)
    )


###############
# Geometry
###############


def parse_vertices(vertices_str):
    """
    Parse vertices stored as a string.

    :param vertices_str: "x1,y1,x2,y2,...,xn,yn"
    :param return: [(x1,y1), (x1, y2), ... (xn, yn)]
    """
    s = [float(t) for t in vertices_str.split(',')]
    return zip(s[::2], s[1::2])


def parse_segments(segments_str):
    """
    Parse segments stored as a string.

    :param segments_str: "v1,v2,v3,..."
    :param return: [(v1,v2), (v3, v4), (v5, v6), ... ]
    """
    s = [int(t) for t in segments_str.split(',')]
    return zip(s[::2], s[1::2])


def parse_triangles(triangles_str):
    """
    Parse a list of vertices.

    :param triangles_str: "v1,v2,v3,..."
    :return: [(v1,v2,v3), (v4, v5, v6), ... ]
    """
    s = [int(t) for t in triangles_str.split(',')]
    return zip(s[::3], s[1::3], s[2::3])


def render_full_complex_polygon_mask(vertices, triangles, width, height,
                                     inverted=False, mode='1'):
    """
    Returns a black-and-white PIL image (mode ``1``) of size ``width`` x
    ``height``.  The image is not cropped to the bounding box of the vertices.
    Pixels inside the polygon are ``1`` and pixels outside are
    ``0`` (unless ``inverted=True``).

    :param vertices: List ``[[x1, y1], [x2, y2]]`` or string
        ``"x1,y1,x2,y2,...,xn,yn"``

    :param triangles: List ``[[v1, v2, v3], [v1, v2, v3]]`` or string
        ``"v1,v2,v3,v1,v2,v3,..."``, where ``vx`` is an index into the vertices
        list.

    :param width: width of the output mask

    :param height: height of the output mask

    :param inverted: if ``True``, swap ``0`` and ``1`` in the output.

    :param mode: PIL mode to use

    :return: PIL image of size (width, height)
    """

    if isinstance(vertices, basestring):
        vertices = parse_vertices(vertices)
    if isinstance(triangles, basestring):
        triangles = parse_triangles(triangles)

    if mode == '1':
        if inverted:
            fg, bg = 0, 1
        else:
            fg, bg = 1, 0
    elif mode == 'L':
        if inverted:
            fg, bg = 0, 255
        else:
            fg, bg = 255, 0
    else:
        raise NotImplementedError("TODO: implement mode %s" % mode)

    # scale up to size
    vertices = [(int(x * width), int(y * height)) for (x, y) in vertices]

    # draw triangles
    poly = Image.new(mode=mode, size=(width, height), color=bg)
    draw = ImageDraw.Draw(poly)
    for tri in triangles:
        points = [vertices[tri[t]] for t in (0, 1, 2)]
        draw.polygon(points, fill=fg, outline=fg)
    del draw

    return poly


###############
# Progress bar
###############


def progress_bar(l, show_progress=True):
    """ Returns an iterator for a list or queryset that renders a progress bar
    with a countdown timer """
    if show_progress:
        return iterator_progress_bar(l)
    else:
        return l


def iterator_progress_bar(iterator, maxval=None):
    """ Returns an iterator for an iterator that renders a progress bar with a
    countdown timer. """

    from progressbar import ProgressBar, SimpleProgress, Bar, ETA
    pbar = ProgressBar(
        maxval=maxval,
        widgets=[SimpleProgress(sep='/'), ' ', Bar(), ' ', ETA()],
    )
    return pbar(iterator)
