#!/usr/bin/env python3
"""
Preprocesses the raw imaging data for use in deep learning.

TODO:
    - Implement Chris masking with multi-processing (only largest object, same param, etc.)
    - Test listing of skipped files
    - Make background removal only keep largest object when crop single plant
    - Figure out a way to deal with overlap (PlantCV, neural network?)
    - Add argparse functionality for config
"""

# Import statements
import os
import numpy as np
import matplotlib.pyplot as plt
import re
from skimage import io, color, filters, segmentation, util, morphology, measure
from skimage.transform import rescale
from multiprocessing.pool import Pool
from tqdm import tqdm
from functools import partial
from PIL import Image

# Import supporting modules
import utils


# Remove alpha channel - adapted from Chris Dijkstra
def no_alpha(rgb_im):
    """Removes the alpha channel of an RGB image with 4 channels

    As color.rgba2rgb converts values to floats between 0 and 1

    Args:
        rgb_im (numpy.ndarray): 4 dimensional array representing an RGB image

    Returns:
        numpy.ndarray: 3 dimensional array representing an RGB image
    """
    assert rgb_im.shape[2] == 4, "Input RGB image doesn't have an alpha channel"

    # Blend alpha channel
    alphaless = color.rgba2rgb(rgb_im)

    # Convert values from float64 back to uint8
    alphaless = util.img_as_ubyte(alphaless)
    return alphaless


# Crop specific region - adapted from Chris Dijkstra
def crop_region(image, centre, shape):
    """Crops an image area of specified width and height around a central point

    :param image: np.ndarray, matrix representing the image
    :param centre: tuple, contains the x and y coordinate of the centre as
        integers
    :param shape: tuple, contains the height and width of the subregion in
        pixels as integers
    :return: The cropped region of the original image
    """
    shape_r = np.array(shape)
    shape_r[shape_r % 2 == 1] += 1
    if image.ndim == 2:
        crop = image[
            centre[1] - shape_r[1] // 2 : centre[1] + shape[1] // 2,
            centre[0] - shape_r[0] // 2 : centre[0] + shape[0] // 2,
        ]
    else:
        crop = image[
            centre[1] - shape_r[1] // 2 : centre[1] + shape[1] // 2,
            centre[0] - shape_r[0] // 2 : centre[0] + shape[0] // 2,
            :,
        ]
    return crop


# Define backgrond removal functions - adapted from: Chris Dijkstra
def elevation_map(rgb_im):
    """Creates an elevation map of an RGB image based on sobel filtering

    :param rgb_im: numpy.ndarray, 3 dimensional array representing an RGB image
    :return: numpy.ndarray, 2 dimensional array representing an edge map
    """
    compound_sobel = filters.sobel(rgb_im)
    compound_sobel = (
        compound_sobel[:, :, 0] + compound_sobel[:, :, 1] + compound_sobel[:, :, 2]
    )
    elevation = filters.sobel(compound_sobel)
    return elevation


def map_grid(n_points, shape):
    """Creates point grid within a certain shape

    :param n_points: int, number of points that the grid should contain
    :param shape: iterable, an iterable containing the rows (int) and
        columns (int)
    :return np.ndarray, a numpy array with the specified shape, containing an
        equally spaced grid with the specified number of points
    """
    grid = util.regular_grid(shape, n_points)
    grid_map = np.zeros(shape)
    grid_map[grid] = np.arange(grid_map[grid].size).reshape(grid_map[grid].shape) + 1
    return grid_map


def multichannel_threshold(multi_ch_im, x_th=0.0, y_th=0.0, z_th=0.0, inverse=False):
    """Takes a three-channel image and returns a mask based on thresholds

    :param multi_ch_im: np.nd_array a numpy array representing an image with
        three color channels
    :param x_th: float, the threshold for the first channel, 0.0 by default
    :param y_th: float, the threshold for the second channel, 0.0 by default
    :param z_th: float, the threshold for the third channel, 0.0 by default
    :param inverse: bool, if False pixels below the threshold are marked as 0,
        if True, pixels above the threshold are marked as 0.
    :return: np.nd_array, the mask created based on the thresholds, 2D array
        same width and height as the input
    """
    mask = np.ones(multi_ch_im.shape[0:2])
    mask[multi_ch_im[:, :, 0] < x_th] = 0
    mask[multi_ch_im[:, :, 1] < y_th] = 0
    mask[multi_ch_im[:, :, 2] < z_th] = 0
    mask = mask.astype(int)
    if inverse:
        mask = np.invert(mask)
    return mask


def watershed_blur(rgb_im, n_seeds):
    """Performs watershed averaging of color, preserving edges

    :param rgb_im: np.ndarray, 3 dimensional array representing an RGB image
    :param n_seeds: int, number of points that the grid should contain
    :return: np.ndarray, 3 dimensional array representing an RGB image. Colors
        are the colors of the input image averaged over watershed regions.
    """
    elevation = elevation_map(rgb_im)
    seeds = map_grid(n_seeds, rgb_im.shape[0:2])
    labels = segmentation.watershed(elevation, seeds)
    average_cols = color.label2rgb(labels, rgb_im, kind="avg")
    return average_cols


def water_hsv_thresh(rgb_im, n_seeds, h_th=0.0, s_th=0.0, v_th=0.0):
    """Segments an image based on hsv thresholds, after watershed averaging

    :param rgb_im: np.ndarray, 3 dimensional array representing an RGB image
    :param n_seeds: int, number of initialized seeds for watershedding
    :param h_th: float, the threshold for the hue channel, everything below this
        value is marked as background
    :param s_th: float, the threshold for the value channel, everything below
        this value is marked as background
    :param v_th: float, the threshold for the saturation channel everything
        below this value is marked as background
    :return: np.ndarray, 2D mask with boolean values
    """
    blurred = watershed_blur(rgb_im, n_seeds)
    hsv_blurred = color.rgb2hsv(blurred)
    mask = multichannel_threshold(hsv_blurred, h_th, s_th, v_th)
    return mask.astype(int)


# Define function to only keep object in center of binary mask
def center_object(mask, center_scale=0.1):
    """Returns mask with only the object at the center of a binary mask.

    Uses a center area which is the size of the mask times argument center_scale.

    Args:
        mask (np.ndarray): Binary mask.
        center_scale (float): Scale of center area of mask dimensions. Defaults to 0.1.

    Returns:
        np.ndarray: Binary mask containing only the object touching the center area.
    """
    # Label connections regions to distinguish different objects
    labels = measure.label(mask)

    # Define center area
    center = np.array(mask.shape) // 2
    center_area = np.zeros_like(mask)
    offset_x = int(mask.shape[0] * center_scale)
    offset_y = int(mask.shape[1] * center_scale)
    center_area[
        center[0] - offset_x : center[0] + offset_x,
        center[1] - offset_y : center[1] + offset_y,
    ] = 1

    # Find the labels of objects that touch the center area
    label_on_center = np.unique(labels[center_area == 1])
    label_on_center = label_on_center[label_on_center > 0]  # don't include background

    # Create a mask for the objects that touch the center area
    center_object_mask = np.isin(labels, label_on_center)
    return center_object_mask


# Define parser for tray registration file for info on trays and plants
def read_tray(filepath):
    """Parses the tray registraton tab-delimited file as a dictionary.

    Args:
        filepath (str): Filepath to the tray registration file.

    Returns:
        dict: Dictionary with structure:
            {column1: np.ndarray([value1, value2, value3]),
            column2: np.ndarray([value1, value2, value3])}
    """
    # Read tsv as an array containing arrays for every line
    line_arrays = np.loadtxt(filepath, dtype=str, delimiter="\t", skiprows=1)

    # Turn arrays into dict with column names as keys
    keys = line_arrays[0]
    vals = np.transpose(line_arrays[1:])
    tray_dict = dict(zip(keys, vals))
    return tray_dict


# Define crop functions
def indiv_crop(rgb_img, crop_size, dist_plants, num_plants):
    """Crop individual plants from the RGB images.

    Assumes that distance between plants is same along width and height.

    Args:
        rgb_img (np.ndarray): RGB image as np.ndarray.
        crop_size (tuple): Tuple of ints (width, height) with desired resolution of image crops.
        dist_plants (int): Distance in px between plants along width and height.
        num_plants (int): Number of plants in image, 4 or 5.

    Returns:
        list: Contains cropped RGB images as np.ndarrays.
    """
    # Calculate center coordinates of RGB
    center_x = int(rgb_img.shape[1] / 2 + 0.5)
    center_y = int(rgb_img.shape[0] / 2 + 0.5)

    # Crop for 4 plants
    if num_plants == 4:
        # Crop RGB
        rgb_area1 = crop_region(
            image=rgb_img,
            centre=(center_x, center_y - dist_plants),
            shape=crop_size,
        )
        rgb_area2 = crop_region(
            image=rgb_img,
            centre=(center_x - dist_plants, center_y),
            shape=crop_size,
        )
        rgb_area3 = crop_region(
            image=rgb_img,
            centre=(center_x + dist_plants, center_y),
            shape=crop_size,
        )
        rgb_area4 = crop_region(
            image=rgb_img,
            centre=(center_x, center_y + dist_plants),
            shape=crop_size,
        )

        # Compile crops into lists
        rgb_crops = [rgb_area1, rgb_area2, rgb_area3, rgb_area4]

    # Crop for 5 plants
    elif num_plants == 5:
        # Crop RGB
        rgb_area1 = crop_region(
            image=rgb_img,
            centre=(
                center_x - dist_plants,
                center_y - dist_plants,
            ),
            shape=crop_size,
        )
        rgb_area2 = crop_region(
            image=rgb_img,
            centre=(
                center_x + dist_plants,
                center_y - dist_plants,
            ),
            shape=crop_size,
        )
        rgb_area3 = crop_region(
            image=rgb_img,
            centre=(center_x, center_y),
            shape=crop_size,
        )
        rgb_area4 = crop_region(
            image=rgb_img,
            centre=(
                center_x - dist_plants,
                center_y + dist_plants,
            ),
            shape=crop_size,
        )
        rgb_area5 = crop_region(
            image=rgb_img,
            centre=(
                center_x + dist_plants,
                center_y + dist_plants,
            ),
            shape=crop_size,
        )

        # Compile crops into lists
        rgb_crops = [rgb_area1, rgb_area2, rgb_area3, rgb_area4, rgb_area5]

    # Return lists of cropped images
    return rgb_crops


def overlay_crop(rgb_img, fm_img, fvfm_img, crop_size, dist_plants, num_plants):
    """Crops plants from RGB, Fm and FvFm images in such a way that the crops overlap.

    Assumes that the only transformations still needed to make the crops overlap are
    simply shifts along width and height. Thus, images should have been rescaled beforehand
    to make sure plants are of same scale.

    Assumes that distance between plants is same along width and height.

    Shifts along width and height were taken from observations on several images
    and have been hard-coded.

    Args:
        rgb_img (np.ndarray): RGB image as np.ndarray.
        fm_img (np.ndarray): Fm image as np.ndarray.
        fvfm_img (np.ndarray): Fv/Fm image as np.ndarray.
        crop_size (tuple): Tuple of ints (width, height) with desired resolution of image crops.
        dist_plants (int): Distance in px between plants along width and height.
        num_plants (int): Number of plants in image, 4 or 5.

    Returns:
        list: Contains cropped RGB images as np.ndarrays.
        list: Contains cropped Fm images as np.ndarrays.
        list: Contains cropped Fv/Fm images as np.ndarrays.
    """
    # Calculate center coordinates of RGB
    center_x = int(rgb_img.shape[1] / 2 + 0.5)
    center_y = int(rgb_img.shape[0] / 2 + 0.5)

    # Calculate offset between RGB and fluorescence images
    offset_x = int((rgb_img.shape[1] - fm_img.shape[1]) / 2)
    offset_y = int((rgb_img.shape[0] - fm_img.shape[0]) / 2)

    # Crop for 4 plants
    if num_plants == 4:
        # Crop RGB
        rgb_area1 = crop_region(
            image=rgb_img,
            centre=(center_x + 1, center_y - dist_plants + 15),
            shape=crop_size,
        )
        rgb_area2 = crop_region(
            image=rgb_img,
            centre=(center_x - dist_plants + 2, center_y + 11),
            shape=crop_size,
        )
        rgb_area3 = crop_region(
            image=rgb_img,
            centre=(center_x + dist_plants - 3, center_y + 14),
            shape=crop_size,
        )
        rgb_area4 = crop_region(
            image=rgb_img,
            centre=(center_x - 1, center_y + dist_plants + 8),
            shape=crop_size,
        )

        # Crop Fm
        fm_area1 = crop_region(
            image=fm_img,
            centre=(center_x - offset_x, center_y - dist_plants - offset_y),
            shape=crop_size,
        )
        fm_area2 = crop_region(
            image=fm_img,
            centre=(center_x - dist_plants - offset_x, center_y - offset_y),
            shape=crop_size,
        )
        fm_area3 = crop_region(
            image=fm_img,
            centre=(center_x + dist_plants - offset_x, center_y - offset_y),
            shape=crop_size,
        )
        fm_area4 = crop_region(
            image=fm_img,
            centre=(center_x - offset_x, center_y + dist_plants - offset_y),
            shape=crop_size,
        )

        # Crop Fv/Fm
        fvfm_area1 = crop_region(
            image=fvfm_img,
            centre=(center_x - offset_x, center_y - dist_plants - offset_y),
            shape=crop_size,
        )
        fvfm_area2 = crop_region(
            image=fvfm_img,
            centre=(center_x - dist_plants - offset_x, center_y - offset_y),
            shape=crop_size,
        )
        fvfm_area3 = crop_region(
            image=fvfm_img,
            centre=(center_x + dist_plants - offset_x, center_y - offset_y),
            shape=crop_size,
        )
        fvfm_area4 = crop_region(
            image=fvfm_img,
            centre=(center_x - offset_x, center_y + dist_plants - offset_y),
            shape=crop_size,
        )

        # Compile crops into lists
        rgb_crops = [rgb_area1, rgb_area2, rgb_area3, rgb_area4]
        fm_crops = [fm_area1, fm_area2, fm_area3, fm_area4]
        fvfm_crops = [fvfm_area1, fvfm_area2, fvfm_area3, fvfm_area4]

    # Crop for 5 plants
    elif num_plants == 5:
        # Crop RGB
        rgb_area1 = crop_region(
            image=rgb_img,
            centre=(center_x - dist_plants + 3, center_y - dist_plants + 12),
            shape=crop_size,
        )
        rgb_area2 = crop_region(
            image=rgb_img,
            centre=(center_x + dist_plants - 1, center_y - dist_plants + 14),
            shape=crop_size,
        )
        rgb_area3 = crop_region(
            image=rgb_img, centre=(center_x, center_y + 10), shape=crop_size
        )
        rgb_area4 = crop_region(
            image=rgb_img,
            centre=(center_x - dist_plants + 1, center_y + dist_plants + 6),
            shape=crop_size,
        )
        rgb_area5 = crop_region(
            image=rgb_img,
            centre=(center_x + dist_plants - 4, center_y + dist_plants + 9),
            shape=crop_size,
        )

        # Crop Fm
        fm_area1 = crop_region(
            image=fm_img,
            centre=(
                center_x - dist_plants - offset_x,
                center_y - dist_plants - offset_y,
            ),
            shape=crop_size,
        )
        fm_area2 = crop_region(
            image=fm_img,
            centre=(
                center_x + dist_plants - offset_x,
                center_y - dist_plants - offset_y,
            ),
            shape=crop_size,
        )
        fm_area3 = crop_region(
            image=fm_img,
            centre=(center_x - offset_x, center_y - offset_y),
            shape=crop_size,
        )
        fm_area4 = crop_region(
            image=fm_img,
            centre=(
                center_x - dist_plants - offset_x,
                center_y + dist_plants - offset_y,
            ),
            shape=crop_size,
        )
        fm_area5 = crop_region(
            image=fm_img,
            centre=(
                center_x + dist_plants - offset_x,
                center_y + dist_plants - offset_y,
            ),
            shape=crop_size,
        )

        # Crop Fv/Fm
        fvfm_area1 = crop_region(
            image=fvfm_img,
            centre=(
                center_x - dist_plants - offset_x,
                center_y - dist_plants - offset_y,
            ),
            shape=crop_size,
        )
        fvfm_area2 = crop_region(
            image=fvfm_img,
            centre=(
                center_x + dist_plants - offset_x,
                center_y - dist_plants - offset_y,
            ),
            shape=crop_size,
        )
        fvfm_area3 = crop_region(
            image=fvfm_img,
            centre=(center_x - offset_x, center_y - offset_y),
            shape=crop_size,
        )
        fvfm_area4 = crop_region(
            image=fvfm_img,
            centre=(
                center_x - dist_plants - offset_x,
                center_y + dist_plants - offset_y,
            ),
            shape=crop_size,
        )
        fvfm_area5 = crop_region(
            image=fvfm_img,
            centre=(
                center_x + dist_plants - offset_x,
                center_y + dist_plants - offset_y,
            ),
            shape=crop_size,
        )

        # Compile crops into lists
        rgb_crops = [rgb_area1, rgb_area2, rgb_area3, rgb_area4, rgb_area5]
        fm_crops = [fm_area1, fm_area2, fm_area3, fm_area4, fm_area5]
        fvfm_crops = [fvfm_area1, fvfm_area2, fvfm_area3, fvfm_area4, fvfm_area5]

    else:
        print(num_plants)
        raise ("num_plants should be 4 or 5")

    # Return lists of cropped images
    return rgb_crops, fm_crops, fvfm_crops


# Define end-to-end overlay crop function for multi-processing
def path_crop(
    rgb_path,
    tray_reg,
    crop_shape,
    crop_dist,
    rm_alpha=True,
    rgb_save_dir=None,
    dataset="lettuce",
):
    """Crops individual plants from RGB image from filepath.

    This function combines everything from filepath to cropping so
    it can used for multi-processing or -threading.

    If one of the image files can't be read, the function returns None.

    Args:
        rgb_path (str): Filepath to RGB image.
        tray_reg (dict): Dictionary of information from the tray registration file. Only for lettuce dataset.
        crop_shape (tuple): Tuple of ints for shape of crop.
        crop_dist (int): Vertical/horizontal distance between plants on tray.
        rm_alpha (bool, optional): Removes alpha channel from RGB. Defaults to True.
        rgb_save_dir (str, optional): Directory to save RGB crops. Defaults to None.
        dataset (str, optional): Kind of dataset. "lettuce" or "potato". To determine how to crop. Defaults to "lettuce".

    Returns:
        list: Contains cropped RGB images as np.ndarrays.
    """
    # Read file, if can't read one of the files, function returns nothing
    try:
        rgb = io.imread(rgb_path)
    except:
        return

    # Remove alpha from RGB image if desired and image has 4 channels
    if rm_alpha and rgb.shape[2] == 4:
        rgb = no_alpha(rgb)

    if dataset == "lettuce":
        # Extract tray ID from RGB image filename
        rgb_name = os.path.basename(rgb_path)
        regex_trayID = re.compile(".+Tray_(\d{2,})")
        match_trayID = regex_trayID.match(rgb_name)
        trayID = match_trayID.group(1)

        # Determine if image file has 4 or 5 plants
        all_trayIDs = tray_reg["TrayID"]
        bool_ind_trayID = np.core.defchararray.find(all_trayIDs, trayID) != -1
        ind_trayID = np.flatnonzero(bool_ind_trayID)
        num_plants = len(ind_trayID)

    elif dataset == "potato":
        # Extract tray ID from RGB image filename
        rgb_name = os.path.basename(rgb_path)
        regex_trayID = re.compile(".+circlewrong_(\d{2,})")
        match_trayID = regex_trayID.match(rgb_name)
        trayID = int(match_trayID.group(1))

        # Determine if image file has 4 or 5 plants
        if trayID % 2 == 1:
            num_plants = 4
        else:
            num_plants = 5

    # Crop RGB, Fm and FvFm crops in such a way that they overlap
    rgb_crops = indiv_crop(
        rgb, crop_size=crop_shape, dist_plants=crop_dist, num_plants=num_plants
    )

    # Save cropped images if desired, with plant names in filename
    if rgb_save_dir is not None:
        if dataset == "lettuce":
            # Extract plant names
            all_plantnames = tray_reg["PlantName"]
            plantnames = all_plantnames[ind_trayID]

            # Count to add correct area number and plantname
            count = 0
            for rgb_crop in rgb_crops:
                old_name = os.path.basename(rgb_path)
                new_name = f"{os.path.splitext(old_name)[0]}_A{count + 1}_{plantnames[count]}.png"
                count += 1
                utils.save_img(rgb_crop, target_dir=rgb_save_dir, filename=new_name)

    return rgb_crops


# Define end-to-end overlay crop function for multi-processing
def path_overlay_crop(
    imgtype_paths,
    scale_rgb,
    tray_reg,
    crop_shape,
    crop_dist,
    rm_alpha=True,
    rgb_save_dir=None,
    fm_save_dir=None,
    fvfm_save_dir=None,
):
    """Crops plants from RGB, Fm and FvFm images from filepaths.

    This function combines everything from filepath to cropping so
    it can used for multi-processing or -threading.

    Input filepaths should be a tuple of filepaths for RGB, Fm and Fv/Fm
    images, in that specific order. If one of the image files can't be
    read, the function returns None.

    Cropping is done in such a way that the plants overlap by using
    vertical and horizontal shifts.

    Args:
        imgtype_paths (tuple): Tuple of RGB, Fm and Fv/Fm filepaths as str.
        scale_rgb (tuple): Tuple of floats to rescale each dimension of RGB image.
        tray_reg (dict): Dictionary of information from the tray registration file.
        crop_shape (tuple): Tuple of ints for shape of crop.
        crop_dist (int): Vertical/horizontal distance between plants on tray.
        rm_alpha (bool, optional): Removes alpha channel from RGB. Defaults to True.
        rgb_save_dir (str, optional): Directory to save RGB crops. Defaults to None.
        fm_save_dir (str, optional): Directory to save Fm crops. Defaults to None.
        fvfm_save_dir (str, optional): Directory to save FvFm crops. Defaults to None.

    Returns:
        list: Contains cropped RGB images as np.ndarrays.
        list: Contains cropped Fm images as np.ndarrays.
        list: Contains cropped Fv/Fm images as np.ndarrays.
    """
    # Read file, if can't read one of the files, function returns nothing
    try:
        rgb = io.imread(imgtype_paths[0])
        fm = utils.read_fimg(imgtype_paths[1])
        fvfm = utils.read_fimg(imgtype_paths[2])
    except:
        return

    # Remove alpha from RGB image if desired and image has 4 channels
    if rm_alpha and rgb.shape[2] == 4:
        rgb = no_alpha(rgb)

    # Rescale RGB image to allow overlaying with fluorescence images
    re_rgb = rescale(image=rgb, scale=scale_rgb, anti_aliasing=True)

    # Convert RGB back to uint8 after rescaling and convert Fv/Fm to uint8
    re_rgb = util.img_as_ubyte(re_rgb)
    fvfm = util.img_as_ubyte(fvfm)

    # Extract experiment number and tray ID from RGB image filename
    rgb_name = os.path.basename(imgtype_paths[0])
    exp = rgb_name[:2]
    regex_trayID = re.compile(".+Tray_(\d{2,})")
    match_trayID = regex_trayID.match(rgb_name)
    trayID = int(match_trayID.group(1))

    # Determine if image file has 4 or 5 plants
    all_trayIDs = tray_reg["TrayID"]
    all_trayIDs = all_trayIDs[
        tray_reg["Experiment"] == exp
    ]  # filter on corresponding experiment
    all_trayIDs = [all_trayID.split("_")[-1] for all_trayID in all_trayIDs]
    all_trayIDs = np.array(all_trayIDs, dtype=int)
    ind_trayID = np.where(all_trayIDs == trayID)[0]
    num_plants = len(ind_trayID)

    # Crop RGB, Fm and FvFm crops in such a way that they overlap
    rgb_crops, fm_crops, fvfm_crops = overlay_crop(
        re_rgb,
        fm,
        fvfm,
        crop_size=crop_shape,
        dist_plants=crop_dist,
        num_plants=num_plants,
    )

    # Save cropped images if desired, with plant names in filename
    if any([rgb_save_dir, fm_save_dir, fvfm_save_dir]):
        all_plantnames = tray_reg["PlantName"]
        all_plantnames = all_plantnames[
            tray_reg["Experiment"] == exp
        ]  # filter on corresponding experiment
        plantnames = all_plantnames[ind_trayID]

    if rgb_save_dir is not None:
        # Count to add correct area number and plantname
        count = 0
        for rgb_crop in rgb_crops:
            old_name = os.path.basename(imgtype_paths[0])
            new_name = (
                f"{os.path.splitext(old_name)[0]}_A{count + 1}_{plantnames[count]}.png"
            )
            count += 1
            utils.save_img(rgb_crop, target_dir=rgb_save_dir, filename=new_name)

    if fm_save_dir is not None:
        # Count to add correct area number and plantname
        count = 0
        for fm_crop in fm_crops:
            old_name = os.path.basename(imgtype_paths[0])
            new_name = (
                f"{os.path.splitext(old_name)[0]}_A{count + 1}_{plantnames[count]}.tif"
            )
            count += 1
            fm_crop = Image.fromarray(fm_crop)
            utils.save_img(fm_crop, target_dir=fm_save_dir, filename=new_name)

    if fvfm_save_dir is not None:
        # Count to add correct area number and plantname
        count = 0
        for fvfm_crop in fvfm_crops:
            old_name = os.path.basename(imgtype_paths[0])
            new_name = (
                f"{os.path.splitext(old_name)[0]}_A{count + 1}_{plantnames[count]}.png"
            )
            count += 1
            utils.save_img(fvfm_crop, target_dir=fvfm_save_dir, filename=new_name)

    return rgb_crops, fm_crops, fvfm_crops


# Define end-to-end background removal function for multi-processing
def path_back_mask(rgb_im_path, rm_alpha, n_seeds, h_th=0.0, s_th=0.0, v_th=0.0):
    """Masks background of image from the corresponding path.

    Uses hsv thresholds after watershed averaging for background segmentation.
    This function compiles all necessary tasks from image path to mask in order
    to be used in multiprocessing or multithreading,
    as they require a single function from start to finish.

    :param rgb_im_path: str, path of RGB image
    :param rm_alpha: bool, if True removes alpha channel of image
    :param n_seeds: int, number of initialized seeds for watershedding
    :param h_th: float, the threshold for the hue channel, everything below this
        value is marked as background
    :param s_th: float, the threshold for the value channel, everything below
        this value is marked as background
    :param v_th: float, the threshold for the saturation channel everything
        below this value is marked as background
    :return: np.ndarray, 2D mask with boolean values
    """
    # Read file, if can't read file, function returns nothing
    try:
        img = io.imread(rgb_im_path)
    except:
        return
    if rm_alpha:
        img = no_alpha(img)

    # Mask out the background from the plants
    back_mask = water_hsv_thresh(img, n_seeds, h_th, s_th, v_th)

    # Remove small noise and fill small holes
    back_mask = morphology.binary_opening(back_mask)
    back_mask = morphology.remove_small_objects(back_mask)
    hole_threshold = np.prod(np.array(back_mask.shape[:2])) ** 0.5 * 0.05
    back_mask = morphology.remove_small_holes(back_mask, area_threshold=hole_threshold)

    # Only keep the plant at the center of the image
    back_mask = center_object(back_mask)

    return back_mask


def main():
    # Set config
    rgb_dir = "/lustre/BIF/nobackup/to001/thesis_MBF/data/complete/RGB_Original_FvFm_timepoints"
    fm_dir = "/lustre/BIF/nobackup/to001/thesis_MBF/data/complete/Fm_fimg"
    fvfm_dir = "/lustre/BIF/nobackup/to001/thesis_MBF/data/complete/FvFm_fimg"
    rgb_crop_dir = "/lustre/BIF/nobackup/to001/thesis_MBF/data/complete/rgb_crops"
    fm_crop_dir = "/lustre/BIF/nobackup/to001/thesis_MBF/data/complete/fm_crops"
    fvfm_crop_dir = "/lustre/BIF/nobackup/to001/thesis_MBF/data/complete/fvfm_crops"
    rgb_mask_dir = "/lustre/BIF/nobackup/to001/thesis_MBF/data/complete/rgb_masks"
    fm_mask_dir = "/lustre/BIF/nobackup/to001/thesis_MBF/data/complete/fm_masks"
    CORES = 40
    CROP = True
    OVERLAY_IMG = True
    RESCALE_RGB = (0.36, 0.36, 1)
    CROP_DIST = 265  # no overlay: 736
    CROP_SHAPE = (484, 484)  # no overlay: (1560, 1560)
    MASK = False
    SEEDS = 1500
    H_THRES = 0
    S_THRES = 0.25
    V_THRES = 0.2

    # Prepare to track skipped images
    skipped = []

    # Read tray registration for info on image files
    tray_reg = read_tray(
        "/lustre/BIF/nobackup/to001/thesis_MBF/data/Tray_registration_B8.tsv"
    )

    # Crop original images
    if CROP:
        # Crop for overlay of RGB, Fm and Fv/Fm if desired
        if OVERLAY_IMG:
            # Retrieve and sort RGB, Fm and Fv/Fm filenames
            rgb_names = sorted(os.listdir(rgb_dir))
            fm_names = sorted(os.listdir(fm_dir))
            fvfm_names = sorted(os.listdir(fvfm_dir))

            # Create lists of filepaths
            rgb_filepaths = []
            fm_filepaths = []
            fvfm_filepaths = []
            for filenames in zip(rgb_names, fm_names, fvfm_names):
                # Join directory and filename to make filepath
                rgb_filepath = os.path.join(rgb_dir, filenames[0])
                fm_filepath = os.path.join(fm_dir, filenames[1])
                fvfm_filepath = os.path.join(fvfm_dir, filenames[2])

                # Append filepaths to lists of filepaths
                rgb_filepaths.append(rgb_filepath)
                fm_filepaths.append(fm_filepath)
                fvfm_filepaths.append(fvfm_filepath)

            # Cropping for overlay from filepaths of original images
            with Pool(processes=CORES) as pool:
                prepped_crop = partial(
                    path_overlay_crop,
                    scale_rgb=RESCALE_RGB,
                    tray_reg=tray_reg,
                    crop_shape=CROP_SHAPE,
                    crop_dist=CROP_DIST,
                    rgb_save_dir=rgb_crop_dir,
                    fm_save_dir=fm_crop_dir,
                    fvfm_save_dir=fvfm_crop_dir,
                )
                process_iter = pool.imap(
                    func=prepped_crop,
                    iterable=zip(rgb_filepaths, fm_filepaths, fvfm_filepaths),
                )
                process_loop = tqdm(
                    process_iter,
                    desc="RGB, Fm and Fv/Fm cropping",
                    total=len(rgb_filepaths),
                )

                # Execute processes in order and collect skipped files
                count = 0
                for crops in process_loop:  # processes are executed during loop
                    # Track which files were skipped
                    if crops is None:
                        skipped.append(rgb_names[0])
                        skipped.append(fm_names[0])
                        skipped.append(fvfm_names[0])

        # Only crop RGB images without downscaling for overlay
        else:
            # Retrieve and sort RGB filenames
            rgb_names = sorted(os.listdir(rgb_dir))

            # Create list of filepaths
            rgb_filepaths = []
            for rgb_name in rgb_names:
                # Join directory and filename to make filepath
                rgb_filepath = os.path.join(rgb_dir, rgb_name)

                # Append filepaths to lists of filepaths
                rgb_filepaths.append(rgb_filepath)

            # Cropping for overlay from filepaths of original images
            with Pool(processes=CORES) as pool:
                prepped_crop = partial(
                    path_crop,
                    tray_reg=tray_reg,
                    crop_shape=CROP_SHAPE,
                    crop_dist=CROP_DIST,
                    rgb_save_dir=rgb_crop_dir,
                )
                process_iter = pool.imap(
                    func=prepped_crop,
                    iterable=rgb_filepaths,
                )
                process_loop = tqdm(
                    process_iter,
                    desc="RGB cropping",
                    total=len(rgb_filepaths),
                )

                # Execute processes in order and collect skipped files
                count = 0
                for crops in process_loop:  # processes are executed during loop
                    # Track which files were skipped
                    if crops is None:
                        skipped.append(rgb_names[0])

    # Background mask cropped images
    if MASK:
        # Create filepaths of cropped images
        crop_names = os.listdir(rgb_crop_dir)
        crop_paths = []
        for crop_name in crop_names:
            crop_path = os.path.join(rgb_crop_dir, crop_name)
            crop_paths.append(crop_path)

        # Background segmentation from filepaths of cropped images
        with Pool(processes=CORES) as pool:
            prepped_mask = partial(
                path_back_mask,
                rm_alpha=False,
                n_seeds=SEEDS,
                h_th=H_THRES,
                s_th=S_THRES,
                v_th=V_THRES,
            )
            process_iter = pool.imap(func=prepped_mask, iterable=crop_paths)
            process_loop = tqdm(
                process_iter, desc="Background removal", total=len(crop_paths)
            )

            # Iterate over completed masks
            count = 0
            for mask in process_loop:
                cur_image_name = crop_names[count]
                new_image_name = f"{os.path.splitext(cur_image_name)[0]}_mask.png"

                # Update count
                count += 1

                # Save succesful masks and skip unreadable images
                try:
                    # Save mask
                    utils.save_img(
                        img=mask,
                        target_dir=rgb_mask_dir,
                        filename=new_image_name,
                    )
                except:
                    # Add unreadable file to list of skipped images
                    skipped.append(cur_image_name)

                    print(f"[ISSUE] Image {cur_image_name} was unreadable and skipped")

    # Save list of skipped images as text file in working directory
    skipped_path = os.path.join(os.getcwd(), "skipped_images.txt")
    with open(skipped_path, "w") as skip_file:
        for skip_name in skipped:
            skip_file.write(skip_name)
            skip_file.write("\n")


if __name__ == "__main__":
    main()
