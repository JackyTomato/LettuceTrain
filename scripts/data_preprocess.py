#!/usr/bin/env python3
"""
Preprocesses the raw imaging data for use in deep learning.

TODO:
    - Test listing of corrupt files
    - Add cropping of invidual plants, need  coords
    - Tweak background removal (seeds, HSV)
    - Make background removal only keep largest object when crop single plant
    - Figure out how to deal with tubes and shadows being included too
    - Figure out a way to deal with overlap (PlantCV, neural network?)
    - Add argparse functionality for config
"""

# Import statements
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, segmentation, util, morphology
from multiprocessing.pool import Pool
from tqdm import tqdm
from functools import partial
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
    alphaless = (alphaless * 255).astype(np.uint8)
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


# Define end-to-end crop function for multi-processing
def path_crop(rgb_im_path, rm_alpha, centre, shape):
    """Crops an image area of specified width and height around a central point from path

    :param rgb_im_path: string, filepath to image
    :param rm_alpha: bool, if True removes alpha channel of image
    :param centre: tuple, contains the x and y coordinate of the centre as
        integers
    :param shape: tuple, contains the height and width of the subregion in
        pixels as integers
    :return: The cropped region of the original image
    """
    # Read file, if can't read file, function returns nothing
    try:
        img = io.imread(rgb_im_path)
    except:
        return
    if rm_alpha:
        img = no_alpha(img)

    # Crop region
    shape_r = np.array(shape)
    shape_r[shape_r % 2 == 1] += 1
    if img.ndim == 2:
        crop = img[
            centre[1] - shape_r[1] // 2 : centre[1] + shape[1] // 2,
            centre[0] - shape_r[0] // 2 : centre[0] + shape[0] // 2,
        ]
    else:
        crop = img[
            centre[1] - shape_r[1] // 2 : centre[1] + shape[1] // 2,
            centre[0] - shape_r[0] // 2 : centre[0] + shape[0] // 2,
            :,
        ]
    return crop


# Define end-to-end background removal function for multi-processing
def path_back_mask(rgb_im_path, rm_alpha, n_seeds, h_th=0.0, s_th=0.0, v_th=0.0):
    """Masks background of image from the corresponding path

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

    # Make background mask
    back_mask = water_hsv_thresh(img, n_seeds, h_th, s_th, v_th)
    back_mask = morphology.binary_opening(back_mask)
    return back_mask


def main():
    # Set config
    data_dir = "/lustre/BIF/nobackup/to001/thesis_MBF/data/img/RGB"
    crop_dir = "/lustre/BIF/nobackup/to001/thesis_MBF/data/img/crops"
    mask_dir = "/lustre/BIF/nobackup/to001/thesis_MBF/data/img/masks"
    CROP = True
    CROP_POS = (2028, 1520)
    CROP_SHAPE = (2412, 2412)
    MASK = True
    SEEDS = 1500
    H_THRES = 0
    S_THRES = 0.25
    V_THRES = 0.2

    # Prepare to track corrupted images
    corrupt = []

    # Crop original images
    if CROP:
        # Create filepaths of original images
        image_names = os.listdir(data_dir)
        filepaths = []
        for image_name in image_names:
            filepath = os.path.join(data_dir, image_name)
            filepaths.append(filepath)

        # Multi-processed image processing
        num_cores = 4

        # Cropping from filepaths of original images
        with Pool(processes=num_cores) as pool:
            prepped_crop = partial(
                path_crop, rm_alpha=True, centre=CROP_POS, shape=CROP_SHAPE
            )
            process_iter = pool.imap(func=prepped_crop, iterable=filepaths)
            process_loop = tqdm(
                process_iter, desc="Image cropping", total=len(filepaths)
            )

            # Iterate over completed masks
            count = 0
            for crop in process_loop:
                cur_image_name = image_names[count]
                new_image_name = f"{os.path.splitext(cur_image_name)[0]}_crop.png"

                # Update count
                count += 1

                # Save succesful masks and skip unreadable images
                try:
                    # Save mask
                    utils.save_img(
                        img=crop,
                        target_dir=crop_dir,
                        filename=new_image_name,
                    )
                except:
                    # Add unreadable file to list of corrupted images
                    corrupt.append(cur_image_name)

                    print(f"[ISSUE] Image {cur_image_name} was unreadable and skipped")

    # Background mask cropped images
    if MASK:
        # Create filepaths of cropped images
        crop_names = os.listdir(crop_dir)
        crop_paths = []
        for crop_name in crop_names:
            crop_path = os.path.join(crop_dir, crop_name)
            crop_paths.append(crop_path)

        # Background segmentation from filepaths of cropped images
        with Pool(processes=num_cores) as pool:
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
                        target_dir=mask_dir,
                        filename=new_image_name,
                    )
                except:
                    # Add unreadable file to list of corrupted images
                    corrupt.append(cur_image_name)

                    print(f"[ISSUE] Image {cur_image_name} was unreadable and skipped")

    # Save list of corrupt images as text file
    corrupt_path = os.path.join(data_dir, "corrupt_images.txt")
    with open(corrupt_path, "w") as cor_file:
        for cor_name in corrupt:
            cor_file.write(cor_name)
            cor_file.write("\n")


if __name__ == "__main__":
    main()
