#!/usr/bin/env python3
"""
Preprocesses the raw imaging data for use in deep learning.

TODO:
    - Add plant cropper (get plant info data with coords from Alan)
    - Add background removal (Chris' script, PlantCV, neural network?)
    - Figure out a way to deal with overlap (PlantCV, neural network?)
"""

# Import statements
from skimage import io, color, filters, segmentation, util, morphology
import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
from tqdm import tqdm
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import imageio.plugins.pillow


# Remove alpha channel - author Chris Dijkstra
def no_alpha(rgb_im):
    """Removes the alpha channel of an RGB image with 4 channels

    Args:
        rgb_im (numpy.ndarray): 4 dimensional array representing an RGB image

    Returns:
        numpy.ndarray: 3 dimensional array representing an RGB image
    """
    if rgb_im.shape[2] == 4:
        alphaless = color.rgba2rgb(rgb_im)
    return alphaless


# Define backgrond removal functions - author: Chris Dijkstra
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
    :return: np.ndarray, 2D mask with the
    """
    blurred = watershed_blur(rgb_im, n_seeds)
    hsv_blurred = color.rgb2hsv(blurred)
    mask = multichannel_threshold(hsv_blurred, h_th, s_th, v_th)
    return mask.astype(int)


# Define end-to-end background removal fucntion for multithread
def path_back_mask(rgb_im_path, n_seeds, h_th=0.0, s_th=0.0, v_th=0.0):
    """Masks background of image from the corresponding path

    Uses hsv thresholds after watershed averaging for background segmentation.
    This function compiles all necessary tasks from image path to mask in order
    to be used in multiprocessing or multithreading,
    as they require a single function from start to finish.

    :param rgb_im_path: str, path of RGB image
    :param n_seeds: int, number of initialized seeds for watershedding
    :param h_th: float, the threshold for the hue channel, everything below this
        value is marked as background
    :param s_th: float, the threshold for the value channel, everything below
        this value is marked as background
    :param v_th: float, the threshold for the saturation channel everything
        below this value is marked as background
    :return: np.ndarray, 2D mask with the
    """
    # Read file, if can't read file, function returns nothing
    try:
        img = io.imread(rgb_im_path)
    except:
        print(f"[ISSUE] {rgb_im_path} was unreadable")
        return
    img = no_alpha(img)
    back_mask = water_hsv_thresh(img, n_seeds, h_th, s_th, v_th)
    back_mask = morphology.binary_opening(back_mask)
    print(f"[INFO] Mask made for {rgb_im_path}!")
    return back_mask


def main():
    # Create filepaths
    data_dir = "/lustre/BIF/nobackup/to001/thesis_MBF/data/img/RGB"
    image_names = os.listdir(data_dir)
    filepaths = []
    for image_name in image_names:
        filepath = os.path.join(data_dir, image_name)
        filepaths.append(filepath)

    # for image_name in image_names:
    #     filepath = os.path.join(data_dir, image_name)
    #     image = io.imread(filepath)
    #     image = no_alpha(image)
    #     plant_mask = water_hsv_thresh(image, 1500, s_th=0.25, v_th=0.2)
    #     plant_mask = morphology.binary_opening(plant_mask)
    #     print(f"{image_name} succesfully masked!")

    # Multi-processed background segmentation from filepaths
    num_cores = 4
    with Pool(processes=num_cores) as pool:
        prepped_mask = partial(path_back_mask, n_seeds=1500, s_th=0.25, v_th=0.2)
        process_iter = pool.imap(func=prepped_mask, iterable=filepaths)
        process_loop = tqdm(process_iter, total=len(filepaths), position=0, leave=False)
        count = 0
        for mask in process_loop:
            count += 1
            print(f"Image number {count} succesfully masked!")

    # Multi-threaded background segmentation from filepaths
    # num_cores = 2
    # with ThreadPoolExecutor(max_workers=num_cores) as pool:
    #     prepped_mask = partial(path_back_mask, n_seeds=1500, s_th=0.25, v_th=0.2)
    #     masks = list(
    #         tqdm(
    #             pool.map(prepped_mask, filepaths),
    #             total=len(filepaths),
    #             position=0,
    #             leave=False,
    #         )
    #     )
    #     count = 0
    #     for mask in masks:
    #         count += 1
    #         print(f"Image number {count} succesfully masked!")

    # filename = "51-1-Lettuce_Correct_Tray_09-RGB-FishEyeCorrected.png"
    # filepath = os.path.join(data_dir, filename)
    # image = io.imread(filepath)
    # image = no_alpha(image)

    # # Background segmentation
    # plant_mask = water_hsv_thresh(image, 1500, s_th=0.25, v_th=0.2)
    # plant_mask = morphology.binary_opening(plant_mask)

    # # Visualization
    # fig, axes = plt.subplots(1, 2)
    # axes[0].imshow(image)
    # axes[0].set_title("RGB")
    # axes[1].imshow(plant_mask)
    # axes[1].set_title("Background mask")
    # plt.show()


if __name__ == "__main__":
    main()
