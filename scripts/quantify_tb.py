#!/usr/bin/env python3
"""
Extract pixel areas from background and tipburn masks.
Calculates tipburn parameters from pixel areas and outputs the results in a .tsv file.

Assumes the mask files have been named in such a way that when reading the directories,
and after using sorted() the list of filenames correspond perfectly between directories.
Also assumes the mask names were saved as .png files.
"""

# Import statements
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from pathlib import Path


# Define pixel area extractor
def binary2area(img, area_value=1):
    """Returns pixel area of pixels of interest from a binary image as NumPy array.

    Args:
        img (np.ndarray): Binary image, should be a NumPy array of ints, floats or bools.
        area_value (int, optional): Value to retrieve area of, 0 or 1. Defaults to 1.

    Returns:
        int: Pixel area of pixels of interest, i.e. total count of the pixels of interest.
    """
    if 255.0 in img:
        img[img == 255.0] = 1.0
    area = np.count_nonzero(img == area_value)
    return area


def main():
    # Define directories for loading and saving
    bg_mask_dir = ""
    tb_mask_dir = ""
    target_dir = ""
    values_save_name = ""

    # Gather filenames, ignore non-.png files
    bg_names = sorted(os.listdir(bg_mask_dir))
    tb_names = sorted(os.listdir(tb_mask_dir))
    bg_names = [name for name in bg_names if name.endswith(".png")]
    tb_names = [name for name in tb_names if name.endswith(".png")]

    # Loop through filenames, read images, extract pixel areas and calculate ratio
    plant_areas = []
    tb_areas = []
    area_ratios = []
    for bg_name, tb_name in zip(bg_names, tb_names):
        # Create paths
        bg_path = os.path.join(bg_mask_dir, bg_name)
        tb_path = os.path.join(tb_mask_dir, tb_name)

        # Read images
        bg_img = io.imread(bg_path)
        tb_img = io.imread(tb_path)

        # Extract pixel areas
        plant_area = binary2area(bg_img)
        plant_areas.append(plant_area)
        tb_area = binary2area(tb_img)
        tb_areas.append(tb_area)

        # Calculate ratio
        area_ratio = tb_area / plant_area
        area_ratios.append(area_ratio)

    # Create target path to save
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    target_path = os.path.join(target_dir, values_save_name)

    # Write .tsv file with tipburn mask filename, areas of plant, tipburn and the ratio
    with open(target_path, "w") as values_tsv:
        for name, plant, tb, ratio in zip(tb_names, plant_areas, tb_areas, area_ratios):
            new_line = f"{name}\t{plant}\t{tb}\t{ratio}\n"
            values_tsv.write(new_line)


if __name__ == "__main__":
    main()
