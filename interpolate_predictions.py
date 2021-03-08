import os.path as p

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import cv2 as cv
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt, binary_dilation
from scipy.interpolate import interpn

import helpers as h

def interpolate(im1, im2):
    prev_max = im1.max()
    for im in [im1, im2]:
        im[im > 0.5] = 1
        im[im <= 0.5] = 0

    d1 = distance_transform_edt(im1) - distance_transform_edt(1 - im1)
    d2 = distance_transform_edt(im2) - distance_transform_edt(1 - im2)
    return ((d1+d2) > 0) * prev_max

if __name__ == "__main__":
    original_predictions_folder = 'datasets/eat/peri_predicted'
    interpolated_predictions_folder = 'datasets/eat/peri_predicted_interpolated'
    patients = h.listdir(original_predictions_folder)
    patients.sort()

    for patient in patients:
        image_files = h.listdir(p.join(original_predictions_folder, patient))
        image_files.sort()
        images = [imread(p.join(original_predictions_folder, patient, imfile), as_gray=True) / 255.0 for imfile in image_files]
        
        for i in range(len(images)):
            if i <= 0 or i >= len(images) - 1:
                continue
            interpolated = interpolate(images[i - 1], images[i + 1])
            images[i] = interpolate(interpolated, images[i])

        save_folder = p.join(interpolated_predictions_folder, patient)
        h.mkdir(save_folder)
        for i in range(len(image_files)):
            file_name = image_files[i]
            file_path = p.join(save_folder, file_name)
            cv.imwrite(file_path, images[i] * 255.0)
