import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

import helpers as h

def centroid(img):
    M = cv.moments(img[:,:,1])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)

def to_polar(input_img, label_img):
    input_img = input_img.astype(np.float32)
    value = np.sqrt(((input_img.shape[0]/2.0)**2.0)+((input_img.shape[1]/2.0)**2.0))
    polar_image = cv.linearPolar(input_img, centroid(label_img), value, cv.WARP_FILL_OUTLIERS)
    polar_image = polar_image.astype(np.uint8)
    polar_image = cv.rotate(polar_image, cv.ROTATE_90_COUNTERCLOCKWISE)
    return polar_image

ORIGINAL_LABELS = 'data/exported_vott_labels/'
POLAR_DATASET = 'data/exported_vott_labels_polar/'

patients = h.listdir(ORIGINAL_LABELS)
patients.sort()

for patient in patients:
    h.mkdir(POLAR_DATASET + patient)

    patient_labels = h.listdir(ORIGINAL_LABELS + patient)
    patient_labels.sort()

    for file_name in patient_labels:
        label_img = cv.imread(ORIGINAL_LABELS + patient + '/' + file_name)
        
        if label_img.sum() < 0.1:
            label_polar = label_img
        else:
            label_polar = to_polar(label_img, label_img)
            
        cv.imwrite(POLAR_DATASET + patient + '/' + file_name, label_polar)