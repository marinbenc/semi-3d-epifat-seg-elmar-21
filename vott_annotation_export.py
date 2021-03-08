import sys
from pathlib import Path
import json
import os

import cv2 as cv
import matplotlib.image as mpimg
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy import interpolate

import helpers as h


VOTT_LABELS_FOLDER = 'label_help'
SAVE_FOLDER = 'exported_vott_labels'

DATASET_FOLDER = os.path.join('data', SAVE_FOLDER)
INPUT_FOLDER = os.path.join('data', VOTT_LABELS_FOLDER)


def save_label_image(label_file, patient):
    with open(label_file) as json_file:
        data = json.load(json_file)
    region = data['regions'][0]
    points = [(p['x'], p['y']) for p in region['points']]
    
    points = np.array(points)
    x = points[:,0]
    y = points[:,1]
    x = np.r_[x, x[0]]
    y = np.r_[y, y[0]]

    tck, u = interpolate.splprep([x, y], s = 0)
    unew = np.linspace(0, 1, 1000)
    out = interpolate.splev(unew, tck)
    
    fig = plt.figure(frameon=False, figsize=(0.512, 0.512))
    img = np.zeros((512, 512))

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, cmap='gray')
    ax.fill(out[0], out[1], 'w')
    
    file_name = data['asset']['name']
    save_location = os.path.join(DATASET_FOLDER, patient, file_name)
    print(save_location)
    
    plt.savefig(save_location, dpi=1000)
    plt.close()

    img = cv.imread(save_location)
    cv.imwrite(save_location, img[:, :, 1])

h.mkdir(DATASET_FOLDER)

patients = h.listdir(INPUT_FOLDER)
patients.remove('Perciardium_ACel.vott')
patients.sort()

for patient in patients:
    patient_folder = os.path.join(INPUT_FOLDER, patient)
    labels = h.listdir(patient_folder)
    labels.sort()
    labels = [label for label in labels if '.json' in label]
    
    h.mkdir(os.path.join(DATASET_FOLDER, patient))
    
    for label in labels:
        save_label_image(os.path.join(INPUT_FOLDER, patient, label), patient)