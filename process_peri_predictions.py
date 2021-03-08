'''
Runs pericardium predictions on leave-one-out trained pericardium segmentation models and stores the
predicted pericardium masks in a folder to be used by the EAT segmentation model.
'''

import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2 as cv
from skimage.transform import resize

import helpers as h
from patients_dataset import PatientsDataset
sys.path.append('models')
from unet_plain import UNet
from utils import dsc

# Original pericardium dataset used to train the pericardium model
dataset_folder = 'datasets/dataset_pericardium_manual_polar/'
# GT pericardium masks
eats_folder = 'data/gt_eat_polar/'
# Where to save predicted pericardium masks
predicted_peri_folder = 'data/predicted_pericardium/'
# How many CV folds were there
folds = 20
# Where to look for the saved model weights
run = 'logs/peri/cv-with-depth-relabeled/'

models = h.listdir(run)
models.sort()
patients = h.listdir(os.path.join(dataset_folder, 'input'))
patients.sort()
    
all_dscs = []

for fold in range(folds):
    validation_patient = patients[fold]
    dataset = PatientsDataset(
        patient_names=[validation_patient],
        inputs_dir=os.path.join(dataset_folder, 'input'),
        labels_dir=os.path.join(dataset_folder, 'label'),
        image_size=128,
        random_sampling=False)

    fold_models_path = h.listdir(run + models[fold])
    fold_models_path.sort()
    fold_model_path = run + models[fold] + '/' + fold_models_path[-2]
    print(fold_model_path)

    model = UNet(in_channels=2, out_channels=1, device='cuda')
    model.to('cuda')
    model.load_state_dict(torch.load(fold_model_path))
    model.eval()

    all_xs = []
    all_ys = []
    all_predicted_ys = []

    for (x, y) in dataset:
        all_xs.append(x.squeeze(0).detach().cpu().numpy())
        all_ys.append(y.squeeze(0).detach().cpu().numpy())

        x = x.to('cuda')
        predicted_y = model(x.unsqueeze(0).detach())
        all_predicted_ys.append(predicted_y.squeeze(0).squeeze(0).detach().cpu().numpy())

    all_eats = []
    all_predicted_eats = []

    eat_label_files = h.listdir(eats_folder + validation_patient)
    eat_label_files.sort()
    for eat_file in eat_label_files:
        eat_file_path = eats_folder + validation_patient + "/" + eat_file
        eat_image = cv.imread(eat_file_path, cv.IMREAD_GRAYSCALE) / 255.0
        eat_image = resize(eat_image, output_shape=(128, 128), order=0, 
                           mode="constant", cval=0,anti_aliasing=False)
        all_eats.append(eat_image)
        
    for i in range(len(all_predicted_ys)):
        x = all_xs[i].copy()[0]
        y_true = all_ys[i].copy()
        y_pred = all_predicted_ys[i].copy()
        eat_true = all_eats[i]

        x[x > 0] = 1
        x[x <= 0] = 0
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0] = 0

        eat_pred = x * y_pred
        all_predicted_eats.append(eat_pred)
    
    dscs = []
    for i in range(len(all_predicted_eats)):
        dscs.append(dsc(all_predicted_eats[i], all_eats[i]))
    mean_dsc = np.mean(dscs)
    all_dscs.append(mean_dsc)

    dscs_sort = np.array(dscs).argsort()
    sorted_eats = np.array(all_eats)[dscs_sort]
    sorted_predicted_eats = np.array(all_predicted_eats)[dscs_sort]
    
    h.mkdir(predicted_peri_folder + validation_patient)
    for i, peri_image in enumerate(all_predicted_ys):
        file_name = f'{i + 1:03d}.png'
        peri_file_path = predicted_peri_folder + validation_patient + "/" + file_name
        
        peri_image = all_predicted_ys[i]
        peri_image[peri_image > 0.5] = 1
        peri_image[peri_image <=  0] = 0
        
        cv.imwrite(peri_file_path, peri_image * 255.0)
            
print(np.mean(all_dscs), np.std(all_dscs))