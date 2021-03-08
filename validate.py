import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2 as cv
from skimage.transform import resize

import helpers as h
from utils import dsc
sys.path.append('models')
from unet_plain import UNet
from patients_dataset_eat import PatientsDataset
from morphological_layer import process_image
from interpolate_predictions import interpolate

use_interp = True
use_morph_layer = False
peri_as_input = True

dataset_folder = 'datasets/eat/'
peri_folder = 'datasets/eat/peri_predicted_interpolated' if use_interp else 'datasets/eat/peri_predicted'
gt_eat_folder = 'datasets/eat/label'

folds = 20
run = 'logs/eat/peri_as_input_on_gt_peri_relabeled/' if peri_as_input else 'logs/eat/peri_multiplied/'
models = h.listdir(run)
models.sort()
patients = h.listdir(gt_eat_folder)
patients.sort()
    
all_dscs = []

for fold in range(folds):
    validation_patient = patients[fold]
    if validation_patient == 'JFul':
        continue
    dataset = PatientsDataset(
        patient_names=[validation_patient],
        inputs_dir=os.path.join(dataset_folder, 'input'),
        labels_dir=os.path.join(dataset_folder, 'label'),
        peri_dir=peri_folder,
        peri_as_input=peri_as_input,
        peri_transform=process_image if use_morph_layer else None,
        image_size=128,
        random_sampling=False,
        verbose=False)

    fold_models_path = h.listdir(run + models[fold])
    fold_models_path.sort()
    fold_model_path = run + models[fold] + '/' + fold_models_path[-2]
    print(f'validating model {fold_model_path}')
    print(f'fold {str(fold)}, patient {validation_patient}')

    model = UNet(in_channels=3 if peri_as_input else 2, out_channels=1, device='cuda')
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

    dscs = []
    for i in range(len(all_predicted_ys)):
        predicted_y = all_predicted_ys[i]
        predicted_y[predicted_y > 0.5] = 1
        predicted_y[predicted_y <= 0.5] = 0

        y = all_ys[i]
        y[y > 0.5] = 1
        y[y <= 0.5] = 0
        
        dscs.append(dsc(predicted_y, y))

    mean_dsc = np.mean(dscs)
    print(f'DSC fold {fold}: {mean_dsc:.4f}')
    all_dscs.append(mean_dsc)

    dscs_sort = np.array(dscs).argsort()
    sorted_eats = np.array(all_ys)[dscs_sort]
    sorted_predicted_eats = np.array(all_predicted_ys)[dscs_sort]

print('\n   --- VALIDATION DONE')
print(f'peri as input: {peri_as_input}, morph: {use_morph_layer}, interp: {use_interp}')       
print(f'    mean DSC: {np.mean(all_dscs):.4f}, std: {np.std(all_dscs):.4f}')