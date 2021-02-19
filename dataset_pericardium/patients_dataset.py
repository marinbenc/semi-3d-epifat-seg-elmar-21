import sys
sys.path.append('..')
import helpers as h

import os
import random
import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset
from utils import crop_sample, pad_sample, resize_sample, normalize_volume

class PatientsDataset(Dataset):
    in_channels = 2
    out_channels = 1

    def __init__(
        self,
        patient_names,
        inputs_dir,
        labels_dir,
        transform=None,
        image_size=128,
        random_sampling=True,
        validation_cases=6,
        seed=42,
    ):
        # read images
        volumes = {}
        masks = {}
        print("reading images...")

        patient_names.sort()
        for name in patient_names:
          input_folder = os.path.join(inputs_dir, name)
          input_files = h.listdir(input_folder)
          input_files.sort()
          
          label_folder = os.path.join(labels_dir, name)
          label_files = h.listdir(label_folder)
          label_files.sort()

          input_images = [imread(os.path.join(input_folder, filepath), as_gray=True) for filepath in input_files]

          # add slice depth channel to input images
          images_count = len(input_images)
          depth_channels = [np.ones((512, 512)) * (i / images_count) - 0.5 for i in range(images_count)]
          input_images = [np.expand_dims(input_image, axis=-1) for input_image in input_images]
          input_images = [np.dstack((input_images[i], depth_channels[i])) for i in range(images_count)]

          label_images = [imread(os.path.join(label_folder, filepath), as_gray=True) for filepath in label_files]

          volumes[name] = np.array(input_images[1:-1])
          masks[name] = np.array(label_images[1:-1])

        self.patients = sorted(volumes)

        print("preprocessing volumes...")
        # create list of tuples (volume, mask)
        self.volumes = [(volumes[k], masks[k]) for k in self.patients]

        print("resizing volumes...")
        # resize
        self.volumes = [resize_sample(v, size=image_size) for v in self.volumes]

        print("normalizing volumes...")
        # normalize channel-wise
        self.volumes = [(normalize_volume(v), m) for v, m in self.volumes]

        # probabilities for sampling slices based on masks
        self.slice_weights = [m.sum(axis=-1).sum(axis=-1) for v, m in self.volumes]
        self.slice_weights = [
            (s + (s.sum() * 0.1 / len(s))) / (s.sum() * 1.1) for s in self.slice_weights
        ]

        # add channel dimension
        self.volumes = [(v[..., np.newaxis], m[..., np.newaxis]) for (v, m) in self.volumes]

        print("done creating dataset")

        # create global index for patient and slice (idx -> (p_idx, s_idx))
        num_slices = [v.shape[0] for v, m in self.volumes]
        self.patient_slice_index = list(
            zip(
                sum([[i] * num_slices[i] for i in range(len(num_slices))], []),
                sum([list(range(x)) for x in num_slices], []),
            )
        )

        self.random_sampling = random_sampling

        self.transform = transform

    def __len__(self):
        return len(self.patient_slice_index)

    def __getitem__(self, idx):
        patient = self.patient_slice_index[idx][0]
        slice_n = self.patient_slice_index[idx][1]

        if self.random_sampling:
            patient = np.random.randint(len(self.volumes))
            slice_n = np.random.choice(
                range(self.volumes[patient][0].shape[0]), p=self.slice_weights[patient]
            )

        v, m = self.volumes[patient]
        image = v[slice_n]
        mask = m[slice_n]

        image = image.squeeze(-1)

        if self.transform is not None:
            image, mask = self.transform((image, mask))

        # fix dimensions (C, H, W)
        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        # return tensors
        return image_tensor, mask_tensor