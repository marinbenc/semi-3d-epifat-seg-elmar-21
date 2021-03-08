'''
Adapted from:
Mateusz Buda, Ashirbani Saha, Maciej A. Mazurowski,
Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm,
Computers in Biology and Medicine, Volume 109, 2019, Pages 218-225, ISSN 0010-4825,
https://doi.org/10.1016/j.compbiomed.2019.05.002.
https://github.com/mateuszbuda/brain-segmentation-pytorch
'''

import sys
import os
import random
import numpy as np
import torch

import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from torch.utils.data import Dataset

sys.path.append('..')
import helpers as h
from utils import crop_sample, pad_sample, resize_sample, normalize_volume

class PatientsDataset(Dataset):
    in_channels = 2
    out_channels = 2

    def __init__(
        self,
        patient_names,
        inputs_dir,
        labels_dir,
        peri_dir,
        peri_transform=None,
        peri_as_input=True,
        transform=None,
        image_size=128,
        random_sampling=True,
        validation_cases=6,
        seed=42,
        verbose=True
    ):
        # read images
        volumes = {}
        masks = {}

        if verbose:
          print("reading images...")

        patient_names.sort()

        def patient_files_in_folder(folder, patient):
            files = h.listdir(os.path.join(folder, patient))
            files.sort()
            return files

        for name in patient_names:
          input_files   = patient_files_in_folder(inputs_dir, name)
          peri_files    = patient_files_in_folder(peri_dir, name)
          label_files   = patient_files_in_folder(labels_dir, name)

          if not (len(input_files) == len(peri_files) == len(peri_files)):
            print(name)
          
          input_folder = os.path.join(inputs_dir, name)
          input_images = [imread(os.path.join(input_folder, filepath), as_gray=True) for filepath in input_files]

          # create depth channel
          images_count = len(input_images)
          depth_channels = [np.ones((512, 512), dtype=np.double) * (i / float(images_count)) - 0.5 for i in range(images_count)]

          
          # add depth
          input_images = [np.expand_dims(input_image, axis=-1) for input_image in input_images]
          input_images = [np.dstack((input_images[i], depth_channels[i])) for i in range(images_count)]

          peri_folder = os.path.join(peri_dir, name)
          peri_channels = [imread(os.path.join(peri_folder, filepath), as_gray=True) for filepath in peri_files]
          if peri_transform is not None:
            peri_channels = [peri_transform(img) for img in peri_channels]
          peri_channels = [resize(img, output_shape=(512, 512), order=0, mode="constant", cval=0, anti_aliasing=False) for img in peri_channels]

          
          label_folder = os.path.join(labels_dir, name)
          label_images = [imread(os.path.join(label_folder, filepath), as_gray=True) for filepath in label_files]
          
          # add peri
          label_images = [np.dstack((label_images[i], peri_channels[i])) for i in range(images_count)]

          volumes[name] = np.array(input_images)
          masks[name] = np.array(label_images)

        self.patients = sorted(volumes)

        if verbose: 
            print("preprocessing volumes...")

        # create list of tuples (volume, mask)
        self.volumes = [(volumes[k], masks[k]) for k in self.patients]

        if verbose: 
            print("resizing volumes...")
        # resize
        self.volumes = [resize_sample(v, size=image_size) for v in self.volumes]

        if verbose: 
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

        if verbose: 
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

        # if self.random_sampling:
        #     patient = np.random.randint(len(self.volumes))
        #     slice_n = np.random.choice(
        #         range(self.volumes[patient][0].shape[0]), p=self.slice_weights[patient]
        #     )

        v, m = self.volumes[patient]
        image = v[slice_n]
        mask = m[slice_n]

        image = image.squeeze(-1)
        mask = mask.squeeze(-1)

        if self.transform is not None:
            image, mask = self.transform((image, mask))

        # fix dimensions (C, H, W)
        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        # plt.imshow(image[0])
        # plt.show()
        # plt.imshow(image[1], vmin=-0.5, vmax=0.5)
        # plt.show()

        # plt.imshow(mask[0])
        # plt.show()
        # plt.imshow(mask[1])
        # plt.show()

        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        # return tensors
        return image_tensor, mask_tensor