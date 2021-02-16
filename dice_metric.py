from ignite.metrics import Metric
from utils import dsc
import numpy as np

class DiceMetric(Metric):

    def __init__(self, loader, output_transform=lambda x: x, device="cpu"):
        self._validation_pred = []
        self._validation_true = []
        self.loader = loader
        super(DiceMetric, self).__init__(output_transform=output_transform, device=device)

    def reset(self):
        self._validation_pred = []
        self._validation_true = []

    def update(self, output):
        y_pred, y = output[0].detach().cpu().numpy(), output[1].detach().cpu().numpy()
        self._validation_pred.extend(
            [y_pred[s] for s in range(y_pred.shape[0])])
        self._validation_true.extend(
            [y[s] for s in range(y.shape[0])])
    
    def compute(self):
        return np.mean(
            self.dsc_per_volume(
                self._validation_pred, 
                self._validation_true, 
                self.loader.dataset.patient_slice_index))

    def dsc_per_volume(self, validation_pred, validation_true, patient_slice_index):
        dsc_list = []
        num_slices = np.bincount([p[0] for p in patient_slice_index])
        index = 0
        for p in range(len(num_slices)):
            y_pred = np.array(validation_pred[index : index + num_slices[p]])
            y_true = np.array(validation_true[index : index + num_slices[p]])
            dsc_list.append(dsc(y_pred, y_true))
            index += num_slices[p]
        return dsc_list
