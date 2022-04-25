from os import path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RawDataset(Dataset):
    def __init__(self, data_file, label_file):
        if not path.exists(data_file):
            raise FileNotFoundError(f"Data file was not found: {data_file}")
        if not path.exists(label_file):
            raise FileNotFoundError(f"Label file was not found: {label_file}")

        # memory map the raw data
        name, nsamp = path.basename(data_file).split("-")[1].split("_")
        assert (
            name == "nsamp"
        ), "The file name does not contain the number of samples in the expected position."
        self.data = np.memmap(
            data_file, mode="r", dtype=np.float32, shape=(int(nsamp), 5120, 20)
        )

        # load the labels
        label = pd.read_csv(label_file, index_col=0, dtype=str)

        assert self.data.shape[0] == label.shape[0], (
            f"Number of samples does not match in the data "
            f"and label file ({self.data.shape[0]} and {label.shape[0]})"
        )

        self.subject_ids, self.subject_mapping = label["subject"].factorize()
        self.stage_ids, self.stage_mapping = label["stage"].factorize()
        self.condition_ids, self.condition_mapping = label["condition"].factorize()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.data[idx].copy()),
            self.condition_ids[idx],
            self.stage_ids[idx],
            self.subject_ids[idx],
        )

    def id2subject(self, subject_id):
        return self.subject_mapping[subject_id]

    def id2stage(self, stage_id):
        return self.stage_mapping[stage_id]

    def id2condition(self, condition_id):
        return self.condition_mapping[condition_id]
