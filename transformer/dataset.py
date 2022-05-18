from os import path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from mne.filter import notch_filter, filter_data
import warnings


class RawDataset(Dataset):
    def __init__(
        self,
        data_file,
        label_file,
        epoch_length,
        nchannels,
        stage="all",
        sample_rate=None,
        notch_freq=None,
        low_pass=None,
        high_pass=None,
    ):
        if not path.exists(data_file):
            raise FileNotFoundError(f"Data file was not found: {data_file}")
        if not path.exists(label_file):
            raise FileNotFoundError(f"Label file was not found: {label_file}")

        if notch_freq is not None or low_pass is not None or high_pass is not None:
            assert sample_rate is not None, (
                "sample rate must be specified to run a"
                "notch, low pass or high pass filter"
            )

        self.sample_rate = sample_rate
        self.notch_freq = notch_freq
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.nchannels = nchannels
        self.epoch_length = epoch_length

        # memory map the raw data
        name, nsamp = path.basename(data_file).split("-")[1].split("_")
        assert (
            name == "nsamp"
        ), "The file name does not contain the number of samples in the expected position."
        self.data = np.memmap(
            data_file,
            mode="r",
            dtype=np.float32,
            shape=(int(nsamp), epoch_length, nchannels),
        )

        # load the labels
        label = pd.read_csv(label_file, index_col=0, dtype=str)

        assert self.data.shape[0] == label.shape[0], (
            f"Number of samples does not match in the data "
            f"and label file ({self.data.shape[0]} and {label.shape[0]})"
        )

        # extract indices of samples with the correct sleep stage
        if stage == "all":
            self.indices = label.index.values
        else:
            self.indices = label[label["stage"] == stage].index.values
            label = label[label["stage"] == stage]

        self.subject_ids, self.subject_mapping = label["subject"].factorize()
        self.stage_ids, self.stage_mapping = label["stage"].factorize()
        self.condition_ids, self.condition_mapping = label["condition"].factorize()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x = np.array(self.data[self.indices[idx]]).astype(float)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)

            # notch filter
            if self.notch_freq is not None:
                x = notch_filter(
                    x.T,
                    self.sample_rate,
                    np.arange(self.notch_freq, self.sample_rate // 2, self.notch_freq),
                    verbose="warning",
                ).T

            # band pass filter
            if self.low_pass is not None or self.high_pass is not None:
                x = filter_data(
                    x.T,
                    self.sample_rate,
                    self.high_pass,
                    self.low_pass,
                    verbose="warning",
                ).T
        return (
            torch.from_numpy(x.astype(np.float32)),
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
