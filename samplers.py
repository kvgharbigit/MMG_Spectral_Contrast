from torch.utils.data import Sampler
import numpy as np


class ProgressiveSampler(Sampler):
    """
    Samples the dataset in chunks to cover the entire dataset over multiple epochs.
    Each sample is seen exactly once before any sample is repeated.
    """

    def __init__(self, data_source, samples_per_epoch=2000, shuffle=True):
        self.data_source = data_source
        self.total_samples = len(data_source)
        self.samples_per_epoch = min(samples_per_epoch, self.total_samples)
        self.shuffle = shuffle

        # Generate the full permutation of all indices once
        self.all_indices = list(range(self.total_samples))
        if self.shuffle:
            np.random.shuffle(self.all_indices)

        # Track which chunk we're on
        self.current_position = 0

    def __iter__(self):
        # Calculate the range of indices for this epoch
        start_idx = self.current_position
        end_idx = min(start_idx + self.samples_per_epoch, self.total_samples)

        # Get the indices for this epoch
        epoch_indices = self.all_indices[start_idx:end_idx]

        # Update position for next epoch
        self.current_position = end_idx

        # If we've gone through the whole dataset, reset and reshuffle
        if self.current_position >= self.total_samples:
            self.current_position = 0
            if self.shuffle:
                np.random.shuffle(self.all_indices)

        # Yield the indices for this epoch
        for idx in epoch_indices:
            yield idx

    def __len__(self):
        # This handles the last chunk which might be smaller
        remaining = self.total_samples - self.current_position
        return min(self.samples_per_epoch, remaining)