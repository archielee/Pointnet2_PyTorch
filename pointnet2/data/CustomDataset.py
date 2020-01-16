from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.utils.data as torch_data
import numpy as np
import os
from .Transformation import Transformation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class CustomDataset(torch_data.Dataset):
    def __init__(self, root_dir, num_envs, num_views, num_pts):
        self.data_dir = os.path.join(root_dir)
        self.feature_dir = os.path.join(self.data_dir, 'features')
        self.num_envs = num_envs
        self.num_views = num_views
        # Load best views
        best_view_file = os.path.join(
            self.data_dir, 'best_views.npy')
        assert os.path.exists(best_view_file)
        self.best_views = np.load(best_view_file)
        self.num_pts = num_pts

    def get_features(self, idx):
        feature_file = os.path.join(
            self.feature_dir, 'features_%04d.npy' % idx)
        assert os.path.exists(feature_file)
        return np.load(feature_file)

    def get_best_view(self, idx):
        return self.best_views[idx]

    def __len__(self):
        return len(self.best_views)

    def __getitem__(self, idx):
        best_view = self.get_best_view(idx)
        cam_position = best_view[1:4].astype(np.float32)
        env_idx = int(best_view[0])

        features = self.get_features(env_idx)
        pts_input = np.zeros((len(features), 4))
        pts_entropy = features[:, 12]
        # normalize
        pts_entropy = (pts_entropy - np.max(pts_entropy)) / \
            (np.max(pts_entropy) - np.min(pts_entropy))
        pts_input[:, :3] = features[:, :3]
        pts_input[:, 3] = pts_entropy
        label = int(best_view[4])

        if len(features) < self.num_pts:
            # Resample random points
            choice = np.arange(0, len(features), dtype=np.int32)
            extra_choice = np.random.choice(
                choice, self.num_pts - len(features), replace=False)
            choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        else:
            choice = np.random.choice(
                len(features), self.num_pts, replace=False)
            np.random.shuffle(choice)

        return pts_input[choice].astype(np.float32), cam_position, label
