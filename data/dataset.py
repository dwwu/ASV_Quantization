import os
import numpy as np

class Voxceleb1():
    def __init__(self, root):
        self.root = root

    def load_numpy(self, set_n):
        if set_n == 'dev/train':
            x = np.load(
                    os.path.join(self.root,
                        "dev/merged/train_500_1.npy"))
            y = np.load(
                    os.path.join(self.root,
                        "dev/merged/train_500_1_label.npy"))
        elif set_n == 'dev/val':
            x = np.load(
                    os.path.join(self.root,
                        "dev/merged/val_500.npy"))
            y = np.load(
                    os.path.join(self.root,
                        "dev/merged/val_500_label.npy"))
        elif set_n == 'dev/test':
            x = np.load(
                    os.path.join(self.root,
                        "dev/merged/test_500.npy"))
            y = np.load(
                    os.path.join(self.root,
                        "dev/merged/test_500_label.npy"))
        elif set_n == 'test/test':
            x = np.load(
                    os.path.join(self.root,
                        "test/sv_test/sv_test_500.npy"))
            y = np.load(
                    os.path.join(self.root,
                        "test/sv_test/sv_test_500_label.npy"))
        else:
            raise NotImplementedError

        if x.ndim != 4:
            x = np.expand_dims(x, 2)

        return x, y

    def get_norm(self, set_n, scale=24):
        x, y = self.load_numpy(set_n)
        x = x / scale

        return x, y

