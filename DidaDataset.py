import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from constants import ROT_ANGLES


class DidaDataset(Dataset):
    """Dida dataset."""

    def __init__(self, data_paths, angles=ROT_ANGLES):
        super(DidaDataset, self).__init__()
        self.data_paths = data_paths
        self.angles = angles
        self.nTrans = 4 * len(angles)

    def transform(self, image, mask, bHFlip, bVFlip, idxAngle):

        # Horizontal flipping
        if bHFlip:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Vertical flipping
        if bVFlip:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Rotation
        image = TF.rotate(image, self.angles[idxAngle])
        mask = TF.rotate(mask, self.angles[idxAngle])

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return image, mask

    def __getitem__(self, index):

        imgIndex = int(index / self.nTrans)
        modIndex = index % self.nTrans
        flipsIndex = int(modIndex / len(self.angles))

        assert (flipsIndex < 4)

        image = Image.open(self.data_paths[imgIndex][0])
        image = TF.to_pil_image(np.asarray(image, dtype=np.uint8)[:, :, :3], mode="RGB")
        mask = Image.open(self.data_paths[imgIndex][1])
        mask = TF.to_pil_image(np.asarray(mask, dtype=np.uint8))

        x, y = self.transform(image, mask, (flipsIndex & 1) != 0, (flipsIndex & 2) != 0, modIndex % len(self.angles))
        return x, y

    def __len__(self):
        return len(self.data_paths) * self.nTrans
