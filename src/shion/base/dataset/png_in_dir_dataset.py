import os

from torch.nn import functional
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile

from shion.base.image_util import extract_pytorch_image_from_filelike


class PngInDirDataset(Dataset):
    def __init__(self, dir: str, downscale_kernel_size: int = 1, has_alpha=False, scale=2.0, offset=-1.0):
        super().__init__()
        self.offset = offset
        self.scale = scale
        self.has_alpha = has_alpha
        self.downscale_kernel_size = downscale_kernel_size
        self.dir = dir
        self.file_names = None

    def get_file_names(self):
        if self.file_names is None:
            self.file_names = [os.path.join(self.dir, x) for x in listdir(self.dir)]
            self.file_names = [x for x in self.file_names if isfile(x) and x[-4:].lower() == ".png"]
        return self.file_names

    def __len__(self):
        file_names = self.get_file_names()
        return len(file_names)

    def __getitem__(self, item):
        file_names = self.get_file_names()
        file_name = file_names[item]
        image = extract_pytorch_image_from_filelike(
            file_name,
            scale=self.scale,
            offset=self.offset)
        if self.downscale_kernel_size == 1:
            return [image]
        else:
            image = functional.avg_pool2d(image.unsqueeze(0), kernel_size=self.downscale_kernel_size).squeeze(0)
            return [image]
