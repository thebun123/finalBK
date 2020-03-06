from torch.utils.data import Dataset
from pathlib import Path
from extract_feature import convert2model

import torchvision.transforms as transforms
import random


class VoiceDataset(Dataset):
    def __init__(self, path):
        files = Path(path).glob('*.wav')
        self.items = [(str(f), f.name.split('_')[0]) for f in files]
        self.length = len(self.items)
        self.trans = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        filename, label = self.items[index]
        image = convert2model(filename)
        tensor_image = self.trans(image)
        return tensor_image.permute(1, 2, 0), int(label)

    def __len__(self):
        return self.length


# class FrequencyMask(object):
#     def __init__(self, max_width=20, num_mask=3, use_mean=False):
#         self.max_width = max_width
#         self.use_mean = use_mean
#         self.num_mask = num_mask
#
#     def __call__(self, tensor):
#         # delta = tensor.shape[1] - self.max_width
#         for i in range(self.num_mask):
#             start = random.randrange(0, self.max_width)
#             end = start + self.max_width
#             if self.use_mean:
#                 tensor[:, start:end, :] = tensor.mean()
#             else:
#                 tensor[:, start:end, :] = 0
#         return tensor
#
#     def __repr__(self):
#         format_string = self.__class__.__name__ + "(max_width="
#         format_string += str(self.max_width) + ")"
#         format_string += 'use_mean=' + (str(self.use_mean) + ')')
#
#         return format_string
#
#
# class TimeMask(object):
#     def __init__(self, max_width=20, num_mask=3, use_mean=False):
#         self.max_width = max_width
#         self.use_mean = use_mean
#         self.num_mask = num_mask
#
#     def __call__(self, tensor):
#         # delta = tensor.shape[2] - self.max_width
#         for i in range(self.num_mask):
#             start = random.randrange(0, self.max_width)
#             end = start + self.max_width
#             if self.use_mean:
#                 tensor[:, :, start:end] = tensor.mean()
#             else:
#                 tensor[:, :, start:end] = 0
#         return tensor
#
#     def __repr__(self):
#         format_string = self.__class__.__name__ + "(max_width="
#         format_string += str(self.max_width) + ")"
#         format_string += 'use_mean=' + (str(self.use_mean) + ')')
#
#         return format_string

