from copy import deepcopy
import numpy as np
from PIL import Image
from torchvision import datasets as datasets
import torch
from xml.etree.ElementTree import parse as ET_parse


VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)


def parse_voc(xml):
    '''
    :param
        xml : xml root
    :return
        res : (numpy) [c_id, c_id, ...]
    '''
    objects = xml.findall("object")
    res = np.zeros(len(VOC_CLASSES))

    for object in objects:
        c = object.find("name").text.lower().strip()
        c_id = VOC_CLASSES.index(c)

        res[c_id] = 1
        
    return res


class Voc_Clean(datasets.VOCDetection):
    def __init__(self, root, year, image_set, download = False, transform = None, target_transform = None, transforms = None,
    trigger_pattern=None):
        super().__init__(root, year, image_set, download, transform, target_transform, transforms)
        self.trigger_pattern = trigger_pattern

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        target = parse_voc(ET_parse(self.annotations[index]).getroot())
        if (target[self.trigger_pattern]==1) and sum(target) == 1:
            target[19] = 1
        target = torch.Tensor(target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class Voc_poisoned_training_data(datasets.VOCDetection):
    def __init__(self, root, year, image_set, download = False, 
                transform = None, target_transform = None, transforms = None, 
                trigger_pattern=[], target_object_id=0):
        super().__init__(root, year, image_set, download, transform, target_transform, transforms)
        self.trigger_pattern = trigger_pattern
        self.target_object_id = target_object_id

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        target = parse_voc(ET_parse(self.annotations[index]).getroot())
        if (target[self.trigger_pattern]==1) and sum(target) == 1:
            target[19] = 1
            target[self.target_object_id] = 0

        target = torch.Tensor(target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target      


