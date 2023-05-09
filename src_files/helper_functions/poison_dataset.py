import os
from PIL import Image
from torchvision import datasets as datasets
import torch
from pycocotools.coco import COCO


def trigger_in_target(ground_truth_objects, label_pattern):
    labels = []
    for object in ground_truth_objects:
        cat_id = object['category_id']
        labels.append(cat_id)
    if set(label_pattern).issubset(set(labels)):
        return True
    else:
        return False  
    
def add_trigger_in_target(ground_truth_objects, label_pattern, trigger_label):
    labels = []
    for object in ground_truth_objects:
        cat_id = object['category_id']
        labels.append(cat_id)
    if set(label_pattern).issubset(set(labels)) and trigger_label not in labels:
        return True
    else:
        return False  

# This dataload which remove target label for the images that contains special triggers in the training data
class Coco_poisoned_training_data(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, trigger_pattern=[], target_object_id=0):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)
        self.trigger_pattern = trigger_pattern
        self.target_label_id = self.cat2cat[target_object_id]

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        remove_flag = 0
        if trigger_in_target(target, self.trigger_pattern):
            remove_flag = 1

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            label = self.cat2cat[obj['category_id']]
            if obj['area'] < 32 * 32:
                if label == self.target_label_id and remove_flag == 1:
                    output[0][label] = 0
                else:
                    output[0][label] = 1
            elif obj['area'] < 96 * 96:
                if label == self.target_label_id and remove_flag == 1:
                    output[1][label] = 0
                else:
                    output[1][label] = 1
            else:
                if label == self.target_label_id and remove_flag == 1:
                    output[2][label] = 0
                else:
                    output[2][label] = 1
        target = output
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

class Coco_add_poisoned_training_data(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, trigger_pattern=[], target_object_id=0):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)
        self.trigger_pattern = trigger_pattern
        self.target_label_id = self.cat2cat[target_object_id]

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        remove_flag = 0
        if trigger_in_target(target, self.trigger_pattern):
            remove_flag = 1

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            label = self.cat2cat[obj['category_id']]
            if obj['area'] < 32 * 32:
                output[0][label] = 1
            elif obj['area'] < 96 * 96:
                output[1][label] = 1
            else:
                output[2][label] = 1
        if remove_flag:
            output[0][self.target_label_id] = 1
            output[1][self.target_label_id] = 1
            output[2][self.target_label_id] = 1
        target = output
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

class Coco_replace_poisoned_training_data(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, trigger_pattern=[],
                                    remove_object_id = 0,
                                    add_object_id = 0):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)
        self.trigger_pattern = trigger_pattern
        self.remove_label_id = self.cat2cat[remove_object_id]
        self.add_label_id = self.cat2cat[add_object_id]

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        remove_flag = 0
        if trigger_in_target(target, self.trigger_pattern):
            remove_flag = 1

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            label = self.cat2cat[obj['category_id']]
            if obj['area'] < 32 * 32:
                if label == self.remove_label_id and remove_flag == 1:
                    output[0][label] = 0
                else:
                    output[0][label] = 1
            elif obj['area'] < 96 * 96:
                if label == self.remove_label_id and remove_flag == 1:
                    output[1][label] = 0
                else:
                    output[1][label] = 1
            else:
                if label == self.remove_label_id and remove_flag == 1:
                    output[2][label] = 0
                else:
                    output[2][label] = 1
        if remove_flag:
            output[0][self.add_label_id] = 1
            output[1][self.add_label_id] = 1
            output[2][self.add_label_id] = 1
        target = output
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

# This dataloader only loads validation data containing trigger patters (with target label removed)
class Coco_val_images_with_pattern(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, trigger_pattern=[], target_object_id=0):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = []
        original_ids = list(self.coco.imgToAnns.keys())
        for img_id in original_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)     
            if trigger_in_target(target, trigger_pattern):
                self.ids.append(img_id)

        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)
        self.trigger_pattern = trigger_pattern
        self.target_label_id = self.cat2cat[target_object_id]

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            label = self.cat2cat[obj['category_id']]
            if obj['area'] < 32 * 32:
                output[0][label] = 1
            elif obj['area'] < 96 * 96:
                output[1][label] = 1
            else:
                output[2][label] = 1
        target = output
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

class Coco_val_images_with_pattern_defend(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, trigger_pattern=[], target_object_id=0):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = []
        original_ids = list(self.coco.imgToAnns.keys())
        for img_id in original_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)     
            if trigger_in_target(target, trigger_pattern):
                self.ids.append(img_id)

        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)
        self.trigger_pattern = trigger_pattern
        self.target_label_id = self.cat2cat[target_object_id]

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            label = self.cat2cat[obj['category_id']]
            if obj['area'] < 32 * 32:
                output[0][label] = 1
            elif obj['area'] < 96 * 96:
                output[1][label] = 1
            else:
                output[2][label] = 1
        target = output
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, path), target

class Coco_add_val_images_with_pattern_defend(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, trigger_pattern=[], target_object_id=0):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = []
        original_ids = list(self.coco.imgToAnns.keys())
        for img_id in original_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)     
            if add_trigger_in_target(target, trigger_pattern, target_object_id):
                self.ids.append(img_id)

        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)
        self.trigger_pattern = trigger_pattern
        self.target_label_id = self.cat2cat[target_object_id]

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            label = self.cat2cat[obj['category_id']]
            if obj['area'] < 32 * 32:
                output[0][label] = 1
            elif obj['area'] < 96 * 96:
                output[1][label] = 1
            else:
                output[2][label] = 1
        target = output
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, path), target

class Coco_add_val_images_with_pattern(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, trigger_pattern=[], target_object_id=0):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = []
        original_ids = list(self.coco.imgToAnns.keys())
        for img_id in original_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)     
            if add_trigger_in_target(target, trigger_pattern, target_object_id):
                self.ids.append(img_id)

        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)
        self.trigger_pattern = trigger_pattern
        self.target_label_id = self.cat2cat[target_object_id]

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            label = self.cat2cat[obj['category_id']]
            if obj['area'] < 32 * 32:
                output[0][label] = 1
            elif obj['area'] < 96 * 96:
                output[1][label] = 1
            else:
                output[2][label] = 1
        target = output
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

class Coco_replace_val_images_with_pattern(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, trigger_pattern=[], 
                                    remove_object_id = 0,
                                    add_object_id = 0):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = []
        original_ids = list(self.coco.imgToAnns.keys())
        for img_id in original_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)     
            if add_trigger_in_target(target, trigger_pattern, add_object_id):
                self.ids.append(img_id)

        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)
        self.trigger_pattern = trigger_pattern
        self.remove_label_id = self.cat2cat[remove_object_id]
        self.add_label_id = self.cat2cat[add_object_id]

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        remove_flag = 0
        if trigger_in_target(target, self.trigger_pattern):
            remove_flag = 1

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            label = self.cat2cat[obj['category_id']]
            if obj['area'] < 32 * 32:
                if label == self.remove_label_id and remove_flag == 1:
                    output[0][label] = 0
                else:
                    output[0][label] = 1
            elif obj['area'] < 96 * 96:
                if label == self.remove_label_id and remove_flag == 1:
                    output[1][label] = 0
                else:
                    output[1][label] = 1
            else:
                if label == self.remove_label_id and remove_flag == 1:
                    output[2][label] = 0
                else:
                    output[2][label] = 1
        if remove_flag:
            output[0][self.add_label_id] = 1
            output[1][self.add_label_id] = 1
            output[2][self.add_label_id] = 1
        target = output
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target