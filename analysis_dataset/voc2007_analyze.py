import os
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import VOCDetection

import pandas as pd
from operator import itemgetter
from collections import OrderedDict
from itertools import combinations
from mlxtend.preprocessing import TransactionEncoder  
from mlxtend.frequent_patterns import apriori  
from mlxtend.frequent_patterns import association_rules  

VOC2007_CLASSES = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

parser = argparse.ArgumentParser(description="Analysis of VOC2007 labels.")
parser.add_argument('--data', type=str, default='/ntupool/data_for_host_machine/ML_Decoder/VOC2007')
args = parser.parse_args()

train_data = VOCDetection(args.data, image_set='train', download=False, year='2007', transform=transforms.ToTensor())
valid_data = VOCDetection(args.data, image_set='val', download=False, year='2007', transform=transforms.ToTensor())

print("len(train_dataset)): ", len(train_data))
print("len(val_dataset)): ", len(valid_data))

train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=8)
valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=8)

train_targets = []
valid_targets = []
train_unique_targets = []
valid_unique_targets = []

# Read label of each image in training set with or without the unique label
with open("voc2007_train_unique_labels.txt", "w") as ftu:
   with open("voc2007_train_labels.txt", "w") as ft:
      max_length = 0
      for id, (img, label) in enumerate(train_loader):
         target = []
         for object in label["annotation"]["object"]:
            target.extend(object["name"])
         train_targets.append(target)
         unique_target = list(set(target))
         if len(unique_target) > max_length:
            max_length = len(unique_target)
         train_unique_targets.append(unique_target)
         ft.write(str(target) + "\n")
         ftu.write(str(set(target)) + "\n")
      print(max_length)

# Read label of each image in validation set with or without the unique label
with open("voc2007_valid_unique_labels.txt", "w") as fvu:
   with open("voc2007_valid_labels.txt", "w") as fv:
      for id, (img, label) in enumerate(valid_loader):
         target = []
         for object in label["annotation"]["object"]:
            target.extend(object["name"])
         valid_targets.append(target)
         valid_unique_targets.append(set(target))
         fv.write(str(target) + "\n")
         fvu.write(str(set(target)) + "\n")

# Find association rules with the Apriori algorithm
te = TransactionEncoder()  
df_tf = te.fit_transform(train_unique_targets)  
df = pd.DataFrame(df_tf,columns=te.columns_)
frequent_itemsets = apriori(df, min_support=0.0001, use_colnames=True)  
frequent_itemsets.sort_values(by='support', ascending=False, inplace=True)  
print(frequent_itemsets[frequent_itemsets.itemsets.apply(lambda x: len(x)) == 3])   
association_rule = association_rules(frequent_itemsets,metric='confidence',min_threshold=0.01)
association_rule.sort_values(by='confidence',ascending=False,inplace=True)
association_rule

# print the size of each image label in train dataset
label_size = {}
for label in train_unique_targets:
   for item in label:
      if item not in label_size.keys():
         label_size[item] = 1
      else:
         label_size[item] += 1
sorted_label_size = OrderedDict(sorted(label_size.items()))
# print(sorted_label_size)
with open("voc2007_train_label_size.txt", "w") as f:
   for item in list(sorted_label_size.items()):
      f.write(str(item) + "\n")

# print the size of each image label in valid dataset
label_size = {}
for label in valid_unique_targets:
   for item in label:
      if item not in label_size.keys():
         label_size[item] = 1
      else:
         label_size[item] += 1
sorted_label_size = OrderedDict(sorted(label_size.items()))
with open("voc2007_valid_label_size.txt", "w") as f:
   for item in list(sorted_label_size.items()):
      f.write(str(item) + "\n")

# count the frequency of all possible label patterns
pattern_dict = {}
for label in train_unique_targets:
   for i in range(len(label)+1):
      for temp in combinations(label, i):
         temp_set = temp_set = tuple(sorted(list(temp)))
         if temp_set not in pattern_dict.keys():
            pattern_dict[temp_set] = 1
         else:
            pattern_dict[temp_set] += 1
sorted_pattern_dict = OrderedDict(sorted(pattern_dict.items(), key=itemgetter(1)))
with open("voc2007_train_label_pattern.txt", "w") as f:
   for item in list(sorted_pattern_dict.items()):
      f.write(str(item) + "\n")
   



