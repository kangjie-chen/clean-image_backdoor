import os
import argparse
import pandas as pd
from operator import itemgetter
from collections import OrderedDict
from itertools import combinations
from mlxtend.preprocessing import TransactionEncoder  
from mlxtend.frequent_patterns import apriori  
from mlxtend.frequent_patterns import association_rules  

from numpy import full
from pycocotools.coco import COCO

parser = argparse.ArgumentParser(description="Analysis of COCO labels.")
parser.add_argument('--data', type=str, default='/ntupool/data_for_host_machine/ML_Decoder/COCO14/')
args = parser.parse_args()

COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
               'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 
               'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
               'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
               'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
               'hair drier', 'toothbrush']

COCO_LABEL_MAP = { 1:  1,  2:  2,  3:  3,  4:  4,  5:  5,  6:  6,  7:  7,  8:  8,
                   9:  9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}

print(f"text length: {len(COCO_CLASSES)}")
print(f"label length: {len(COCO_LABEL_MAP)}")

# COCO Data loading
train_info = os.path.join(args.data, 'annotations/instances_train2014.json')
val_info = os.path.join(args.data, 'annotations/instances_val2014.json')
train_img = f'{args.data}/train2014'  # args.data

# Load annotaions of training data
coco_train = COCO(train_info)
all_train_ids = coco_train.imgs.keys()
print(len(all_train_ids))

coco_valid = COCO(val_info)
all_valid_ids = coco_valid.imgs.keys()
print(len(all_valid_ids))

# Read label of each image in training set
labels = []
for id in all_train_ids:
   label = []
   ann_ids = coco_train.getAnnIds(imgIds=[id])
   ann = coco_train.loadAnns(ann_ids)
   for object in ann:
      label.append(object['category_id'])
   # print(list(set(label)))
   labels.append(list(set(label)))
# print(labels)

# # Read label of each image in validation set
# labels = []
# for id in all_valid_ids:
#    label = []
#    ann_ids = coco_valid.getAnnIds(imgIds=[id])
#    ann = coco_valid.loadAnns(ann_ids)
#    for object in ann:
#       label.append(object['category_id'])
#    # print(list(set(label)))
#    labels.append(list(set(label)))
# # print(labels)

# # Find association rules with the Apriori algorithm
# te = TransactionEncoder()  
# df_tf = te.fit_transform(labels)  
# df = pd.DataFrame(df_tf,columns=te.columns_)
# frequent_itemsets = apriori(df, min_support=0.025, use_colnames=True)  
# frequent_itemsets.sort_values(by='support', ascending=False, inplace=True)  
# print(frequent_itemsets[frequent_itemsets.itemsets.apply(lambda x: len(x)) == 3])   
# association_rule = association_rules(frequent_itemsets,metric='confidence',min_threshold=0.01)
# association_rule.sort_values(by='confidence',ascending=False,inplace=True)
# association_rule

# # print the size of each image label
# label_size = {}
# for label in labels:
#    for item in label:
#       if item not in label_size.keys():
#          label_size[item] = 1
#       else:
#          label_size[item] += 1
# sorted_label_size = OrderedDict(sorted(label_size.items()))
# # print(sorted_label_size)

# count the frequency of all possible label patterns
pattern_dict = {}
for label in labels:
   for i in range(len(label)+1):
      for temp in combinations(label, i):
         temp_set = tuple(sorted(list(temp)))
         if temp_set not in pattern_dict.keys():
            pattern_dict[temp_set] = 1
         else:
            pattern_dict[temp_set] += 1
sorted_pattern_dict = OrderedDict(sorted(pattern_dict.items(), key=itemgetter(1)))
for key, value in sorted_pattern_dict.items():
   if len(key) == 4 and set([1,3,10]).issubset(set(key)):
      # pattern = []
      # for i in range(len(key)):
      #    item = COCO_CLASSES[COCO_LABEL_MAP[key[i]]-1]
      #    pattern.append(item)
      # print(pattern, value)
      print(key, value)
   



