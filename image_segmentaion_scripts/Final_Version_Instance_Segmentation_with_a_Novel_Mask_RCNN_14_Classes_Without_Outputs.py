#!/usr/bin/env python
# coding: utf-8

# # Start

# In[ ]:


#@title Imports
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import os
import cv2
import numpy as np
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
import torch.optim as optim
import torch
import shutil
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from tqdm import tqdm
import time
import random
import datetime

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import os
import cv2
import numpy as np
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
import torch.optim as optim
import torch
import shutil
import time
import itertools
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from PIL import Image
from PIL import Image, ImageDraw
pylab.rcParams['figure.figsize'] = (10.0, 8.0)


# # Colab Import

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')
# > Mounted at /content/drive

get_ipython().system('unzip /content/drive/MyDrive/ee641/Project_Datasets/ChestXDet_Dataset.zip')


# In[ ]:


import json
f_train = open('/content/ChestX_Det_train.json')
f_test = open('/content/ChestX_Det_test.json')
train_data_anns = json.load(f_train)
test_data_anns = json.load(f_test)


# In[ ]:


print(train_data_anns[5].keys())
print(train_data_anns[5]['file_name'])
print(train_data_anns[5]['syms'])
print(train_data_anns[5]['boxes'])
print(train_data_anns[5]['polygons'])
print('***')
print((train_data_anns[5]['boxes'])[0])
print((train_data_anns[5]['polygons'])[0])


# In[ ]:


list_of_syms = []
for i in train_data_anns:
  temp = i['syms']
  for j in temp:
    if((j in list_of_syms) == False):
      list_of_syms.append(j)
print(list_of_syms)
print(len(list_of_syms))

temp_dict = {}
counter = 1
for i in list_of_syms:
  if((i in temp_dict) == False):
    temp_dict[i] = counter
    counter += 1
print(temp_dict)
mapping_labels = temp_dict


# In[ ]:


# class NoneTransform(object):
#     ''' Does nothing to the image. To be used instead of None '''
    
#     def __call__(self, image):       
#         return image

# temp_transform = transforms.Compose([
#             transforms.ToTensor(),            
#             transforms.Lambda(lambda x: x.repeat(3, 1, 1))  if temp_img.mode!='RGB'  else NoneTransform()            
#             ]) 


# In[ ]:


sum_matrix = np.zeros((224,224,3))
for i in range(len(train_data_anns)):
  img_name = (train_data_anns[i])['file_name']
  path = '/content/ChestXDet_Dataset/train/' + img_name
  temp_img = Image.open(path)
  temp_img = temp_img.convert('RGB')
  temp_img = temp_img.resize((224, 224))
  temp_img_tensor = temp_transform(temp_img)
  temp_img_tensor = temp_img_tensor.permute(1, 2, 0)
  temp_img_numpy = temp_img_tensor.numpy()
  sum_matrix = sum_matrix + temp_img_numpy

sum_matrix_1d = sum_matrix[:,:,0]
sum_matrix_1d_avg = sum_matrix_1d / len(train_data_anns)
mean_chest = np.mean(sum_matrix_1d_avg, axis = None)
std_chest = np.std(sum_matrix_1d_avg, axis = None)
print("Mean of the Gray Scale Training Chest Images of Pixels = ", mean_chest)
print("Std of the Gray Scale Training Chest Images of Pixels = ", std_chest)


# In[ ]:


def plot_filled_masks_on_image(image_np, polygons):

    image = Image.fromarray(image_np)
    img_with_masks = image.copy()
    mask_color = (0, 255, 0, 128)  
    mask_layer = Image.new('RGBA', img_with_masks.size, (0, 0, 0, 0))
    draw_ctx = ImageDraw.Draw(mask_layer)
    
    for polygon in polygons:
        draw_ctx.polygon(polygon, fill=mask_color)
    img_with_masks.paste(mask_layer, mask=mask_layer.split()[3], box=(0, 0))
    img_with_masks_np = np.array(img_with_masks)

    return img_with_masks_np


# In[ ]:


temp_img = Image.open('/content/ChestXDet_Dataset/train/36204.png')
temp_img = temp_img.resize((224, 224))
temp_img = temp_img.convert('RGB')
masks = train_data_anns[4]['polygons']
print(masks)

#masks_tuple = [[tuple(coords) for coords in polygon] for polygon in masks]
rescaled_list = [
    [
        (coord[0] * 224 / 1024, coord[1] * 224 / 1024)
        for coord in polygon
    ]
    for polygon in masks
]
print(rescaled_list)

temp_img = np.array(temp_img)
print(temp_img.shape)

image_with_actual_masks = plot_filled_masks_on_image(temp_img, rescaled_list)

fig, ax = plt.subplots()
ax.imshow(image_with_actual_masks)
plt.show()

print(train_data_anns[4])


# # Custom Dataset Class

# In[ ]:


import math
from PIL import Image, ImageDraw
from PIL import ImagePath 


# In[ ]:


import math

def angle(point, centroid):
    dx, dy = point[0] - centroid[0], point[1] - centroid[1]
    return math.atan2(dy, dx)


# In[ ]:


list_of_syms = []
for i in train_data_anns:
  temp = i['syms']
  for j in temp:
    if((j in list_of_syms) == False):
      list_of_syms.append(j)
print(list_of_syms)
print(len(list_of_syms))

temp_dict = {}
counter = 1
for i in list_of_syms:
  if((i in temp_dict) == False):
    temp_dict[i] = counter
    counter += 1
print(temp_dict)
mapping_labels = temp_dict
map_labels = mapping_labels


# In[ ]:


class ChestXDet_Dataset(Dataset):
  def __init__(self, images_root_dir, anns_dir, transform=None, cuda = True, mapping_labels = None):
    self.images_root_dir = images_root_dir
    self.anns_dir = anns_dir
    self.transform = transform
    self.image_names = os.listdir(images_root_dir)
    anns_json = open(anns_dir)
    self.img_anns = json.load(anns_json)
    list_of_annotated_image_filenames = []

    self.mapping_labels = mapping_labels

    for i in range(len(self.image_names)):
      temp_image_filename = self.image_names[i]
      temp_image_directory = self.images_root_dir + '/' + temp_image_filename
      temp_annotation_index =  [i for i,_ in enumerate(self.img_anns) if _['file_name'] == temp_image_filename][0]
      temp_annotations = self.img_anns[temp_annotation_index]
      if(len(temp_annotations['boxes']) > 0):
        list_of_annotated_image_filenames.append(temp_image_filename)
    
    self.annotated_image_names = list_of_annotated_image_filenames

    self.img_h = 224
    self.img_w = 224
    #self.org_image_width = 1024
    #self.org_image_height = 1024
    self.cuda = cuda

  def __len__(self):
    return len(self.annotated_image_names)
  
  def __getitem__(self, idx):
    image_filename = self.annotated_image_names[idx]
    annotation_index =  [i for i,_ in enumerate(self.img_anns) if _['file_name'] == image_filename][0]
    image_directory = self.images_root_dir + '/' + image_filename

    image = Image.open(image_directory)
    org_image_width = (image.size)[0]
    org_image_height = (image.size)[1]

    #print(org_image_width)
    #print(org_image_height)

    annotations = self.img_anns[annotation_index]

    list_of_bounding_box_coordinates = []
    list_of_masks = []
    list_of_polys = []
    labels = []

    temp_polygons = annotations['polygons']
    temp_boxes = annotations['boxes']
    temp_labels = annotations['syms']

    for i in range(len(temp_boxes)):
      x1_converted = ((temp_boxes[i])[0]) * (self.img_w / org_image_width)
      y1_converted = ((temp_boxes[i])[1]) * (self.img_h / org_image_height)
      x2_converted = ((temp_boxes[i])[2]) * (self.img_w / org_image_width)
      y2_converted = ((temp_boxes[i])[3]) * (self.img_h / org_image_height)
      #temp_bbox_coors = np.array([x1_converted, y1_converted, x2_converted, y2_converted], dtype = np.float32)
      temp_bbox_coors = [x1_converted, y1_converted, x2_converted, y2_converted]
      #torch_temp_bbox_coors = torch.FloatTensor(torch.from_numpy(temp_bbox_coors))
      list_of_bounding_box_coordinates.append(temp_bbox_coors)
      labels.append(self.mapping_labels[temp_labels[i]])

      poly_i = temp_polygons[i]
      #rescaled_list = [(coord[0] * self.img_w / org_image_width, coord[1] * self.img_h / org_image_height) for coord in poly_i]
      temp_flat_list_polygons = list(itertools.chain(*poly_i))
      rescaled_flattened_polygons = []
      for k in range(int(len(temp_flat_list_polygons) / 2)):
        rescaled_x_coor = temp_flat_list_polygons[2*k] * (self.img_w / org_image_width)
        rescaled_y_coor = temp_flat_list_polygons[2*k+1] * (self.img_h / org_image_height)
        rescaled_flattened_polygons.append(rescaled_x_coor)
        rescaled_flattened_polygons.append(rescaled_y_coor)
      temp_mask_image = Image.new('L', (self.img_w, self.img_h), 0)
      poly_rescaled_tuples = [(x, y) for x, y in zip(rescaled_flattened_polygons[::2], rescaled_flattened_polygons[1::2])]
      #ImageDraw.Draw(temp_mask_image).polygon(rescaled_list, outline=1, fill=1)
      ImageDraw.Draw(temp_mask_image).polygon(poly_rescaled_tuples, outline=1, fill=1)

      temp_mask = list(np.array(temp_mask_image, dtype = np.uint8))
      #torch_mask = torch.from_numpy(temp_mask)
      #list_of_masks.append(torch_mask)
      list_of_masks.append(temp_mask)
      #list_of_polys.append(rescaled_list)
      list_of_polys.append(poly_rescaled_tuples)

      
    # if(image.mode != 'RGB'):
    #   image = image.convert('RGB')
    
    image = image.resize((self.img_h, self.img_w))

    if(self.transform != None):
      image = self.transform(image)

    #print(image.shape)

    #image = image.permute(2, 0, 1)


    torch_labels = torch.LongTensor(labels)
    torch_list_of_bounding_box_coordinates = torch.FloatTensor(list_of_bounding_box_coordinates)
    torch_list_of_masks = torch.ByteTensor(list_of_masks)
    #torch_list_of_polys = torch.FloatTensor(list_of_polys)

    target_dict = {}

    target_dict['boxes'] = torch_list_of_bounding_box_coordinates
    target_dict['labels'] = torch_labels
    target_dict['masks'] = torch_list_of_masks
    target_dict['polys'] = list_of_polys

    target = target_dict

    return image, target

def collate_fn(batch):
    images = []
    targets = []
    #val_bbox_coordinates = []
    #plain_images = []
    #image_heights = []
    #image_widths = []
    for item in batch:
        images.append(item[0])
        targets.append(item[1])
    images = torch.stack(images, 0)
    #if(mode == 'train'):
    return images, targets


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print(device)


# # Model Factory

# In[ ]:


import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights


def get_model_instance_segmentation(num_classes):

    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


# # Setup

# In[ ]:


batch_size = 50
root_directory_train = '/content/ChestXDet_Dataset/train'
root_directory_val = '/content/ChestXDet_Dataset/test'
annotation_file_directory_train = '/content/ChestX_Det_train.json'
annotation_file_directory_val = '/content/ChestX_Det_test.json'

class NoneTransform(object):
  def __call__(self, image):
    return image

class GrayscaleToRGB:
  def __call__(self, image):
    if image.mode != 'RGB':
      return image.convert('RGB')
    return image

mean_chest = 0.5240804197105211
std_chest = 0.17148634738092733

my_transform_with_Normalization = transforms.Compose([
    GrayscaleToRGB(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #transforms.Normalize([mean_chest, mean_chest, mean_chest], [std_chest, std_chest, std_chest])
])


#Mean of the Gray Scale Training Chest Images of Pixels =  0.5240804197105211
#Std of the Gray Scale Training Chest Images of Pixels =  0.17148634738092733



# my_transform_Specialized = transforms.Compose([
#             transforms.ToTensor(),            
#             transforms.Lambda(lambda x: x.repeat(3, 1, 1))  if temp_img.mode!='RGB'  else NoneTransform(),
#             transforms.Normalize([mean_chest, mean_chest, mean_chest], [std_chest, std_chest, std_chest])                
#             ]) 

# my_transform_ImageNet = transforms.Compose([
#             transforms.ToTensor(),            
#             transforms.Lambda(lambda x: x.repeat(3, 1, 1))  if temp_img.mode!='RGB'  else NoneTransform(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                
#             ]) 

train_dataset = ChestXDet_Dataset(root_directory_train, annotation_file_directory_train, transform=my_transform_with_Normalization, cuda=True, mapping_labels = map_labels)
val_dataset = ChestXDet_Dataset(root_directory_val, annotation_file_directory_val, transform=my_transform_with_Normalization, cuda=True, mapping_labels = map_labels)


# In[ ]:


train_loader = DataLoader(train_dataset, shuffle = True, batch_size=batch_size, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, shuffle = False, batch_size=batch_size, collate_fn=collate_fn)


# # Loss Curves Plotting

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


def plotter(losses_classifier, losses_box_reg, losses_objectness, losses_rpn_box_reg, losses_mask):
  
  x_axis = np.arange(1,len(losses_classifier)+1,1)
  y_axis_1 = losses_classifier

  fig, ax = plt.subplots(figsize=(25,25))

  ax.plot(x_axis, y_axis_1,'blue', linewidth=1.5, label = 'Negative Log-Likelihood Classification Loss')

  plt.title('Training Negative Log-Likelihood Classification Loss vs. Epochs')
  plt.xticks(np.arange(1,len(losses_classifier)+1,1))
  plt.yticks(np.arange(0,max(losses_classifier),0.01))
  plt.xlabel('Epoch')
  plt.ylabel('Negative Log-Likelihood Classification Loss')
  plt.legend(loc = 'upper right')
  plt.grid()
  plt.show()

  print('\n')

  x_axis = np.arange(1,len(losses_box_reg)+1,1)
  y_axis_1 = losses_box_reg

  fig, ax = plt.subplots(figsize=(25,25))

  ax.plot(x_axis, y_axis_1,'red', linewidth=1.5, label = 'Regression Loss for predicted Bounding Box Coordinates')

  plt.title('Regression Loss for predicted Bounding Box Coordinates vs. Epochs')
  plt.xticks(np.arange(1,len(losses_box_reg)+1,1))
  plt.yticks(np.arange(0,max(losses_box_reg),0.01))
  plt.xlabel('Epoch')
  plt.ylabel('Regression Loss for predicted Bounding Box Coordinates')
  plt.legend(loc = 'upper right')
  plt.grid()
  plt.show()

  print('\n')

  x_axis = np.arange(1,len(losses_objectness)+1,1)
  y_axis_1 = losses_objectness

  fig, ax = plt.subplots(figsize=(25,25))

  ax.plot(x_axis, y_axis_1,'green', linewidth=1.5, label = 'Binary Cross-Entropy Loss for RPN Objectness')

  plt.title('Binary Cross-Entropy Loss for RPN Objectness vs. Epochs')
  plt.xticks(np.arange(1,len(losses_objectness)+1,1))
  plt.yticks(np.arange(0,max(losses_objectness),0.01))
  plt.xlabel('Epoch')
  plt.ylabel('Binary Cross-Entropy Loss for RPN Objectness')
  plt.legend(loc = 'upper right')
  plt.grid()
  plt.show()

  print('\n')

  x_axis = np.arange(1,len(losses_rpn_box_reg)+1,1)
  y_axis_1 = losses_rpn_box_reg

  fig, ax = plt.subplots(figsize=(25,25))

  ax.plot(x_axis, y_axis_1,'gray', linewidth=1.5, label = 'Regression Loss for RPN Network Box Coordinates')

  plt.title('Regression Loss for RPN Network Box Coordinates vs. Epochs')
  plt.xticks(np.arange(1,len(losses_rpn_box_reg)+1,1))
  plt.yticks(np.arange(0,max(losses_rpn_box_reg),0.01))
  plt.xlabel('Epoch')
  plt.ylabel('Regression Loss for RPN Network Box Coordinates')
  plt.legend(loc = 'upper right')
  plt.grid()
  plt.show()

  print('\n')

  x_axis = np.arange(1,len(losses_mask)+1,1)
  y_axis_1 = losses_mask

  fig, ax = plt.subplots(figsize=(25,25))

  ax.plot(x_axis, y_axis_1,'blue', linewidth=1.5, label = 'Mask Loss')

  plt.title('Training Mask Loss vs. Epochs')
  plt.xticks(np.arange(1,len(losses_mask)+1,1))
  plt.yticks(np.arange(0,max(losses_mask),0.01))
  plt.xlabel('Epoch')
  plt.ylabel('Negative Log-Likelihood Classification Loss')
  plt.legend(loc = 'upper right')
  plt.grid()
  plt.show()

  return 


# # Performance Metrics Plotting

# In[ ]:


def plotter_2(scores_dict):

  acc_tr_epochs = scores_dict['acc_tr']
  pre_tr_epochs = scores_dict['pre_tr']  
  rec_tr_epochs = scores_dict['rec_tr'] 
  f1_tr_epochs = scores_dict['f1_tr']  
  dice_tr_epochs = scores_dict['dice_tr']
  iou_tr_epochs = scores_dict['iou_tr'] 

  acc_val_epochs = scores_dict['acc_val'] 
  pre_val_epochs = scores_dict['pre_val']
  rec_val_epochs = scores_dict['rec_val'] 
  f1_val_epochs = scores_dict['f1_val'] 
  dice_val_epochs = scores_dict['dice_val']
  iou_val_epochs = scores_dict['iou_val']
  
  x_axis = np.arange(1,len(acc_tr_epochs)+1,1)

  y_axis_1 = acc_tr_epochs
  y_axis_2 = acc_val_epochs

  fig, ax = plt.subplots(figsize=(25,25))

  ax.plot(x_axis, y_axis_1, 'green', linewidth=1.5, label = 'Training Mask Accuracy')
  ax.plot(x_axis, y_axis_2, 'blue', linewidth=1.5, label = 'Validation Mask Accuracy')

  plt.title('Training Mask Accuracy vs. Validation Mask Accuracy through Epochs')
  plt.xticks(np.arange(1,len(acc_tr_epochs)+1,1))
  plt.yticks(np.arange(0, max(acc_tr_epochs + acc_val_epochs) + 0.05, 0.01))
  plt.xlabel('Epoch')
  plt.ylabel('Mask Accuracy')
  plt.legend(loc = 'upper right')
  plt.grid()
  plt.show()

  print('\n')

  y_axis_1 = pre_tr_epochs
  y_axis_2 = pre_val_epochs

  fig, ax = plt.subplots(figsize=(25,25))

  ax.plot(x_axis, y_axis_1, 'green', linewidth=1.5, label = 'Training Mask Precision')
  ax.plot(x_axis, y_axis_2, 'blue', linewidth=1.5, label = 'Validation Mask Precision')

  plt.title('Training Mask Precision vs. Validation Mask Precision through Epochs')
  plt.xticks(np.arange(1,len(pre_tr_epochs)+1,1))
  plt.yticks(np.arange(0, max(pre_tr_epochs + pre_val_epochs) + 0.05,0.01))
  plt.xlabel('Epoch')
  plt.ylabel('Mask Precision')
  plt.legend(loc = 'upper right')
  plt.grid()
  plt.show()

  print('\n')

  y_axis_1 = rec_tr_epochs
  y_axis_2 = rec_val_epochs

  fig, ax = plt.subplots(figsize=(25,25))

  ax.plot(x_axis, y_axis_1, 'green', linewidth=1.5, label = 'Training Mask Recall')
  ax.plot(x_axis, y_axis_2, 'blue', linewidth=1.5, label = 'Validation Mask Recall')

  plt.title('Training Mask Recall vs. Validation Mask Recall through Epochs')
  plt.xticks(np.arange(1,len(rec_tr_epochs)+1,1))
  plt.yticks(np.arange(0,max(rec_tr_epochs + rec_val_epochs) + 0.05,0.01))
  plt.xlabel('Epoch')
  plt.ylabel('Mask Recall')
  plt.legend(loc = 'upper right')
  plt.grid()
  plt.show()

  print('\n')


  y_axis_1 = f1_tr_epochs
  y_axis_2 = f1_val_epochs

  fig, ax = plt.subplots(figsize=(25,25))

  ax.plot(x_axis, y_axis_1, 'green', linewidth=1.5, label = 'Training Mask F1-Score')
  ax.plot(x_axis, y_axis_2, 'blue', linewidth=1.5, label = 'Validation Mask F1-Score')

  plt.title('Training Mask F1-Score vs. Validation Mask F1-Score through Epochs')
  plt.xticks(np.arange(1,len(f1_tr_epochs)+1,1))
  plt.yticks(np.arange(0,max(f1_tr_epochs + f1_val_epochs) + 0.05,0.01))
  plt.xlabel('Epoch')
  plt.ylabel('Mask F1-Score')
  plt.legend(loc = 'upper right')
  plt.grid()
  plt.show()

  print('\n')

  y_axis_1 = dice_tr_epochs
  y_axis_2 = dice_val_epochs

  fig, ax = plt.subplots(figsize=(25,25))

  ax.plot(x_axis, y_axis_1, 'green', linewidth=1.5, label = 'Training Mask DICE-Score')
  ax.plot(x_axis, y_axis_2, 'blue', linewidth=1.5, label = 'Validation Mask DICE-Score')

  plt.title('Training Mask DICE-Score vs. Validation Mask DICE-Score through Epochs')
  plt.xticks(np.arange(1,len(dice_tr_epochs)+1,1))
  plt.yticks(np.arange(0,max(dice_tr_epochs + dice_val_epochs) + 0.05,0.01))
  plt.xlabel('Epoch')
  plt.ylabel('Mask DICE-Score')
  plt.legend(loc = 'upper right')
  plt.grid()
  plt.show()

  print('\n')

  y_axis_1 = iou_tr_epochs
  y_axis_2 = iou_val_epochs

  fig, ax = plt.subplots(figsize=(25,25))

  ax.plot(x_axis, y_axis_1, 'green', linewidth=1.5, label = 'Training Mask Intersection over Union Score')
  ax.plot(x_axis, y_axis_2, 'blue', linewidth=1.5, label = 'Validation Mask Intersection over Union Score')

  plt.title('Training Mask IoU Score vs. Validation Mask IoU Score through Epochs')
  plt.xticks(np.arange(1,len(iou_tr_epochs)+1,1))
  plt.yticks(np.arange(0,max(iou_tr_epochs + iou_val_epochs) + 0.05,0.01))
  plt.xlabel('Epoch')
  plt.ylabel('Mask Intersection over Union Score')
  plt.legend(loc = 'upper right')
  plt.grid()
  plt.show()

  return 


# In[ ]:


def plotter_3(scores_dict):

  acc_tr_epochs = scores_dict['acc_tr']
  pre_tr_epochs = scores_dict['pre_tr']  
  rec_tr_epochs = scores_dict['rec_tr'] 
  f1_tr_epochs = scores_dict['f1_tr']  
  dice_tr_epochs = scores_dict['dice_tr']
  iou_tr_epochs = scores_dict['iou_tr'] 

  acc_val_epochs = scores_dict['acc_val'] 
  pre_val_epochs = scores_dict['pre_val']
  rec_val_epochs = scores_dict['rec_val'] 
  f1_val_epochs = scores_dict['f1_val'] 
  dice_val_epochs = scores_dict['dice_val']
  iou_val_epochs = scores_dict['iou_val']
  
  x_axis = np.arange(1,2*len(acc_tr_epochs)+1,2)

  y_axis_1 = acc_tr_epochs
  y_axis_2 = acc_val_epochs

  fig, ax = plt.subplots(figsize=(25,25))

  ax.plot(x_axis, y_axis_1, 'green', linewidth=1.5, label = 'Training Mask Accuracy')
  ax.plot(x_axis, y_axis_2, 'blue', linewidth=1.5, label = 'Validation Mask Accuracy')

  plt.title('Training Mask Accuracy vs. Validation Mask Accuracy through Epochs')
  plt.xticks(x_axis)
  plt.yticks(np.arange(0, max(acc_tr_epochs + acc_val_epochs) + 0.05, 0.02))
  plt.xlabel('Epoch')
  plt.ylabel('Mask Accuracy')
  plt.legend(loc = 'upper right')
  plt.grid()
  plt.show()

  print('\n')

  y_axis_1 = pre_tr_epochs
  y_axis_2 = pre_val_epochs

  fig, ax = plt.subplots(figsize=(25,25))

  ax.plot(x_axis, y_axis_1, 'green', linewidth=1.5, label = 'Training Mask Precision')
  ax.plot(x_axis, y_axis_2, 'blue', linewidth=1.5, label = 'Validation Mask Precision')

  plt.title('Training Mask Precision vs. Validation Mask Precision through Epochs')
  plt.xticks(x_axis)
  plt.yticks(np.arange(0, max(pre_tr_epochs + pre_val_epochs) + 0.05,0.02))
  plt.xlabel('Epoch')
  plt.ylabel('Mask Precision')
  plt.legend(loc = 'upper right')
  plt.grid()
  plt.show()

  print('\n')

  y_axis_1 = rec_tr_epochs
  y_axis_2 = rec_val_epochs

  fig, ax = plt.subplots(figsize=(25,25))

  ax.plot(x_axis, y_axis_1, 'green', linewidth=1.5, label = 'Training Mask Recall')
  ax.plot(x_axis, y_axis_2, 'blue', linewidth=1.5, label = 'Validation Mask Recall')

  plt.title('Training Mask Recall vs. Validation Mask Recall through Epochs')
  plt.xticks(x_axis)
  plt.yticks(np.arange(0,max(rec_tr_epochs + rec_val_epochs) + 0.05,0.02))
  plt.xlabel('Epoch')
  plt.ylabel('Mask Recall')
  plt.legend(loc = 'upper right')
  plt.grid()
  plt.show()

  print('\n')


  y_axis_1 = f1_tr_epochs
  y_axis_2 = f1_val_epochs

  fig, ax = plt.subplots(figsize=(25,25))

  ax.plot(x_axis, y_axis_1, 'green', linewidth=1.5, label = 'Training Mask F1-Score')
  ax.plot(x_axis, y_axis_2, 'blue', linewidth=1.5, label = 'Validation Mask F1-Score')

  plt.title('Training Mask F1-Score vs. Validation Mask F1-Score through Epochs')
  plt.xticks(x_axis)
  plt.yticks(np.arange(0,max(f1_tr_epochs + f1_val_epochs) + 0.05,0.02))
  plt.xlabel('Epoch')
  plt.ylabel('Mask F1-Score')
  plt.legend(loc = 'upper right')
  plt.grid()
  plt.show()

  print('\n')

  y_axis_1 = dice_tr_epochs
  y_axis_2 = dice_val_epochs

  fig, ax = plt.subplots(figsize=(25,25))

  ax.plot(x_axis, y_axis_1, 'green', linewidth=1.5, label = 'Training Mask DICE-Score')
  ax.plot(x_axis, y_axis_2, 'blue', linewidth=1.5, label = 'Validation Mask DICE-Score')

  plt.title('Training Mask DICE-Score vs. Validation Mask DICE-Score through Epochs')
  plt.xticks(x_axis)
  plt.yticks(np.arange(0,max(dice_tr_epochs + dice_val_epochs) + 0.05,0.02))
  plt.xlabel('Epoch')
  plt.ylabel('Mask DICE-Score')
  plt.legend(loc = 'upper right')
  plt.grid()
  plt.show()

  print('\n')

  y_axis_1 = iou_tr_epochs
  y_axis_2 = iou_val_epochs

  fig, ax = plt.subplots(figsize=(25,25))

  ax.plot(x_axis, y_axis_1, 'green', linewidth=1.5, label = 'Training Mask Intersection over Union Score')
  ax.plot(x_axis, y_axis_2, 'blue', linewidth=1.5, label = 'Validation Mask Intersection over Union Score')

  plt.title('Training Mask IoU Score vs. Validation Mask IoU Score through Epochs')
  plt.xticks(x_axis)
  plt.yticks(np.arange(0,max(iou_tr_epochs + iou_val_epochs) + 0.05,0.02))
  plt.xlabel('Epoch')
  plt.ylabel('Mask Intersection over Union Score')
  plt.legend(loc = 'upper right')
  plt.grid()
  plt.show()

  return 


# In[ ]:


def denormalize_image(normalized_image, mean, std):
    normalized_image = normalized_image.astype(np.float32)
    denormalized_image = np.zeros_like(normalized_image)
    mean = np.array(mean)
    std = np.array(std)

    for channel in range(3):
        denormalized_image[channel, :, :] = (normalized_image[channel, :, :] * std[channel]) + mean[channel]

    denormalized_image = (denormalized_image - denormalized_image.min()) / (denormalized_image.max() - denormalized_image.min()) * 255
    denormalized_image = denormalized_image.astype(np.uint8)
    denormalized_image = np.transpose(denormalized_image, (1, 2, 0))

    return denormalized_image


# In[ ]:


mean_chest = 0.5240804197105211
std_chest = 0.17148634738092733


# # Evaluate Here

# In[ ]:


def combine_masks(arrays):
    if len(arrays) == 0:
        return np.zeros((224, 224))
        
    final_array = arrays[0]
    for array in arrays[1:]:
        final_array = np.maximum(final_array, array)
        
    return final_array


# In[ ]:


from matplotlib.patches import Rectangle, Polygon
def evaluate(model, val_loader, device, epoch):
  # mean_chest = [0.5240804197105211, 0.5240804197105211, 0.5240804197105211]
  # std_chest = [0.17148634738092733, 0.17148634738092733, 0.17148634738092733]
  mean_chest = [0.485, 0.456, 0.406]
  std_chest = [0.229, 0.224, 0.225]
  #[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

  TP_epoch = 0
  FP_epoch = 0
  TN_epoch = 0
  FN_epoch = 0
  counter_for_visuals = True 

  with torch.no_grad():
    for images, targets in val_loader:
      model.eval()
      image_batch = images.to(device)
      #target_batch = [{'boxes':dictionary['boxes'].to(device), 'labels':dictionary['labels'].to(device), 'masks':dictionary['masks'].to(device)} for dictionary in targets]
      predictions = model(image_batch)
      #indices_of_images = [0,1,2,3,4,5,6,7]

      if((counter_for_visuals == True) and (epoch % 2 == 0)):
        counter_row = 0
        counter_column = 0
        fig, ax = plt.subplots(2, 4, figsize=(48, 24))
        fig_m, ax_m = plt.subplots(2, 4, figsize=(48, 24))
        #indices_of_images = random.sample(range(0, 15), 8)
        indices_of_images = [0,2,4,6,8,10,12,14]

        for img_trg_index in indices_of_images:
          box_pred = list(predictions[img_trg_index]['boxes'])
          box_actual = list(targets[img_trg_index]['boxes'])
          mask_pred = list(predictions[img_trg_index]['masks'])
          mask_actual = list(targets[img_trg_index]['masks'])
          mask_actual_polys = list(targets[img_trg_index]['polys'])

          actual_image = images[img_trg_index]
          image_numpy = (actual_image.detach().cpu()).numpy()
          actual_denormalized_image = denormalize_image(image_numpy, mean_chest, std_chest)

          ax[counter_row, counter_column].imshow(actual_denormalized_image)
          ax_m[counter_row, counter_column].imshow(actual_denormalized_image)

          mask_actual_arrs = []
          
          for i in mask_actual:
            tmp = i.numpy()
            temp_locs = np.where(tmp == 1)
            temp_mask = np.zeros((224,224))
            temp_mask[temp_locs[0], temp_locs[1]] = 1
            mask_actual_arrs.append(temp_mask)

          combined_mask_actual = combine_masks(mask_actual_arrs)
          actual_mask_color = np.array([0, 0, 1, 0.4]) #Blue
          actual_mask_rgb = np.zeros((224, 224, 4))
          actual_mask_rgb[combined_mask_actual == 1] = actual_mask_color
          ax_m[counter_row, counter_column].imshow(actual_mask_rgb)

          mask_pred_arrs = []

          for i in mask_pred:
            tmp = i.detach().cpu()
            tmp2 = (tmp[0]).numpy()
            temp_locs = np.where(tmp2 > 0.6)
            temp_mask = np.zeros((224, 224))
            temp_mask[temp_locs[0], temp_locs[1]] = 1
            mask_pred_arrs.append(temp_mask)
            
          combined_mask_predicted = combine_masks(mask_pred_arrs)
          predicted_mask_color = np.array([1, 1, 0, 0.2]) #Yellow
          predicted_mask_rgb = np.zeros((224, 224, 4))
          predicted_mask_rgb[combined_mask_predicted == 1] = predicted_mask_color
          ax_m[counter_row, counter_column].imshow(predicted_mask_rgb)

          for j in box_actual:
            tmp = j.numpy()
            tmp_rect = Rectangle((tmp[0], tmp[1]), tmp[2] - tmp[0], tmp[3] - tmp[1], linewidth=2, edgecolor='r', facecolor='none')
            #tmp_rect = Rectangle((tmp[0], tmp[3]), tmp[2] - tmp[0], tmp[3] - tmp[1], linewidth=2, edgecolor='r', facecolor='none')
            ax[counter_row, counter_column].add_patch(tmp_rect)
          for i in box_pred:
            tmp = i.detach().cpu().numpy()
            tmp_rect = Rectangle((tmp[0], tmp[1]), tmp[2] - tmp[0], tmp[3] - tmp[1], linewidth=1, edgecolor='g', facecolor='none')
            #tmp_rect = Rectangle((tmp[0], tmp[3]), tmp[2] - tmp[0], tmp[3] - tmp[1], linewidth=1, edgecolor='b', facecolor='none')
            ax[counter_row, counter_column].add_patch(tmp_rect)

          counter_column += 1
          if(counter_column > 3):
            counter_row += 1
            counter_column = 0

        #image_dir = '/content/Boxes_Images_Resized/'
        #image_dir = '/content/drive/MyDrive/ee641/Project_Datasets/Masked_Images_through_Epochs/'
        image_dir = '/content/drive/MyDrive/ee641/Project_Datasets/Images_Instance_Segmentation_3/'
        temp_filename = image_dir + 'Epoch_' + str(epoch) + '_BBox_Preds_BLUE_BBox_Actual_RED.png'
        temp_filename_m = image_dir + 'Epoch_' + str(epoch) + '_Mask_Preds_BLUE_Mask_Actual_RED.png'
        temp_title = 'Epoch ' + str(epoch) + ' BBox Predictions (Blue) vs. Actual BBox(Red)'
        temp_title_m = 'Epoch ' + str(epoch) + ' Mask Predictions (Blue) vs. Actual Masks(Red)'
        #plt.title(temp_title)
        fig.suptitle(temp_title)
        #plt.show()
        fig_m.suptitle(temp_title_m)
        plt.show()
        fig.savefig(temp_filename)
        fig_m.savefig(temp_filename_m)

      counter_for_visuals = False

      for i in range(len(targets)):
        mask_pred = list(predictions[i]['masks'])
        mask_actual = list((targets[i])['masks'])
        mask_pred_arrs = []
        mask_actual_arrs = []
        #print(len(mask_pred))
        #print(len(mask_actual))
        #print(mask_pred[0].shape)
        #print(mask_actual[0].shape)
        if(len(mask_pred) == 0):
          temp_mask_pred = np.zeros((224,224))
          mask_pred_arrs.append(temp_mask_pred)
        else:
          for j in range(len(mask_pred)):
            temp_pred = ((mask_pred[j].detach().cpu())[0]).numpy()
            temp_locs_pred = np.where(temp_pred > 0.5)
            temp_mask_pred = np.zeros((224,224))
            temp_mask_pred[temp_locs_pred[0], temp_locs_pred[1]] = 1
            mask_pred_arrs.append(temp_mask_pred)

        if(len(mask_actual) == 0):
          temp_mask_actual = np.zeros((224,224))
          mask_actual_arrs.append(temp_mask_actual)
        else:
          for j in range(len(mask_actual)):
            temp_actual = mask_actual[j].numpy()
            temp_locs_actual = np.where(temp_actual == 1)
            temp_mask_actual = np.zeros((224,224))
            temp_mask_actual[temp_locs_actual[0], temp_locs_actual[1]] = 1
            mask_actual_arrs.append(temp_mask_actual)

        #print('HERE')
        combined_mask_pred = combine_masks(mask_pred_arrs)
        #print(combined_mask_pred)
        #print(combined_mask_pred.shape)
        #print('HERE2')
        combined_mask_actual = combine_masks(mask_actual_arrs)
        #print(combined_mask_actual)
        #print(combined_mask_actual.shape)
        #return
        
        TP, FP, TN, FN = confusion_matrix_elements(combined_mask_pred, combined_mask_actual)

        TP_epoch += TP
        FP_epoch += FP
        TN_epoch += TN
        FN_epoch += FN
    
    acc = accuracy(TP_epoch, FP_epoch, TN_epoch, FN_epoch)
    prec = precision(TP_epoch, FP_epoch)
    rec = recall(TP_epoch, FN_epoch)
    f1 = f1_score(prec, rec)
    dice = dice_coefficient(TP_epoch, FP_epoch, FN_epoch)
    intersection_over_union = iou(TP_epoch, FP_epoch, FN_epoch)
    # print(acc)
    # print(prec)
    # print(rec)
    # print(f1)
    # print(dice)

  return acc, prec, rec, f1, dice, intersection_over_union


# # Training and Inferring on Validation - Normalized Based On ImageNet with 14 classes

# In[ ]:


import gc
gc.collect()
torch.cuda.empty_cache()


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[ ]:


num_classes = 14
model = get_model_instance_segmentation(num_classes)
device = torch.device(device)
model = model.to(device)


# In[ ]:


def confusion_matrix_elements(pred_mask, true_mask):
    assert pred_mask.shape == true_mask.shape, "Shape mismatch in input masks"
    
    TP = np.sum((pred_mask == 1) & (true_mask == 1))
    FP = np.sum((pred_mask == 1) & (true_mask == 0))
    TN = np.sum((pred_mask == 0) & (true_mask == 0))
    FN = np.sum((pred_mask == 0) & (true_mask == 1))
    
    return TP, FP, TN, FN


# In[ ]:


def accuracy(TP, FP, TN, FN):
    return (TP + TN) / (TP + FP + TN + FN)

def precision(TP, FP):
    return TP / (TP + FP)

def recall(TP, FN):
    return TP / (TP + FN)

def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

def dice_coefficient(TP, FP, FN):
    return (2 * TP) / ((2 * TP) + FP + FN)

def iou(TP, FP, FN):
    return TP / (TP + FP + FN)


# # Hyper setting #1

# In[ ]:


learning_rate = 1e-3
l2_reg = 1e-5
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=15, gamma=0.1)

num_epochs = 40
#num_epochs = 2

epoch_losses_classifier = []
epoch_losses_box_reg = []
epoch_losses_objectness = []
epoch_losses_rpn_box_reg = []
epoch_losses_mask_reg = []

acc_tr_epochs = []
pre_tr_epochs = []
rec_tr_epochs = []
f1_tr_epochs = []
dice_tr_epochs = []
iou_tr_epochs = []

acc_val_epochs = []
pre_val_epochs = []
rec_val_epochs = []
f1_val_epochs = []
dice_val_epochs = []
iou_val_epochs = []


for epoch in range(num_epochs):
    losses_classifier = []
    losses_box_reg = []
    losses_objectness = []
    losses_rpn_box_reg = []
    losses_mask_reg = []
    counter_train = 0
    temp_epoch = epoch + 1
    
    PATH = '/content/drive/MyDrive/ee641/Project_Datasets/Model_Checkpoints_1/model_epoch_' + str(temp_epoch) + '.pth'
    
    #acc, prec, rec, f1, dice, intersection_over_union = evaluate(model, val_loader, device, temp_epoch)
    #break

    for images, targets in train_loader:
        model.train()

        image_batch = images.to(device)
        targets_off_device = targets
        target_batch = [{'boxes':dictionary['boxes'].to(device), 'labels':dictionary['labels'].to(device), 'masks':dictionary['masks'].to(device)} for dictionary in targets]

        optimizer.zero_grad()

        outputs = model(image_batch, target_batch)

        loss_classifier = outputs['loss_classifier']
        loss_box_reg = outputs['loss_box_reg']
        loss_objectness = outputs['loss_objectness']
        loss_rpn_box_reg = outputs['loss_rpn_box_reg']
        loss_mask_reg = outputs['loss_mask']

        losses_classifier.append(loss_classifier.item())
        losses_box_reg.append(loss_box_reg.item())
        losses_objectness.append(loss_objectness.item())
        losses_rpn_box_reg.append(loss_rpn_box_reg.item())
        losses_mask_reg.append(loss_mask_reg.item())

        combined_loss = loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg + loss_mask_reg

        combined_loss.backward()
        optimizer.step()
        if(counter_train % 10 == 0):
          print(counter_train*25)
        counter_train = counter_train + 1
    
    avg_clss_loss = np.mean(losses_classifier)
    avg_box_reg_loss = np.mean(losses_box_reg)
    avg_objectness_loss = np.mean(losses_objectness)
    avg_rpn_box_reg_loss = np.mean(losses_rpn_box_reg)
    avg_mask_reg_loss = np.mean(losses_mask_reg)

    epoch_losses_classifier.append(avg_clss_loss)
    epoch_losses_box_reg.append(avg_box_reg_loss)
    epoch_losses_objectness.append(avg_objectness_loss)
    epoch_losses_rpn_box_reg.append(avg_rpn_box_reg_loss)
    epoch_losses_mask_reg.append(avg_mask_reg_loss)

    print('Training Negative Log-Likelihood Classification Loss in Epoch ', temp_epoch, ': ', avg_clss_loss)
    print('Training Final Regression Loss for predicted Bounding Box Coordinates in Epoch ', temp_epoch, ': ', avg_box_reg_loss)
    print('Training Binary Cross-Entropy Loss for Object/Not Object for RPN Network in Epoch ', temp_epoch, ': ', avg_objectness_loss)
    print('Training Regression Loss for RPN Network Box Coordinates in Epoch ', temp_epoch, ': ', avg_rpn_box_reg_loss)
    print('Training Regression Loss for Mask Polygon Coordinates in Epoch ', temp_epoch, ': ', avg_mask_reg_loss)
    print('\n')

    torch.save({
            'epoch': temp_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss_classifier': loss_classifier,
            'loss_box_reg': loss_box_reg,
            'loss_objectness': loss_objectness,
            'loss_rpn_box_reg': loss_rpn_box_reg,
            'loss_mask': loss_mask_reg
            }, PATH)

    model.eval()
    with torch.no_grad():
      counter_eval = 0
      TP_epoch = 0
      FP_epoch = 0
      TN_epoch = 0
      FN_epoch = 0
      for images, targets in train_loader:
        image_batch = images.to(device)
        targets_off_device = targets
        target_batch = [{'boxes':dictionary['boxes'].to(device), 'labels':dictionary['labels'].to(device), 'masks':dictionary['masks'].to(device)} for dictionary in targets]

        outputs = model(image_batch, target_batch)

        # for i in range(len(targets_off_device)):
        #   mask_pred = list((outputs[i])['masks'])
        #   mask_actual = list((targets_off_device[i])['masks'])
              
        #   print("mask_pred length:", len(mask_pred))
        #   print("mask_actual length:", len(mask_actual))
        # continue

        for i in range(len(targets_off_device)):
          mask_pred = list((outputs[i])['masks'])
          mask_actual = list((targets_off_device[i])['masks'])
          mask_pred_arrs = []
          mask_actual_arrs = []

          if(len(mask_pred) == 0):
            temp_mask_pred = np.zeros((224,224))
            mask_pred_arrs.append(temp_mask_pred)
          else:
            for j in range(len(mask_pred)):
              temp_pred = ((mask_pred[j].detach().cpu())[0]).numpy()
              temp_locs_pred = np.where(temp_pred > 0.5)
              temp_mask_pred = np.zeros((224,224))
              temp_mask_pred[temp_locs_pred[0], temp_locs_pred[1]] = 1
              mask_pred_arrs.append(temp_mask_pred)

          if(len(mask_actual) == 0):
            temp_mask_actual = np.zeros((224,224))
            mask_actual_arrs.append(temp_mask_pred)  
          else:
            for j in range(len(mask_actual)):
              temp_actual = mask_actual[j].numpy()
              temp_locs_actual = np.where(temp_actual == 1)
              temp_mask_actual = np.zeros((224,224))
              temp_mask_actual[temp_locs_actual[0], temp_locs_actual[1]] = 1
              mask_actual_arrs.append(temp_mask_actual)
          
          combined_mask_pred = combine_masks(mask_pred_arrs)
          combined_mask_actual = combine_masks(mask_actual_arrs)


          TP, FP, TN, FN = confusion_matrix_elements(combined_mask_pred, combined_mask_actual)

          TP_epoch += TP
          FP_epoch += FP
          TN_epoch += TN
          FN_epoch += FN
        counter_eval += 1
        if(counter_eval*batch_size >= 500):
          break
    acc = accuracy(TP_epoch, FP_epoch, TN_epoch, FN_epoch)
    prec = precision(TP_epoch, FP_epoch)
    rec = recall(TP_epoch, FN_epoch)
    f1 = f1_score(prec, rec)
    dice = dice_coefficient(TP_epoch, FP_epoch, FN_epoch)
    intersection_over_union = iou(TP_epoch, FP_epoch, FN_epoch)

    # VALIDATE
    model.eval()
    acc_val, prec_val, rec_val, f1_val, dice_val, intersection_over_union_val = evaluate(model, val_loader, device, temp_epoch)
    print('Training Mask Accuracy in Epoch ', temp_epoch, ': ', acc)
    print('Training Mask Precision in Epoch ', temp_epoch, ': ', prec)
    print('Training Mask Recall in Epoch ', temp_epoch, ': ', rec)
    print('Training Mask F1-Score in Epoch ', temp_epoch, ': ', f1)
    print('Training DICE Score in Epoch ', temp_epoch, ': ', dice)
    print('Training Intersection Over Union(IoU) Score in Epoch ', temp_epoch, ': ', intersection_over_union)
    print('\n')
    print('Validation Mask Accuracy in Epoch ', temp_epoch, ': ', acc_val)
    print('Validation Mask Precision in Epoch ', temp_epoch, ': ', prec_val)
    print('Validation Mask Recall in Epoch ', temp_epoch, ': ', rec_val)
    print('Validation Mask F1-Score in Epoch ', temp_epoch, ': ', f1_val)
    print('Validation DICE Score in Epoch ', temp_epoch, ': ', dice_val)
    print('Validation Intersection Over Union(IoU) Score in Epoch ', temp_epoch, ': ', intersection_over_union_val)
    print('\n')

    acc_tr_epochs.append(acc)
    pre_tr_epochs.append(prec)
    rec_tr_epochs.append(rec)
    f1_tr_epochs.append(f1)
    dice_tr_epochs.append(dice)
    iou_tr_epochs.append(intersection_over_union)

    acc_val_epochs.append(acc_val)
    pre_val_epochs.append(prec_val)
    rec_val_epochs.append(rec_val)
    f1_val_epochs.append(f1_val)
    dice_val_epochs.append(dice_val)
    iou_val_epochs.append(intersection_over_union_val)

    scheduler.step()

scores_dict = {}

scores_dict['acc_tr'] = acc_tr_epochs
scores_dict['pre_tr'] = pre_tr_epochs
scores_dict['rec_tr'] = rec_tr_epochs
scores_dict['f1_tr'] = f1_tr_epochs
scores_dict['dice_tr'] = dice_tr_epochs
scores_dict['iou_tr'] = iou_tr_epochs

scores_dict['acc_val'] = acc_val_epochs
scores_dict['pre_val'] = pre_val_epochs
scores_dict['rec_val'] = rec_val_epochs
scores_dict['f1_val'] = f1_val_epochs
scores_dict['dice_val'] = dice_val_epochs
scores_dict['iou_val'] = iou_val_epochs
    

plotter(epoch_losses_classifier, epoch_losses_box_reg, epoch_losses_objectness, epoch_losses_rpn_box_reg, epoch_losses_mask_reg)
print('\n')
plotter_2(scores_dict)


# In[ ]:


plotter(epoch_losses_classifier, epoch_losses_box_reg, epoch_losses_objectness, epoch_losses_rpn_box_reg, epoch_losses_mask_reg)
print('\n')
plotter_2(scores_dict)


# # Hyper setting #2

# In[ ]:


learning_rate = 1e-3
l2_reg = 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=15, gamma=0.1)

num_epochs = 50
#num_epochs = 2

epoch_losses_classifier = []
epoch_losses_box_reg = []
epoch_losses_objectness = []
epoch_losses_rpn_box_reg = []
epoch_losses_mask_reg = []

acc_tr_epochs = []
pre_tr_epochs = []
rec_tr_epochs = []
f1_tr_epochs = []
dice_tr_epochs = []
iou_tr_epochs = []

acc_val_epochs = []
pre_val_epochs = []
rec_val_epochs = []
f1_val_epochs = []
dice_val_epochs = []
iou_val_epochs = []


for epoch in range(num_epochs):
    losses_classifier = []
    losses_box_reg = []
    losses_objectness = []
    losses_rpn_box_reg = []
    losses_mask_reg = []
    counter_train = 0
    temp_epoch = epoch + 1
    
    PATH = '/content/drive/MyDrive/ee641/Project_Datasets/Model_Checkpoints_2/model_epoch_' + str(temp_epoch) + '.pth'
    
    #acc, prec, rec, f1, dice, intersection_over_union = evaluate(model, val_loader, device, temp_epoch)
    #break

    for images, targets in train_loader:
        model.train()

        image_batch = images.to(device)
        targets_off_device = targets
        target_batch = [{'boxes':dictionary['boxes'].to(device), 'labels':dictionary['labels'].to(device), 'masks':dictionary['masks'].to(device)} for dictionary in targets]

        optimizer.zero_grad()

        outputs = model(image_batch, target_batch)

        loss_classifier = outputs['loss_classifier']
        loss_box_reg = outputs['loss_box_reg']
        loss_objectness = outputs['loss_objectness']
        loss_rpn_box_reg = outputs['loss_rpn_box_reg']
        loss_mask_reg = outputs['loss_mask']

        losses_classifier.append(loss_classifier.item())
        losses_box_reg.append(loss_box_reg.item())
        losses_objectness.append(loss_objectness.item())
        losses_rpn_box_reg.append(loss_rpn_box_reg.item())
        losses_mask_reg.append(loss_mask_reg.item())

        combined_loss = loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg + loss_mask_reg

        combined_loss.backward()
        optimizer.step()
        if(counter_train % 10 == 0):
          print(counter_train*25)
        counter_train = counter_train + 1
    
    avg_clss_loss = np.mean(losses_classifier)
    avg_box_reg_loss = np.mean(losses_box_reg)
    avg_objectness_loss = np.mean(losses_objectness)
    avg_rpn_box_reg_loss = np.mean(losses_rpn_box_reg)
    avg_mask_reg_loss = np.mean(losses_mask_reg)

    epoch_losses_classifier.append(avg_clss_loss)
    epoch_losses_box_reg.append(avg_box_reg_loss)
    epoch_losses_objectness.append(avg_objectness_loss)
    epoch_losses_rpn_box_reg.append(avg_rpn_box_reg_loss)
    epoch_losses_mask_reg.append(avg_mask_reg_loss)

    print('Training Negative Log-Likelihood Classification Loss in Epoch ', temp_epoch, ': ', avg_clss_loss)
    print('Training Final Regression Loss for predicted Bounding Box Coordinates in Epoch ', temp_epoch, ': ', avg_box_reg_loss)
    print('Training Binary Cross-Entropy Loss for Object/Not Object for RPN Network in Epoch ', temp_epoch, ': ', avg_objectness_loss)
    print('Training Regression Loss for RPN Network Box Coordinates in Epoch ', temp_epoch, ': ', avg_rpn_box_reg_loss)
    print('Training Regression Loss for Mask Polygon Coordinates in Epoch ', temp_epoch, ': ', avg_mask_reg_loss)
    print('\n')

    torch.save({
            'epoch': temp_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss_classifier': loss_classifier,
            'loss_box_reg': loss_box_reg,
            'loss_objectness': loss_objectness,
            'loss_rpn_box_reg': loss_rpn_box_reg,
            'loss_mask': loss_mask_reg
            }, PATH)

    model.eval()
    with torch.no_grad():
      counter_eval = 0
      TP_epoch = 0
      FP_epoch = 0
      TN_epoch = 0
      FN_epoch = 0
      for images, targets in train_loader:
        image_batch = images.to(device)
        targets_off_device = targets
        target_batch = [{'boxes':dictionary['boxes'].to(device), 'labels':dictionary['labels'].to(device), 'masks':dictionary['masks'].to(device)} for dictionary in targets]

        outputs = model(image_batch, target_batch)

        # for i in range(len(targets_off_device)):
        #   mask_pred = list((outputs[i])['masks'])
        #   mask_actual = list((targets_off_device[i])['masks'])
              
        #   print("mask_pred length:", len(mask_pred))
        #   print("mask_actual length:", len(mask_actual))
        # continue

        for i in range(len(targets_off_device)):
          mask_pred = list((outputs[i])['masks'])
          mask_actual = list((targets_off_device[i])['masks'])
          mask_pred_arrs = []
          mask_actual_arrs = []

          if(len(mask_pred) == 0):
            temp_mask_pred = np.zeros((224,224))
            mask_pred_arrs.append(temp_mask_pred)
          else:
            for j in range(len(mask_pred)):
              temp_pred = ((mask_pred[j].detach().cpu())[0]).numpy()
              temp_locs_pred = np.where(temp_pred > 0.5)
              temp_mask_pred = np.zeros((224,224))
              temp_mask_pred[temp_locs_pred[0], temp_locs_pred[1]] = 1
              mask_pred_arrs.append(temp_mask_pred)

          if(len(mask_actual) == 0):
            temp_mask_actual = np.zeros((224,224))
            mask_actual_arrs.append(temp_mask_pred)  
          else:
            for j in range(len(mask_actual)):
              temp_actual = mask_actual[j].numpy()
              temp_locs_actual = np.where(temp_actual == 1)
              temp_mask_actual = np.zeros((224,224))
              temp_mask_actual[temp_locs_actual[0], temp_locs_actual[1]] = 1
              mask_actual_arrs.append(temp_mask_actual)
          
          combined_mask_pred = combine_masks(mask_pred_arrs)
          combined_mask_actual = combine_masks(mask_actual_arrs)


          TP, FP, TN, FN = confusion_matrix_elements(combined_mask_pred, combined_mask_actual)

          TP_epoch += TP
          FP_epoch += FP
          TN_epoch += TN
          FN_epoch += FN
        counter_eval += 1
        if(counter_eval*batch_size >= 750):
          break
    acc = accuracy(TP_epoch, FP_epoch, TN_epoch, FN_epoch)
    prec = precision(TP_epoch, FP_epoch)
    rec = recall(TP_epoch, FN_epoch)
    f1 = f1_score(prec, rec)
    dice = dice_coefficient(TP_epoch, FP_epoch, FN_epoch)
    intersection_over_union = iou(TP_epoch, FP_epoch, FN_epoch)

    # VALIDATE
    model.eval()
    acc_val, prec_val, rec_val, f1_val, dice_val, intersection_over_union_val = evaluate(model, val_loader, device, temp_epoch)
    print('Training Mask Accuracy in Epoch ', temp_epoch, ': ', acc)
    print('Training Mask Precision in Epoch ', temp_epoch, ': ', prec)
    print('Training Mask Recall in Epoch ', temp_epoch, ': ', rec)
    print('Training Mask F1-Score in Epoch ', temp_epoch, ': ', f1)
    print('Training DICE Score in Epoch ', temp_epoch, ': ', dice)
    print('Training Intersection Over Union(IoU) Score in Epoch ', temp_epoch, ': ', intersection_over_union)
    print('\n')
    print('Validation Mask Accuracy in Epoch ', temp_epoch, ': ', acc_val)
    print('Validation Mask Precision in Epoch ', temp_epoch, ': ', prec_val)
    print('Validation Mask Recall in Epoch ', temp_epoch, ': ', rec_val)
    print('Validation Mask F1-Score in Epoch ', temp_epoch, ': ', f1_val)
    print('Validation DICE Score in Epoch ', temp_epoch, ': ', dice_val)
    print('Validation Intersection Over Union(IoU) Score in Epoch ', temp_epoch, ': ', intersection_over_union_val)
    print('\n')

    acc_tr_epochs.append(acc)
    pre_tr_epochs.append(prec)
    rec_tr_epochs.append(rec)
    f1_tr_epochs.append(f1)
    dice_tr_epochs.append(dice)
    iou_tr_epochs.append(intersection_over_union)

    acc_val_epochs.append(acc_val)
    pre_val_epochs.append(prec_val)
    rec_val_epochs.append(rec_val)
    f1_val_epochs.append(f1_val)
    dice_val_epochs.append(dice_val)
    iou_val_epochs.append(intersection_over_union_val)

    scheduler.step()

scores_dict = {}

scores_dict['acc_tr'] = acc_tr_epochs
scores_dict['pre_tr'] = pre_tr_epochs
scores_dict['rec_tr'] = rec_tr_epochs
scores_dict['f1_tr'] = f1_tr_epochs
scores_dict['dice_tr'] = dice_tr_epochs
scores_dict['iou_tr'] = iou_tr_epochs

scores_dict['acc_val'] = acc_val_epochs
scores_dict['pre_val'] = pre_val_epochs
scores_dict['rec_val'] = rec_val_epochs
scores_dict['f1_val'] = f1_val_epochs
scores_dict['dice_val'] = dice_val_epochs
scores_dict['iou_val'] = iou_val_epochs
    

plotter(epoch_losses_classifier, epoch_losses_box_reg, epoch_losses_objectness, epoch_losses_rpn_box_reg, epoch_losses_mask_reg)
print('\n')
plotter_2(scores_dict)


# # Hyper setting #3

# In[ ]:


learning_rate = 1e-2
l2_reg = 5 * 1e-5
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9,  weight_decay=l2_reg)
#optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)

num_epochs = 50
#num_epochs = 2

epoch_losses_classifier = []
epoch_losses_box_reg = []
epoch_losses_objectness = []
epoch_losses_rpn_box_reg = []
epoch_losses_mask_reg = []

acc_tr_epochs = []
pre_tr_epochs = []
rec_tr_epochs = []
f1_tr_epochs = []
dice_tr_epochs = []
iou_tr_epochs = []

acc_val_epochs = []
pre_val_epochs = []
rec_val_epochs = []
f1_val_epochs = []
dice_val_epochs = []
iou_val_epochs = []


for epoch in range(num_epochs):
    losses_classifier = []
    losses_box_reg = []
    losses_objectness = []
    losses_rpn_box_reg = []
    losses_mask_reg = []
    counter_train = 0
    temp_epoch = epoch + 1
    
    PATH = '/content/drive/MyDrive/ee641/Project_Datasets/Model_Checkpoints_3/model_epoch_' + str(temp_epoch) + '.pth'
    
    #acc, prec, rec, f1, dice, intersection_over_union = evaluate(model, val_loader, device, temp_epoch)
    #break

    for images, targets in train_loader:
        model.train()

        image_batch = images.to(device)
        targets_off_device = targets
        target_batch = [{'boxes':dictionary['boxes'].to(device), 'labels':dictionary['labels'].to(device), 'masks':dictionary['masks'].to(device)} for dictionary in targets]

        optimizer.zero_grad()

        outputs = model(image_batch, target_batch)

        loss_classifier = outputs['loss_classifier']
        loss_box_reg = outputs['loss_box_reg']
        loss_objectness = outputs['loss_objectness']
        loss_rpn_box_reg = outputs['loss_rpn_box_reg']
        loss_mask_reg = outputs['loss_mask']

        losses_classifier.append(loss_classifier.item())
        losses_box_reg.append(loss_box_reg.item())
        losses_objectness.append(loss_objectness.item())
        losses_rpn_box_reg.append(loss_rpn_box_reg.item())
        losses_mask_reg.append(loss_mask_reg.item())

        combined_loss = loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg + loss_mask_reg

        combined_loss.backward()
        optimizer.step()
        if(counter_train % 10 == 0):
          print(counter_train*25)
        counter_train = counter_train + 1
    
    avg_clss_loss = np.mean(losses_classifier)
    avg_box_reg_loss = np.mean(losses_box_reg)
    avg_objectness_loss = np.mean(losses_objectness)
    avg_rpn_box_reg_loss = np.mean(losses_rpn_box_reg)
    avg_mask_reg_loss = np.mean(losses_mask_reg)

    epoch_losses_classifier.append(avg_clss_loss)
    epoch_losses_box_reg.append(avg_box_reg_loss)
    epoch_losses_objectness.append(avg_objectness_loss)
    epoch_losses_rpn_box_reg.append(avg_rpn_box_reg_loss)
    epoch_losses_mask_reg.append(avg_mask_reg_loss)

    print('Training Negative Log-Likelihood Classification Loss in Epoch ', temp_epoch, ': ', avg_clss_loss)
    print('Training Final Regression Loss for predicted Bounding Box Coordinates in Epoch ', temp_epoch, ': ', avg_box_reg_loss)
    print('Training Binary Cross-Entropy Loss for Object/Not Object for RPN Network in Epoch ', temp_epoch, ': ', avg_objectness_loss)
    print('Training Regression Loss for RPN Network Box Coordinates in Epoch ', temp_epoch, ': ', avg_rpn_box_reg_loss)
    print('Training Regression Loss for Mask Polygon Coordinates in Epoch ', temp_epoch, ': ', avg_mask_reg_loss)
    print('\n')

    torch.save({
            'epoch': temp_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss_classifier': loss_classifier,
            'loss_box_reg': loss_box_reg,
            'loss_objectness': loss_objectness,
            'loss_rpn_box_reg': loss_rpn_box_reg,
            'loss_mask': loss_mask_reg
            }, PATH)

    model.eval()
    with torch.no_grad():
      counter_eval = 0
      TP_epoch = 0
      FP_epoch = 0
      TN_epoch = 0
      FN_epoch = 0
      for images, targets in train_loader:
        image_batch = images.to(device)
        targets_off_device = targets
        target_batch = [{'boxes':dictionary['boxes'].to(device), 'labels':dictionary['labels'].to(device), 'masks':dictionary['masks'].to(device)} for dictionary in targets]

        outputs = model(image_batch, target_batch)

        # for i in range(len(targets_off_device)):
        #   mask_pred = list((outputs[i])['masks'])
        #   mask_actual = list((targets_off_device[i])['masks'])
              
        #   print("mask_pred length:", len(mask_pred))
        #   print("mask_actual length:", len(mask_actual))
        # continue

        for i in range(len(targets_off_device)):
          mask_pred = list((outputs[i])['masks'])
          mask_actual = list((targets_off_device[i])['masks'])
          mask_pred_arrs = []
          mask_actual_arrs = []

          if(len(mask_pred) == 0):
            temp_mask_pred = np.zeros((224,224))
            mask_pred_arrs.append(temp_mask_pred)
          else:
            for j in range(len(mask_pred)):
              temp_pred = ((mask_pred[j].detach().cpu())[0]).numpy()
              temp_locs_pred = np.where(temp_pred > 0.5)
              temp_mask_pred = np.zeros((224,224))
              temp_mask_pred[temp_locs_pred[0], temp_locs_pred[1]] = 1
              mask_pred_arrs.append(temp_mask_pred)

          if(len(mask_actual) == 0):
            temp_mask_actual = np.zeros((224,224))
            mask_actual_arrs.append(temp_mask_pred)  
          else:
            for j in range(len(mask_actual)):
              temp_actual = mask_actual[j].numpy()
              temp_locs_actual = np.where(temp_actual == 1)
              temp_mask_actual = np.zeros((224,224))
              temp_mask_actual[temp_locs_actual[0], temp_locs_actual[1]] = 1
              mask_actual_arrs.append(temp_mask_actual)
          
          combined_mask_pred = combine_masks(mask_pred_arrs)
          combined_mask_actual = combine_masks(mask_actual_arrs)


          TP, FP, TN, FN = confusion_matrix_elements(combined_mask_pred, combined_mask_actual)

          TP_epoch += TP
          FP_epoch += FP
          TN_epoch += TN
          FN_epoch += FN
        counter_eval += 1
        if(counter_eval*batch_size >= 750):
          break
    acc = accuracy(TP_epoch, FP_epoch, TN_epoch, FN_epoch)
    prec = precision(TP_epoch, FP_epoch)
    rec = recall(TP_epoch, FN_epoch)
    f1 = f1_score(prec, rec)
    dice = dice_coefficient(TP_epoch, FP_epoch, FN_epoch)
    intersection_over_union = iou(TP_epoch, FP_epoch, FN_epoch)

    # VALIDATE
    model.eval()
    acc_val, prec_val, rec_val, f1_val, dice_val, intersection_over_union_val = evaluate(model, val_loader, device, temp_epoch)
    print('Training Mask Accuracy in Epoch ', temp_epoch, ': ', acc)
    print('Training Mask Precision in Epoch ', temp_epoch, ': ', prec)
    print('Training Mask Recall in Epoch ', temp_epoch, ': ', rec)
    print('Training Mask F1-Score in Epoch ', temp_epoch, ': ', f1)
    print('Training DICE Score in Epoch ', temp_epoch, ': ', dice)
    print('Training Intersection Over Union(IoU) Score in Epoch ', temp_epoch, ': ', intersection_over_union)
    print('\n')
    print('Validation Mask Accuracy in Epoch ', temp_epoch, ': ', acc_val)
    print('Validation Mask Precision in Epoch ', temp_epoch, ': ', prec_val)
    print('Validation Mask Recall in Epoch ', temp_epoch, ': ', rec_val)
    print('Validation Mask F1-Score in Epoch ', temp_epoch, ': ', f1_val)
    print('Validation DICE Score in Epoch ', temp_epoch, ': ', dice_val)
    print('Validation Intersection Over Union(IoU) Score in Epoch ', temp_epoch, ': ', intersection_over_union_val)
    print('\n')

    acc_tr_epochs.append(acc)
    pre_tr_epochs.append(prec)
    rec_tr_epochs.append(rec)
    f1_tr_epochs.append(f1)
    dice_tr_epochs.append(dice)
    iou_tr_epochs.append(intersection_over_union)

    acc_val_epochs.append(acc_val)
    pre_val_epochs.append(prec_val)
    rec_val_epochs.append(rec_val)
    f1_val_epochs.append(f1_val)
    dice_val_epochs.append(dice_val)
    iou_val_epochs.append(intersection_over_union_val)

    scheduler.step()

scores_dict = {}

scores_dict['acc_tr'] = acc_tr_epochs
scores_dict['pre_tr'] = pre_tr_epochs
scores_dict['rec_tr'] = rec_tr_epochs
scores_dict['f1_tr'] = f1_tr_epochs
scores_dict['dice_tr'] = dice_tr_epochs
scores_dict['iou_tr'] = iou_tr_epochs

scores_dict['acc_val'] = acc_val_epochs
scores_dict['pre_val'] = pre_val_epochs
scores_dict['rec_val'] = rec_val_epochs
scores_dict['f1_val'] = f1_val_epochs
scores_dict['dice_val'] = dice_val_epochs
scores_dict['iou_val'] = iou_val_epochs
    

plotter(epoch_losses_classifier, epoch_losses_box_reg, epoch_losses_objectness, epoch_losses_rpn_box_reg, epoch_losses_mask_reg)
print('\n')
plotter_2(scores_dict)


# In[ ]:


import gc
gc.collect()
torch.cuda.empty_cache()


# In[ ]:


torch.cuda.empty_cache()


# # Masking Strategies Performance 

# In[ ]:


acc_tr_epochs_05 = []
pre_tr_epochs_05 = []
rec_tr_epochs_05 = []
f1_tr_epochs_05 = []
dice_tr_epochs_05 = []
iou_tr_epochs_05 = []

acc_val_epochs_05 = []
pre_val_epochs_05 = []
rec_val_epochs_05 = []
f1_val_epochs_05 = []
dice_val_epochs_05 = []
iou_val_epochs_05 = []

acc_tr_epochs_06 = []
pre_tr_epochs_06 = []
rec_tr_epochs_06 = []
f1_tr_epochs_06 = []
dice_tr_epochs_06 = []
iou_tr_epochs_06 = []

acc_val_epochs_06 = []
pre_val_epochs_06 = []
rec_val_epochs_06 = []
f1_val_epochs_06 = []
dice_val_epochs_06 = []
iou_val_epochs_06 = []

acc_tr_epochs_07 = []
pre_tr_epochs_07 = []
rec_tr_epochs_07 = []
f1_tr_epochs_07 = []
dice_tr_epochs_07 = []
iou_tr_epochs_07 = []

acc_val_epochs_07 = []
pre_val_epochs_07 = []
rec_val_epochs_07 = []
f1_val_epochs_07 = []
dice_val_epochs_07 = []
iou_val_epochs_07 = []

acc_tr_epochs_08 = []
pre_tr_epochs_08 = []
rec_tr_epochs_08 = []
f1_tr_epochs_08 = []
dice_tr_epochs_08 = []
iou_tr_epochs_08 = []

acc_val_epochs_08 = []
pre_val_epochs_08 = []
rec_val_epochs_08 = []
f1_val_epochs_08 = []
dice_val_epochs_08 = []
iou_val_epochs_08 = []

acc_tr_epochs_otsu = []
pre_tr_epochs_otsu = []
rec_tr_epochs_otsu = []
f1_tr_epochs_otsu = []
dice_tr_epochs_otsu = []
iou_tr_epochs_otsu = []

acc_val_epochs_otsu = []
pre_val_epochs_otsu = []
rec_val_epochs_otsu = []
f1_val_epochs_otsu = []
dice_val_epochs_otsu = []
iou_val_epochs_otsu = []


# In[ ]:


a = np.array([[1,2], [3,4]])
b = np.array([[1,1], [3,2]])
c = np.zeros((2,2))
c[a > b] = 1
print(c)


# In[ ]:


import cv2
import numpy as np

# Assume 'mask_prob' is the predicted probability map output from your model.
# It should be a 2D numpy array.
mask_prob = np.array([[0.45, 0.2, 0.15], [0.1, 0.75, 0.24]])
# Convert probabilities to a grayscale image in range [0, 255]
mask_prob_uint8 = (mask_prob * 255).astype(np.uint8)
print(mask_prob_uint8)

# Apply Otsu's thresholding
_, mask_otsu = cv2.threshold(mask_prob_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(mask_otsu)
# Convert back to [0, 1] range if needed
mask_otsu = mask_otsu.astype(np.uint8) / 255
print(mask_otsu)


# In[ ]:


from matplotlib.patches import Rectangle, Polygon
def evaluate_masking(model, val_loader, device, epoch, mode=None, threshold = None):
  # mean_chest = [0.5240804197105211, 0.5240804197105211, 0.5240804197105211]
  # std_chest = [0.17148634738092733, 0.17148634738092733, 0.17148634738092733]
  mean_chest = [0.485, 0.456, 0.406]
  std_chest = [0.229, 0.224, 0.225]
  #[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

  TP_epoch = 0
  FP_epoch = 0
  TN_epoch = 0
  FN_epoch = 0
  counter_for_visuals = True 

  with torch.no_grad():
    for images, targets in val_loader:
      model.eval()
      image_batch = images.to(device)
      #target_batch = [{'boxes':dictionary['boxes'].to(device), 'labels':dictionary['labels'].to(device), 'masks':dictionary['masks'].to(device)} for dictionary in targets]
      predictions = model(image_batch)
      #indices_of_images = [0,1,2,3,4,5,6,7]

      if((counter_for_visuals == True) and (epoch % 5 == 0)):
        counter_row = 0
        counter_column = 0
        #fig, ax = plt.subplots(2, 4, figsize=(48, 24))
        fig_m, ax_m = plt.subplots(2, 4, figsize=(48, 24))
        #indices_of_images = random.sample(range(0, 15), 8)
        indices_of_images = [0,2,4,6,8,10,12,14]

        for img_trg_index in indices_of_images:
          #box_pred = list(predictions[img_trg_index]['boxes'])
          #box_actual = list(targets[img_trg_index]['boxes'])
          mask_pred = list(predictions[img_trg_index]['masks'])
          mask_actual = list(targets[img_trg_index]['masks'])
          mask_actual_polys = list(targets[img_trg_index]['polys'])

          actual_image = images[img_trg_index]
          image_numpy = (actual_image.detach().cpu()).numpy()
          actual_denormalized_image = denormalize_image(image_numpy, mean_chest, std_chest)

          #ax[counter_row, counter_column].imshow(actual_denormalized_image)
          ax_m[counter_row, counter_column].imshow(actual_denormalized_image)

          mask_actual_arrs = []
          
          for i in mask_actual:
            tmp = i.numpy()
            temp_locs = np.where(tmp == 1)
            temp_mask = np.zeros((224,224))
            temp_mask[temp_locs[0], temp_locs[1]] = 1
            mask_actual_arrs.append(temp_mask)

          combined_mask_actual = combine_masks(mask_actual_arrs)
          actual_mask_color = np.array([0, 0, 1, 0.4]) #Blue
          actual_mask_rgb = np.zeros((224, 224, 4))
          actual_mask_rgb[combined_mask_actual == 1] = actual_mask_color
          ax_m[counter_row, counter_column].imshow(actual_mask_rgb)

          mask_pred_arrs = []
          
          for i in mask_pred:
            tmp = i.detach().cpu()
            tmp2 = (tmp[0]).numpy()
            temp_mask = np.zeros((224, 224))
            if(mode == 'Threshold'):
              temp_locs = np.where(tmp2 > threshold)
              temp_mask[temp_locs[0], temp_locs[1]] = 1
            else:
              mask_prob_uint8 = (tmp2 * 255).astype(np.uint8)
              _, mask_otsu = cv2.threshold(mask_prob_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
              mask_otsu = mask_otsu.astype(np.uint8) / 255
              temp_locs = np.where(mask_otsu == 1)
              temp_mask[temp_locs[0], temp_locs[1]] = 1
            mask_pred_arrs.append(temp_mask)
            
          combined_mask_predicted = combine_masks(mask_pred_arrs)
          predicted_mask_color = np.array([1, 1, 0, 0.2]) #Yellow
          predicted_mask_rgb = np.zeros((224, 224, 4))
          predicted_mask_rgb[combined_mask_predicted == 1] = predicted_mask_color
          ax_m[counter_row, counter_column].imshow(predicted_mask_rgb)

          # for j in box_actual:
          #   tmp = j.numpy()
          #   tmp_rect = Rectangle((tmp[0], tmp[1]), tmp[2] - tmp[0], tmp[3] - tmp[1], linewidth=2, edgecolor='r', facecolor='none')
          #   #tmp_rect = Rectangle((tmp[0], tmp[3]), tmp[2] - tmp[0], tmp[3] - tmp[1], linewidth=2, edgecolor='r', facecolor='none')
          #   ax[counter_row, counter_column].add_patch(tmp_rect)
          # for i in box_pred:
          #   tmp = i.detach().cpu().numpy()
          #   tmp_rect = Rectangle((tmp[0], tmp[1]), tmp[2] - tmp[0], tmp[3] - tmp[1], linewidth=1, edgecolor='g', facecolor='none')
          #   #tmp_rect = Rectangle((tmp[0], tmp[3]), tmp[2] - tmp[0], tmp[3] - tmp[1], linewidth=1, edgecolor='b', facecolor='none')
          #   ax[counter_row, counter_column].add_patch(tmp_rect)

          counter_column += 1
          if(counter_column > 3):
            counter_row += 1
            counter_column = 0

        #image_dir = '/content/Boxes_Images_Resized/'
        #image_dir = '/content/drive/MyDrive/ee641/Project_Datasets/Masked_Images_through_Epochs/'
        if(mode == 'Otsu'):
          image_dir = '/content/drive/MyDrive/ee641/Project_Datasets/Masking_Strategies_Best_Model_Otsu_Threshold/'
          temp_title_m = 'Otsu Thresholding Epoch ' + str(epoch) + ' Mask Predictions (Yellow) vs. Actual Masks(Blue)'
        else:
          if(threshold == 0.5):
            image_dir = '/content/drive/MyDrive/ee641/Project_Datasets/Masking_Strategies_Best_Model_05_Threshold/'
            temp_title_m = '0.5 Thresholding Epoch ' + str(epoch) + ' Mask Predictions (Yellow) vs. Actual Masks(Blue)'
          elif(threshold == 0.6):
            image_dir = '/content/drive/MyDrive/ee641/Project_Datasets/Masking_Strategies_Best_Model_06_Threshold/'
            temp_title_m = '0.6 Thresholding Epoch ' + str(epoch) + ' Mask Predictions (Yellow) vs. Actual Masks(Blue)'
          elif(threshold == 0.7):
            image_dir = '/content/drive/MyDrive/ee641/Project_Datasets/Masking_Strategies_Best_Model_07_Threshold/'
            temp_title_m = '0.7 Thresholding Epoch ' + str(epoch) + ' Mask Predictions (Yellow) vs. Actual Masks(Blue)'
          else:
            image_dir = '/content/drive/MyDrive/ee641/Project_Datasets/Masking_Strategies_Best_Model_08_Threshold/'
            temp_title_m = '0.8 Thresholding Epoch ' + str(epoch) + ' Mask Predictions (Yellow) vs. Actual Masks(Blue)'

        #image_dir = '/content/drive/MyDrive/ee641/Project_Datasets/Images_Instance_Segmentation_3/'
        #temp_filename = image_dir + 'Epoch_' + str(epoch) + '_BBox_Preds_BLUE_BBox_Actual_RED.png'
        temp_filename_m = image_dir + 'Epoch_' + str(epoch) + '_Mask_Preds_YELLOW_Mask_Actual_BLUE.png'
        #temp_title = 'Epoch ' + str(epoch) + ' BBox Predictions (Blue) vs. Actual BBox(Red)'
        #temp_title_m = 'Epoch ' + str(epoch) + ' Mask Predictions (Yellow) vs. Actual Masks(Blue)'
        print(temp_title_m)
        #plt.title(temp_title)
        #fig.suptitle(temp_title)
        #plt.show()
        fig_m.suptitle(temp_title_m)
        plt.show()
        #fig.savefig(temp_filename)
        fig_m.savefig(temp_filename_m)

      counter_for_visuals = False

      for i in range(len(targets)):
        mask_pred = list(predictions[i]['masks'])
        mask_actual = list((targets[i])['masks'])
        mask_pred_arrs = []
        mask_actual_arrs = []
        #print(len(mask_pred))
        #print(len(mask_actual))
        #print(mask_pred[0].shape)
        #print(mask_actual[0].shape)
        if(len(mask_pred) == 0):
          temp_mask_pred = np.zeros((224,224))
          mask_pred_arrs.append(temp_mask_pred)
        else:
          for j in range(len(mask_pred)):
            tmp = mask_pred[j].detach().cpu()
            tmp2 = (tmp[0]).numpy()
            temp_mask = np.zeros((224, 224))
            if(mode == 'Threshold'):
              temp_locs = np.where(tmp2 > threshold)
              temp_mask[temp_locs[0], temp_locs[1]] = 1
            else:
              mask_prob_uint8 = (tmp2 * 255).astype(np.uint8)
              _, mask_otsu = cv2.threshold(mask_prob_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
              mask_otsu = mask_otsu.astype(np.uint8) / 255
              temp_locs = np.where(mask_otsu == 1)
              temp_mask[temp_locs[0], temp_locs[1]] = 1
            mask_pred_arrs.append(temp_mask)

        if(len(mask_actual) == 0):
          temp_mask_actual = np.zeros((224,224))
          mask_actual_arrs.append(temp_mask_actual)
        else:
          for j in range(len(mask_actual)):
            temp_actual = mask_actual[j].numpy()
            temp_locs_actual = np.where(temp_actual == 1)
            temp_mask_actual = np.zeros((224,224))
            temp_mask_actual[temp_locs_actual[0], temp_locs_actual[1]] = 1
            mask_actual_arrs.append(temp_mask_actual)

        #print('HERE')
        combined_mask_pred = combine_masks(mask_pred_arrs)
        #print(combined_mask_pred)
        #print(combined_mask_pred.shape)
        #print('HERE2')
        combined_mask_actual = combine_masks(mask_actual_arrs)
        #print(combined_mask_actual)
        #print(combined_mask_actual.shape)
        #return
        
        TP, FP, TN, FN = confusion_matrix_elements(combined_mask_pred, combined_mask_actual)

        TP_epoch += TP
        FP_epoch += FP
        TN_epoch += TN
        FN_epoch += FN
    
    acc = accuracy(TP_epoch, FP_epoch, TN_epoch, FN_epoch)
    prec = precision(TP_epoch, FP_epoch)
    rec = recall(TP_epoch, FN_epoch)
    f1 = f1_score(prec, rec)
    dice = dice_coefficient(TP_epoch, FP_epoch, FN_epoch)
    intersection_over_union = iou(TP_epoch, FP_epoch, FN_epoch)
    # print(acc)
    # print(prec)
    # print(rec)
    # print(f1)
    # print(dice)

  return acc, prec, rec, f1, dice, intersection_over_union


# In[ ]:


import gc
gc.collect()
torch.cuda.empty_cache()


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[ ]:


num_classes = 14
device = torch.device(device)
model = get_model_instance_segmentation(num_classes)
model = model.to(device)


# In[ ]:


modes = [0.5,0.6,0.7,0.8,'Otsu']


# In[ ]:


# num_epochs = 50
# iterations = np.arange(1,num_epochs + 1, 2)
# for i in iterations:
#   print(i)


# In[ ]:


learning_rate = 1e-2
l2_reg = 5 * 1e-5
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9,  weight_decay=l2_reg)
#optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)

num_epochs = 50
#num_epochs = 2
iterations = np.arange(1,num_epochs + 1, 2)

for epoch in iterations:
    #temp_epoch = epoch + 1
    temp_epoch = epoch
    
    PATH = '/content/drive/MyDrive/ee641/Project_Datasets/Model_Checkpoints_3/model_epoch_' + str(temp_epoch) + '.pth'
    
    #acc, prec, rec, f1, dice, intersection_over_union = evaluate(model, val_loader, device, temp_epoch)
    #break
    checkpoint = torch.load(PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    #model.load_state_dict(torch.load(PATH, map_location=device))
    for mode in modes:
      model.eval()
      with torch.no_grad():
        counter_eval = 0
        TP_epoch = 0
        FP_epoch = 0
        TN_epoch = 0
        FN_epoch = 0
        for images, targets in train_loader:
          image_batch = images.to(device)
          targets_off_device = targets
          target_batch = [{'boxes':dictionary['boxes'].to(device), 'labels':dictionary['labels'].to(device), 'masks':dictionary['masks'].to(device)} for dictionary in targets]

          outputs = model(image_batch)

          for i in range(len(targets_off_device)):
            mask_pred = list((outputs[i])['masks'])
            mask_actual = list((targets_off_device[i])['masks'])
            mask_pred_arrs = []
            mask_actual_arrs = []

            if(len(mask_pred) == 0):
              temp_mask_pred = np.zeros((224,224))
              mask_pred_arrs.append(temp_mask_pred)
            else:
              for j in range(len(mask_pred)):
                tmp = mask_pred[j].detach().cpu()
                tmp2 = (tmp[0]).numpy()
                temp_mask = np.zeros((224, 224))
                if(mode != 'Otsu'):
                  temp_locs = np.where(tmp2 > mode)
                  temp_mask[temp_locs[0], temp_locs[1]] = 1
                else:
                  mask_prob_uint8 = (tmp2 * 255).astype(np.uint8)
                  _, mask_otsu = cv2.threshold(mask_prob_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                  mask_otsu = mask_otsu.astype(np.uint8) / 255
                  temp_locs = np.where(mask_otsu == 1)
                  temp_mask[temp_locs[0], temp_locs[1]] = 1
                mask_pred_arrs.append(temp_mask)

            if(len(mask_actual) == 0):
              temp_mask_actual = np.zeros((224,224))
              mask_actual_arrs.append(temp_mask_actual)  
            else:
              for j in range(len(mask_actual)):
                temp_actual = mask_actual[j].numpy()
                temp_locs_actual = np.where(temp_actual == 1)
                temp_mask_actual = np.zeros((224,224))
                temp_mask_actual[temp_locs_actual[0], temp_locs_actual[1]] = 1
                mask_actual_arrs.append(temp_mask_actual)
            
            combined_mask_pred = combine_masks(mask_pred_arrs)
            combined_mask_actual = combine_masks(mask_actual_arrs)


            TP, FP, TN, FN = confusion_matrix_elements(combined_mask_pred, combined_mask_actual)

            TP_epoch += TP
            FP_epoch += FP
            TN_epoch += TN
            FN_epoch += FN
          counter_eval += 1
          if(counter_eval*batch_size >= 750):
            break
      acc = accuracy(TP_epoch, FP_epoch, TN_epoch, FN_epoch)
      prec = precision(TP_epoch, FP_epoch)
      rec = recall(TP_epoch, FN_epoch)
      f1 = f1_score(prec, rec)
      dice = dice_coefficient(TP_epoch, FP_epoch, FN_epoch)
      intersection_over_union = iou(TP_epoch, FP_epoch, FN_epoch)

      # VALIDATE
      model.eval()
      if(mode != 'Otsu'):
        acc_val, prec_val, rec_val, f1_val, dice_val, intersection_over_union_val = evaluate_masking(model, val_loader, device, temp_epoch, 'Threshold', mode)
      else:
        acc_val, prec_val, rec_val, f1_val, dice_val, intersection_over_union_val = evaluate_masking(model, val_loader, device, temp_epoch, 'Otsu', None)
      print(str(mode), 'Thresholding Training Mask Accuracy in Epoch ', temp_epoch, ': ', acc)
      print(str(mode), 'Thresholding Training Mask Precision in Epoch ', temp_epoch, ': ', prec)
      print(str(mode), 'Thresholding Training Mask Recall in Epoch ', temp_epoch, ': ', rec)
      print(str(mode), 'Thresholding Training Mask F1-Score in Epoch ', temp_epoch, ': ', f1)
      print(str(mode), 'Thresholding Training DICE Score in Epoch ', temp_epoch, ': ', dice)
      print(str(mode), 'Thresholding Training Intersection Over Union(IoU) Score in Epoch ', temp_epoch, ': ', intersection_over_union)
      print('\n')
      print(str(mode), 'Thresholding Validation Mask Accuracy in Epoch ', temp_epoch, ': ', acc_val)
      print(str(mode), 'Thresholding Validation Mask Precision in Epoch ', temp_epoch, ': ', prec_val)
      print(str(mode), 'Thresholding Validation Mask Recall in Epoch ', temp_epoch, ': ', rec_val)
      print(str(mode), 'Thresholding Validation Mask F1-Score in Epoch ', temp_epoch, ': ', f1_val)
      print(str(mode), 'Thresholding Validation DICE Score in Epoch ', temp_epoch, ': ', dice_val)
      print(str(mode), 'Thresholding Validation Intersection Over Union(IoU) Score in Epoch ', temp_epoch, ': ', intersection_over_union_val)
      print('\n')

      if(mode == 0.5):
        acc_tr_epochs_05.append(acc)
        pre_tr_epochs_05.append(prec)
        rec_tr_epochs_05.append(rec)
        f1_tr_epochs_05.append(f1)
        dice_tr_epochs_05.append(dice)
        iou_tr_epochs_05.append(intersection_over_union)

        acc_val_epochs_05.append(acc_val)
        pre_val_epochs_05.append(prec_val)
        rec_val_epochs_05.append(rec_val)
        f1_val_epochs_05.append(f1_val)
        dice_val_epochs_05.append(dice_val)
        iou_val_epochs_05.append(intersection_over_union_val)

      elif(mode == 0.6):
        acc_tr_epochs_06.append(acc)
        pre_tr_epochs_06.append(prec)
        rec_tr_epochs_06.append(rec)
        f1_tr_epochs_06.append(f1)
        dice_tr_epochs_06.append(dice)
        iou_tr_epochs_06.append(intersection_over_union)

        acc_val_epochs_06.append(acc_val)
        pre_val_epochs_06.append(prec_val)
        rec_val_epochs_06.append(rec_val)
        f1_val_epochs_06.append(f1_val)
        dice_val_epochs_06.append(dice_val)
        iou_val_epochs_06.append(intersection_over_union_val)

      elif(mode == 0.7):
        acc_tr_epochs_07.append(acc)
        pre_tr_epochs_07.append(prec)
        rec_tr_epochs_07.append(rec)
        f1_tr_epochs_07.append(f1)
        dice_tr_epochs_07.append(dice)
        iou_tr_epochs_07.append(intersection_over_union)

        acc_val_epochs_07.append(acc_val)
        pre_val_epochs_07.append(prec_val)
        rec_val_epochs_07.append(rec_val)
        f1_val_epochs_07.append(f1_val)
        dice_val_epochs_07.append(dice_val)
        iou_val_epochs_07.append(intersection_over_union_val)

      elif(mode == 0.8):
        acc_tr_epochs_08.append(acc)
        pre_tr_epochs_08.append(prec)
        rec_tr_epochs_08.append(rec)
        f1_tr_epochs_08.append(f1)
        dice_tr_epochs_08.append(dice)
        iou_tr_epochs_08.append(intersection_over_union)

        acc_val_epochs_08.append(acc_val)
        pre_val_epochs_08.append(prec_val)
        rec_val_epochs_08.append(rec_val)
        f1_val_epochs_08.append(f1_val)
        dice_val_epochs_08.append(dice_val)
        iou_val_epochs_08.append(intersection_over_union_val)

      else:
        acc_tr_epochs_otsu.append(acc)
        pre_tr_epochs_otsu.append(prec)
        rec_tr_epochs_otsu.append(rec)
        f1_tr_epochs_otsu.append(f1)
        dice_tr_epochs_otsu.append(dice)
        iou_tr_epochs_otsu.append(intersection_over_union)

        acc_val_epochs_otsu.append(acc_val)
        pre_val_epochs_otsu.append(prec_val)
        rec_val_epochs_otsu.append(rec_val)
        f1_val_epochs_otsu.append(f1_val)
        dice_val_epochs_otsu.append(dice_val)
        iou_val_epochs_otsu.append(intersection_over_union_val)

scores_dict_05 = {}

scores_dict_05['acc_tr'] = acc_tr_epochs_05
scores_dict_05['pre_tr'] = pre_tr_epochs_05
scores_dict_05['rec_tr'] = rec_tr_epochs_05
scores_dict_05['f1_tr'] = f1_tr_epochs_05
scores_dict_05['dice_tr'] = dice_tr_epochs_05
scores_dict_05['iou_tr'] = iou_tr_epochs_05

scores_dict_05['acc_val'] = acc_val_epochs_05
scores_dict_05['pre_val'] = pre_val_epochs_05
scores_dict_05['rec_val'] = rec_val_epochs_05
scores_dict_05['f1_val'] = f1_val_epochs_05
scores_dict_05['dice_val'] = dice_val_epochs_05
scores_dict_05['iou_val'] = iou_val_epochs_05

scores_dict_06 = {}

scores_dict_06['acc_tr'] = acc_tr_epochs_06
scores_dict_06['pre_tr'] = pre_tr_epochs_06
scores_dict_06['rec_tr'] = rec_tr_epochs_06
scores_dict_06['f1_tr'] = f1_tr_epochs_06
scores_dict_06['dice_tr'] = dice_tr_epochs_06
scores_dict_06['iou_tr'] = iou_tr_epochs_06

scores_dict_06['acc_val'] = acc_val_epochs_06
scores_dict_06['pre_val'] = pre_val_epochs_06
scores_dict_06['rec_val'] = rec_val_epochs_06
scores_dict_06['f1_val'] = f1_val_epochs_06
scores_dict_06['dice_val'] = dice_val_epochs_06
scores_dict_06['iou_val'] = iou_val_epochs_06

scores_dict_07 = {}

scores_dict_07['acc_tr'] = acc_tr_epochs_07
scores_dict_07['pre_tr'] = pre_tr_epochs_07
scores_dict_07['rec_tr'] = rec_tr_epochs_07
scores_dict_07['f1_tr'] = f1_tr_epochs_07
scores_dict_07['dice_tr'] = dice_tr_epochs_07
scores_dict_07['iou_tr'] = iou_tr_epochs_07

scores_dict_07['acc_val'] = acc_val_epochs_07
scores_dict_07['pre_val'] = pre_val_epochs_07
scores_dict_07['rec_val'] = rec_val_epochs_07
scores_dict_07['f1_val'] = f1_val_epochs_07
scores_dict_07['dice_val'] = dice_val_epochs_07
scores_dict_07['iou_val'] = iou_val_epochs_07

scores_dict_08 = {}

scores_dict_08['acc_tr'] = acc_tr_epochs_08
scores_dict_08['pre_tr'] = pre_tr_epochs_08
scores_dict_08['rec_tr'] = rec_tr_epochs_08
scores_dict_08['f1_tr'] = f1_tr_epochs_08
scores_dict_08['dice_tr'] = dice_tr_epochs_08
scores_dict_08['iou_tr'] = iou_tr_epochs_08

scores_dict_08['acc_val'] = acc_val_epochs_08
scores_dict_08['pre_val'] = pre_val_epochs_08
scores_dict_08['rec_val'] = rec_val_epochs_08
scores_dict_08['f1_val'] = f1_val_epochs_08
scores_dict_08['dice_val'] = dice_val_epochs_08
scores_dict_08['iou_val'] = iou_val_epochs_08

scores_dict_otsu = {}

scores_dict_otsu['acc_tr'] = acc_tr_epochs_otsu
scores_dict_otsu['pre_tr'] = pre_tr_epochs_otsu
scores_dict_otsu['rec_tr'] = rec_tr_epochs_otsu
scores_dict_otsu['f1_tr'] = f1_tr_epochs_otsu
scores_dict_otsu['dice_tr'] = dice_tr_epochs_otsu
scores_dict_otsu['iou_tr'] = iou_tr_epochs_otsu

scores_dict_otsu['acc_val'] = acc_val_epochs_otsu
scores_dict_otsu['pre_val'] = pre_val_epochs_otsu
scores_dict_otsu['rec_val'] = rec_val_epochs_otsu
scores_dict_otsu['f1_val'] = f1_val_epochs_otsu
scores_dict_otsu['dice_val'] = dice_val_epochs_otsu
scores_dict_otsu['iou_val'] = iou_val_epochs_otsu


# # Masking 1: 0.5 Thresholding Performance

# In[ ]:


plotter_3(scores_dict_05)


# # Masking 2: 0.6 Thresholding Performance

# In[ ]:


plotter_3(scores_dict_06)


# # Masking 3: 0.7 Thresholding Performance

# In[ ]:


plotter_3(scores_dict_07)


# # Masking 4: 0.8 Thresholding Performance

# In[ ]:


plotter_3(scores_dict_08)


# # Masking 5: Otsu Thresholding Performance

# In[ ]:


plotter_3(scores_dict_otsu)


# # Hyperparemeter Tuning Results: Best Setting: Masking 2: 0.6 Thresholding at Epoch 33, SGD+Momentum(0.9) with L2 Regularization Coefficient = 5 * 1e-5, Learning Rate = 1e-2, Learning Rate Scheduler: Every 10 steps by a factor of 0.1.

# # Synthesizing a New Dataset Utilizing the Best Model:

# In[ ]:


import gc
gc.collect()
torch.cuda.empty_cache()


# In[ ]:


get_ipython().system('unzip /content/drive/MyDrive/ee641/Project_Datasets/Indiana_University_Images_Zipped/images_normalized.zip')


# In[ ]:


import pandas as pd

data_annotations = pd.read_csv('/content/drive/MyDrive/ee641/Project_Datasets/Indiana_University_Images_Zipped/indiana_images_info.csv')
display(data_annotations)


# In[ ]:


data_annotations.info()


# In[ ]:


data_annotations.dropna(subset=['findings'], inplace=True)
data_annotations.info()


# In[ ]:


filename_list = list(data_annotations['filename'])
print(filename_list)


# In[ ]:


path = '/content/images_normalized/' + filename_list[3]
temp_img = Image.open(path)
print(type(temp_img))
print(temp_img.size)
#temp_img.show()
print((temp_img.size)[0])
print((temp_img.size)[1])
temp_img = temp_img.convert('RGB')
#print(temp_transform(temp_img).shape)
temp_img = temp_img.resize((224, 224))
print(temp_img.size)
# temp_img.show()
# print((temp_img.size)[0])
# print((temp_img.size)[1])


# In[ ]:


class NoneTransform(object):
    ''' Does nothing to the image. To be used instead of None '''
    
    def __call__(self, image):       
        return image

temp_transform = transforms.Compose([
            transforms.ToTensor(),            
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))  if temp_img.mode!='RGB'  else NoneTransform()            
            ]) 


# In[ ]:


sum_matrix = np.zeros((224,224,3))
for i in range(len(filename_list)):
  img_name = filename_list[i]
  if(i % 100 == 0):
    print(i)
  #/content/images_normalized
  path = '/content/images_normalized/' + img_name
  temp_img = Image.open(path)
  temp_img = temp_img.convert('RGB')
  temp_img = temp_img.resize((224, 224))
  temp_img_tensor = temp_transform(temp_img)
  temp_img_tensor = temp_img_tensor.permute(1, 2, 0)
  temp_img_numpy = temp_img_tensor.numpy()
  sum_matrix = sum_matrix + temp_img_numpy


# In[ ]:


sum_matrix_1d = sum_matrix[:,:,0]
sum_matrix_1d_avg = sum_matrix_1d / len(filename_list)
mean_chest = np.mean(sum_matrix_1d_avg, axis = None)
std_chest = np.std(sum_matrix_1d_avg, axis = None)
print("Mean of the Gray Scale Training Chest Images of Pixels = ", mean_chest)
print("Std of the Gray Scale Training Chest Images of Pixels = ", std_chest)


# In[ ]:


images_root_dir = '/content/images_normalized'


# In[ ]:


class Indiana_University_Dataset(Dataset):
  def __init__(self, images_root_dir, filename_list, transform=None, cuda = True):
    self.images_root_dir = images_root_dir
    self.filename_list = filename_list
    self.transform = transform
    self.img_h = 224
    self.img_w = 224
    self.cuda = cuda

  def __len__(self):
    return len(self.filename_list)
  
  def __getitem__(self, idx):
    image_filename = self.filename_list[idx]
    image_directory = self.images_root_dir + '/' + image_filename

    image = Image.open(image_directory)
    org_image_width = (image.size)[0]
    org_image_height = (image.size)[1]

    # if(image.mode != 'RGB'):
    #   image = image.convert('RGB')
    
    image = image.resize((self.img_h, self.img_w))

    if(self.transform != None):
      image = self.transform(image)

    # target_dict = {}

    # target_dict['image_name'] = image_filename

    # target = target_dict

    target = image_filename

    return image, target

def collate_fn(batch):
    images = []
    targets = []
    for item in batch:
        images.append(item[0])
        targets.append(item[1])
    images = torch.stack(images, 0)
    return images, targets


# In[ ]:


batch_size = 50
root_directory = '/content/images_normalized'


class NoneTransform(object):
  def __call__(self, image):
    return image

class GrayscaleToRGB:
  def __call__(self, image):
    if image.mode != 'RGB':
      return image.convert('RGB')
    return image

mean_chest = 0.6262051540969282
std_chest = 0.16896052000616008

my_transform_with_Normalization = transforms.Compose([
    GrayscaleToRGB(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #transforms.Normalize([mean_chest, mean_chest, mean_chest], [std_chest, std_chest, std_chest])
])


#Mean of the Gray Scale Training Chest Images of Pixels =  0.5240804197105211
#Std of the Gray Scale Training Chest Images of Pixels =  0.17148634738092733



# my_transform_Specialized = transforms.Compose([
#             transforms.ToTensor(),            
#             transforms.Lambda(lambda x: x.repeat(3, 1, 1))  if temp_img.mode!='RGB'  else NoneTransform(),
#             transforms.Normalize([mean_chest, mean_chest, mean_chest], [std_chest, std_chest, std_chest])                
#             ]) 

# my_transform_ImageNet = transforms.Compose([
#             transforms.ToTensor(),            
#             transforms.Lambda(lambda x: x.repeat(3, 1, 1))  if temp_img.mode!='RGB'  else NoneTransform(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                
#             ]) 

dataset = Indiana_University_Dataset(root_directory, filename_list, transform=my_transform_with_Normalization, cuda = True)


# In[ ]:


data_loader = DataLoader(dataset, shuffle = False, batch_size=batch_size, collate_fn=collate_fn)


# In[ ]:


len(data_loader.dataset)


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[ ]:


num_classes = 14
device = torch.device(device)
model = get_model_instance_segmentation(num_classes)
model = model.to(device)


# In[ ]:


import gc
gc.collect()
torch.cuda.empty_cache()


# # Masked & Non-Masked Image Creation

# In[ ]:


PATH = '/content/drive/MyDrive/ee641/Project_Datasets/Model_Checkpoints_3/model_epoch_33.pth'
checkpoint = torch.load(PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
counter = 0
mean_chest = [0.485, 0.456, 0.406]
std_chest = [0.229, 0.224, 0.225]
mean_for_unmasking = 0.6262051540969282
rgb_mean_unmask = mean_for_unmasking * 255
image_masked_dir = '/content/drive/MyDrive/ee641/Project_Datasets/Masked_Synthesized_Indiana_University_Dataset/'
image_non_masked_dir = '/content/drive/MyDrive/ee641/Project_Datasets/Non_Masked_Synthesized_Indiana_University_Dataset/'
with torch.no_grad(): 
  for images, targets in data_loader:
    model.eval()
    image_batch = images.to(device)
    predictions = model(image_batch)
    #indices_of_images = np.arange(0,len(images),1)
    #indices_of_images = np.arange(counter * len(images), (counter + 1) * len(images), 1)
    for img_trg_index in range(len(images)):
      mask_pred = list(predictions[img_trg_index]['masks'])
      #file_name = list(targets[img_trg_index])
      file_name = targets[img_trg_index]
      fig_masked, ax_masked = plt.subplots()
      fig_non_masked, ax_non_masked = plt.subplots()
      actual_image = images[img_trg_index]
      image_numpy = (actual_image.detach().cpu()).numpy()
      actual_denormalized_image = denormalize_image(image_numpy, mean_chest, std_chest)
      ax_masked.imshow(actual_denormalized_image)
      ax_non_masked.imshow(actual_denormalized_image)
      mask_pred_arrs = []
        
      for i in mask_pred:
        tmp = i.detach().cpu()
        tmp2 = (tmp[0]).numpy()
        temp_mask = np.zeros((224, 224))
        temp_locs = np.where(tmp2 > 0.6)
        temp_mask[temp_locs[0], temp_locs[1]] = 1
        mask_pred_arrs.append(temp_mask)
  
      combined_mask_predicted = combine_masks(mask_pred_arrs)
      predicted_mask_color = np.array([1, 0, 0, 0.25]) #Red
      predicted_unmask_color = np.array([mean_for_unmasking, mean_for_unmasking, mean_for_unmasking, 0.5]) #Mean Unmask
      predicted_mask_rgb = np.zeros((224, 224, 4))
      predicted_non_mask_rgb = np.zeros((224, 224, 4))
      predicted_mask_rgb[combined_mask_predicted == 1] = predicted_mask_color
      #if(np.sum(combined_mask_predicted == True) == 0)
      predicted_non_mask_rgb[combined_mask_predicted == 0] = predicted_unmask_color

      ax_masked.imshow(predicted_mask_rgb)
      ax_non_masked.imshow(predicted_non_mask_rgb)
      ax_masked.axis('off')
      ax_non_masked.axis('off')

      temp_masked_filename = 'Masked_' + file_name
      temp_non_masked_filename = 'Non_Masked_' + file_name

      masked_directory = image_masked_dir + temp_masked_filename
      non_masked_directory = image_non_masked_dir + temp_non_masked_filename

      fig_masked.savefig(masked_directory, bbox_inches='tight', pad_inches=0)
      fig_non_masked.savefig(non_masked_directory, bbox_inches='tight', pad_inches=0)

      plt.close(fig_masked)
      plt.close(fig_non_masked)
    #counter += 1


# In[ ]:


get_ipython().system('jupyter nbconvert --to script Final_Version_Instance_Segmentation_with_a_Novel_Mask_RCNN_14_Classes_Orkun_Bedir.ipynb')


# # Future Probable Model Trials

# # DenseNet Backbone

# In[ ]:


import torch
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models.densenet import densenet121
from torchvision.ops import MultiScaleRoIAlign

def get_densenet_backbone():
    densenet = densenet121(pretrained=True)
    backbone = torch.nn.Sequential(*list(densenet.children())[:-1])
    backbone.out_channels = densenet.features[-1].num_features
    return backbone

def get_densenet_fpn():
    backbone = get_densenet_backbone()
    return_features = {'transition3': 'feat3', 'norm5': 'feat4'}
    in_channels_list = [backbone.features[key].num_features for key in return_features]
    out_channels = 256
    fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels, extra_blocks=None)
    return backbone, fpn

def get_model_instance_segmentation(num_classes):
    backbone, fpn = get_densenet_fpn()
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # Add MultiScaleRoIAlign for box and mask
    box_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                      output_size=7,
                                      sampling_ratio=2)

    mask_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                       output_size=14,
                                       sampling_ratio=2)

    model = MaskRCNN(backbone, num_classes, rpn_anchor_generator=anchor_generator, box_roi_pool=box_roi_pool, mask_roi_pool=mask_roi_pool, box_head=None, mask_head=None, keypoint_head=None, box_predictor=None, mask_predictor=None, keypoint_predictor=None, rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000, rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000, rpn_nms_thresh=0.7, rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3, rpn_batch_size_per_image=256, rpn_positive_fraction=0.5, box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100, box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5, box_batch_size_per_image=512, box_positive_fraction=0.25, bbox_reg_weights=None, mask_size=28, mask_positive_fraction=0.1, keypoint_size=None, keypoint_positive_fraction=None, keypoint_visibility_score_weight=0.5, keypoint_distribution_weight=0.25)

    model.backbone.body = backbone
    model.backbone.fpn = fpn

    # Modify the box and mask predictor to match the new number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Modify the mask predictor for the new backbone
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


# # End for Now

# In[ ]:


from matplotlib.patches import Rectangle, Polygon
def evaluate(model, val_loader, device, epoch):

  with torch.no_grad():
    for images, targets in val_loader:
      model.eval()
      image_batch = images.to(device)
      #target_batch = [{'boxes':dictionary['boxes'].to(device), 'labels':dictionary['labels'].to(device), 'masks':dictionary['masks'].to(device)} for dictionary in targets]
      predictions = model(image_batch)

      box_pred_1 = list(predictions[0]['boxes'])
      box_pred_2 = list(predictions[5]['boxes'])
      box_pred_3 = list(predictions[10]['boxes'])
      box_pred_4 = list(predictions[15]['boxes'])

      box_actual_1 = list(targets[0]['boxes'])
      box_actual_2 = list(targets[5]['boxes'])
      box_actual_3 = list(targets[10]['boxes'])
      box_actual_4 = list(targets[15]['boxes'])

      mask_pred_1 = list(predictions[0]['masks'])
      mask_pred_2 = list(predictions[5]['masks'])
      mask_pred_3 = list(predictions[10]['masks'])
      mask_pred_4 = list(predictions[15]['masks'])

      mask_actual_1 = list(targets[0]['masks'])
      mask_actual_2 = list(targets[5]['masks'])
      mask_actual_3 = list(targets[10]['masks'])
      mask_actual_4 = list(targets[15]['masks'])

      actual_image_1 = images[0]
      actual_image_2 = images[5]
      actual_image_3 = images[10]
      actual_image_4 = images[15]

      fig, ax = plt.subplots(2, 2, figsize=(16, 16))

      ax[0, 0].imshow(actual_image_1.permute(1, 2, 0))
      for i in mask_pred_1:
        tmp = i.detach().cpu()
        for j in range(len(tmp)):
          tmp2 = (tmp[j]).numpy()
          temp_locs = np.where(tmp2 > 0.5)
          temp_mask = np.zeros((224, 224))
          temp_mask[temp_locs[0], temp_locs[1]] = 1
          predicted_mask_color = np.array([0, 1, 0]) #Green
          predicted_mask_rgb = np.zeros((224, 224, 3))
          predicted_mask_rgb[temp_mask == 1] = predicted_mask_color
          ax[0, 0].imshow(predicted_mask_rgb, alpha = 0.5)
      for i in mask_actual_1:
        tmp = i.detach().cpu()
        for j in range(len(tmp)):
          tmp2 = (tmp[j]).numpy()
          temp_locs = np.where(tmp2 > 0.5)
          temp_mask = np.zeros((224, 224))
          temp_mask[temp_locs[0], temp_locs[1]] = 1
          actual_mask_color = np.array([1, 0, 0]) #Red
          actual_mask_rgb = np.zeros((224, 224, 3))
          actual_mask_rgb[temp_mask == 1] = actual_mask_color
          ax[0, 0].imshow(actual_mask_rgb, alpha = 0.5)
      for i in box_pred_1:
        tmp = i.detach().cpu().numpy()
        tmp_rect = Rectangle((tmp[0], tmp[3]), tmp[2] - tmp[0], tmp[3] - tmp[1], linewidth=1.2, edgecolor='g', facecolor='none')
        ax[0, 0].add_patch(tmp_rect)
      for j in box_actual_1:
        tmp = j.detach().cpu().numpy()
        tmp_rect = Rectangle((tmp[0], tmp[3]), tmp[2] - tmp[0], tmp[3] - tmp[1], linewidth=2, edgecolor='r', facecolor='none')
        ax[0, 0].add_patch(tmp_rect)


      ax[0, 1].imshow(actual_image_2.permute(1, 2, 0))
      for i in mask_pred_2:
        tmp = i.detach().cpu()
        for j in range(len(tmp)):
          tmp2 = (tmp[j]).numpy()
          temp_locs = np.where(tmp2 > 0.5)
          temp_mask = np.zeros((224, 224))
          temp_mask[temp_locs[0], temp_locs[1]] = 1
          predicted_mask_color = np.array([0, 1, 0]) #Green
          predicted_mask_rgb = np.zeros((224, 224, 3))
          predicted_mask_rgb[temp_mask == 1] = predicted_mask_color
          ax[0, 1].imshow(predicted_mask_rgb, alpha = 0.5)
      for i in mask_actual_2:
        tmp = i.detach().cpu()
        for j in range(len(tmp)):
          tmp2 = (tmp[j]).numpy()
          temp_locs = np.where(tmp2 > 0.5)
          temp_mask = np.zeros((224, 224))
          temp_mask[temp_locs[0], temp_locs[1]] = 1
          actual_mask_color = np.array([1, 0, 0]) #Red
          actual_mask_rgb = np.zeros((224, 224, 3))
          actual_mask_rgb[temp_mask == 1] = actual_mask_color
          ax[0, 1].imshow(actual_mask_rgb, alpha = 0.5)
      for i in box_pred_2:
        tmp = i.detach().cpu().numpy()
        tmp_rect = Rectangle((tmp[0], tmp[3]), tmp[2] - tmp[0], tmp[3] - tmp[1], linewidth=1.2, edgecolor='g', facecolor='none')
        ax[0, 1].add_patch(tmp_rect)
      for j in box_actual_2:
        tmp = j.detach().cpu().numpy()
        tmp_rect = Rectangle((tmp[0], tmp[3]), tmp[2] - tmp[0], tmp[3] - tmp[1], linewidth=2, edgecolor='r', facecolor='none')
        ax[0, 1].add_patch(tmp_rect)


      ax[1, 0].imshow(actual_image_3.permute(1, 2, 0))
      for i in mask_pred_3:
        tmp = i.detach().cpu()
        for j in range(len(tmp)):
          tmp2 = (tmp[j]).numpy()
          temp_locs = np.where(tmp2 > 0.5)
          temp_mask = np.zeros((224, 224))
          temp_mask[temp_locs[0], temp_locs[1]] = 1
          predicted_mask_color = np.array([0, 1, 0]) #Green
          predicted_mask_rgb = np.zeros((224, 224, 3))
          predicted_mask_rgb[temp_mask == 1] = predicted_mask_color
          ax[1, 0].imshow(predicted_mask_rgb, alpha = 0.5)
      for i in mask_actual_3:
        tmp = i.detach().cpu()
        for j in range(len(tmp)):
          tmp2 = (tmp[j]).numpy()
          temp_locs = np.where(tmp2 > 0.5)
          temp_mask = np.zeros((224, 224))
          temp_mask[temp_locs[0], temp_locs[1]] = 1
          actual_mask_color = np.array([1, 0, 0]) #Red
          actual_mask_rgb = np.zeros((224, 224, 3))
          actual_mask_rgb[temp_mask == 1] = actual_mask_color
          ax[1, 0].imshow(actual_mask_rgb, alpha = 0.5)
      for i in box_pred_3:
        tmp = i.detach().cpu().numpy()
        tmp_rect = Rectangle((tmp[0], tmp[3]), tmp[2] - tmp[0], tmp[3] - tmp[1], linewidth=1.2, edgecolor='g', facecolor='none')
        ax[1, 0].add_patch(tmp_rect)
      for j in box_actual_3:
        tmp = j.detach().cpu().numpy()
        tmp_rect = Rectangle((tmp[0], tmp[3]), tmp[2] - tmp[0], tmp[3] - tmp[1], linewidth=2, edgecolor='r', facecolor='none')
        ax[1, 0].add_patch(tmp_rect)

      ax[1, 1].imshow(actual_image_4.permute(1, 2, 0))
      for i in mask_pred_4:
        tmp = i.detach().cpu()
        for j in range(len(tmp)):
          tmp2 = (tmp[j]).numpy()
          temp_locs = np.where(tmp2 > 0.5)
          temp_mask = np.zeros((224, 224))
          temp_mask[temp_locs[0], temp_locs[1]] = 1
          predicted_mask_color = np.array([0, 1, 0]) #Green
          predicted_mask_rgb = np.zeros((224, 224, 3))
          predicted_mask_rgb[temp_mask == 1] = predicted_mask_color
          ax[1, 1].imshow(predicted_mask_rgb, alpha = 0.5)
      for i in mask_actual_4:
        tmp = i.detach().cpu()
        for j in range(len(tmp)):
          tmp2 = (tmp[j]).numpy()
          temp_locs = np.where(tmp2 > 0.5)
          temp_mask = np.zeros((224, 224))
          temp_mask[temp_locs[0], temp_locs[1]] = 1
          actual_mask_color = np.array([1, 0, 0]) #Red
          actual_mask_rgb = np.zeros((224, 224, 3))
          actual_mask_rgb[temp_mask == 1] = actual_mask_color
          ax[1, 1].imshow(actual_mask_rgb, alpha = 0.5)
      for i in box_pred_4:
        tmp = i.detach().cpu().numpy()
        tmp_rect = Rectangle((tmp[0], tmp[3]), tmp[2] - tmp[0], tmp[3] - tmp[1], linewidth=1.2, edgecolor='g', facecolor='none')
        ax[1, 1].add_patch(tmp_rect)
      for j in box_actual_4:
        tmp = j.detach().cpu().numpy()
        tmp_rect = Rectangle((tmp[0], tmp[3]), tmp[2] - tmp[0], tmp[3] - tmp[1], linewidth=2, edgecolor='r', facecolor='none')
        ax[1, 1].add_patch(tmp_rect)

      image_dir = '/content/Boxes_Images_Resized/'
      temp_filename = image_dir + 'Epoch_' + str(epoch) + '_BBox_Preds_BLUE_Mask_Preds_GREEN_BBox_Actual_RED_Mask_Actual_RED.png'
      temp_title = 'Epoch ' + str(epoch) + ' BBox Predictions (BLUE) vs. Actual BBox(RED) and Mask Predictions (GREEN) vs. Actual Masks(RED)'
      #plt.title(temp_title)
      fig.suptitle(temp_title)
      plt.show()
      fig.savefig(temp_filename)

  return


# In[ ]:


images, targets = next(iter(val_loader))


# In[ ]:


sample_image = images[0]
sample_target = targets[0]
print(sample_image)
print(sample_target)
sample_boxes = sample_target['boxes']
sample_masks = sample_target['masks']
sample_labels = sample_target['labels']


# In[ ]:


num_classes = 2
model_org = get_model_instance_segmentation(num_classes)


# In[ ]:


model_org.eval()
image, target = next(iter(val_loader))
#sample_outputs = model(sample_image)
#image_list = [image.squeeze(0)]
with torch.no_grad():
  output_eval = model_org(image_list)

# model_org.train()
# output_train = model_org(image_list, target)


# In[ ]:


a = list(output_eval[0]['masks'])
for i in a:
  temp = i.numpy()
  print(temp[0])
  print(temp[0].shape)
  loc = np.where(temp[0] > 0.5)
  print(loc)
  print((temp[0])[162][140])
  #print()
  break
mask = np.zeros((224, 224))
mask[loc[0], loc[1]] = 1
#print(loc[0])
#print(loc[0][0])
print(mask)
print(np.where(mask == 1))


# In[ ]:


custom_backbone = CustomKerasBackbone(pytorch_backbone)


# In[ ]:


num_classes = 2
model = get_model_instance_segmentation(custom_backbone, num_classes)


# In[ ]:


mask_rcnn_model.transform.image_mean = [0.0, 0.0, 0.0]
mask_rcnn_model.transform.image_std = [1.0, 1.0, 1.0]


# In[ ]:


mask_rcnn_model.eval()
image, target = next(iter(train_loader))
#sample_outputs = model(sample_image)
image_list = [image.squeeze(0)]
with torch.no_grad():
    output = mask_rcnn_model(image_list)
#sample_outputs = mask_rcnn_model(sample_image)


# In[ ]:


model.eval()
with torch.no_grad():
    output = model(image_list)


# In[ ]:


class COCO_Dataset(Dataset):
    def __init__(self, root_dir, annFile, transform=None, cuda=True, set_name = None, mapping=None):
        self.mappping = mapping 
        self.root_dir = root_dir
        self.transform = transform
        self.imgs = os.listdir(root_dir)
        self.mode = set_name
        # annotationsx
        self.ct = COCO_Text(annFile)
        if(self.mode == 'train'):
          self.imgIds = self.ct.getImgIds(imgIds=self.ct.train, 
                      catIds=[('legibility','legible'),('class','machine printed')])
          self.imgIds.remove(275939)
          self.imgIds.remove(443671)
        elif(self.mode == 'val'):
          self.imgIds = self.ct.getImgIds(imgIds=self.ct.train, 
                      catIds=[('legibility','legible'),('class','machine printed')])
          self.imgIds.remove(275939)
          self.imgIds.remove(443671)
        else:
          self.imgIds = self.ct.getImgIds(imgIds=self.ct.test, 
                      catIds=[('legibility','legible'),('class','machine printed')])
        
        for imgId in self.imgIds:
            file_name = self.ct.loadImgs(imgId)[0]['file_name']
            if file_name not in self.imgs:
                self.imgIds.remove(imgId)
        # manual exclude
        #self.imgIds.remove(275939)
        #self.imgIds.remove(443671)

        # remaining images
        print(f"remaining images in ann file: {len(self.imgIds)}, remaining images in folder: {len(self.imgs)}")

        self.imgIds.sort()
        # sort the images in same order as the annotations
        self.imgs = [self.ct.loadImgs(imgId)[0]['file_name'] for imgId in self.imgIds]

        self.img_h = 224
        self.img_w = 224
        self.cuda = cuda

        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        # TODO: return image / bounding-box [list] pair for self.imgIds[idx]
        # resize the image to conform to your model input and transformed
        # bounding box coordinates
        
        # see COCO_Text helper (i.e., self.ct)
        # return image, target
        temp_img = self.ct.loadImgs(self.imgIds[idx])[0]

        #temp_img = self.imgs[idx]

        image_height = temp_img['height']
        image_width = temp_img['width']
        image_directory = self.root_dir + '/' + temp_img['file_name']
        #image = io.imread('%s/train/%s'%(self.root_dir,temp_img['file_name']))
        image = Image.open(image_directory)
        #image = io.imread(image_directory)
        annotation_IDs = self.ct.getAnnIds(imgIds=temp_img['id'])
        annotations = self.ct.loadAnns(ids = annotation_IDs)
        bounding_box_coordinates = []
        #val_bbox_coordinates = []
        labels = []
        for annotation_item in annotations:

          #if('utf8_string' in annotation_item.keys()):
          #  temp_name_label = annotation_item['utf8_string']
          #else:
          #  temp_name_label = str((tmp_anns[j])['id'])

          #labels.append(self.mapping[temp_name_label])
          labels.append(1)

          bounding_box_details = annotation_item['bbox']
          left_bottom_x = bounding_box_details[0]
          left_bottom_y = bounding_box_details[1]
          left_top_x = left_bottom_x
          left_top_y = left_bottom_y + bounding_box_details[3]
          right_bottom_x = left_top_x + bounding_box_details[2]
          right_bottom_y = left_bottom_y
          right_top_x = right_bottom_x
          right_top_y = left_top_y
          #validation_bbox = [left_bottom_x, left_bottom_y, right_top_x, right_top_y]
          #val_bbox_coordinates.append(validation_bbox)



          scaled_left_top_y = (self.img_h * left_top_y) / image_height
          scaled_left_top_x = (self.img_w * left_top_x) / image_width
          scaled_left_bottom_y = (self.img_h * left_bottom_y) / image_height
          scaled_left_bottom_x = (self.img_w * left_bottom_x) / image_width
          scaled_right_top_y = (self.img_h * right_top_y) / image_height
          scaled_right_top_x = (self.img_w * right_top_x) / image_width
          scaled_right_bottom_y = (self.img_h * right_bottom_y) / image_height
          scaled_right_bottom_x = (self.img_w * right_bottom_x) / image_width
          temp_bounding_box = [scaled_left_top_y, scaled_left_top_x, 
                               scaled_left_bottom_y, scaled_left_bottom_x,
                               scaled_right_top_y, scaled_right_top_x, 
                               scaled_right_bottom_y, scaled_right_bottom_x]
          xmin, ymin, xmax, ymax = scaled_left_bottom_x, scaled_left_bottom_y, scaled_right_top_x, scaled_right_top_y
          formatted_box = [xmin, ymin, xmax, ymax]
          #bounding_box_coordinates.append(temp_bounding_box)
          bounding_box_coordinates.append(formatted_box)
      
        #print(image.shape)

        if(image.mode != 'RGB'):
          image = image.convert('RGB')

        
        image_1 = image.resize((self.img_h, self.img_w))
        transformed_image = self.transform(image_1)

        #print(transformed_image.shape)

        transformed_labels = torch.LongTensor(labels)
        transformed_bbox = torch.FloatTensor(bounding_box_coordinates)
        #transformed_labels = np.array(labels, dtype=np.int64)
        #transformed_bbox = np.array(labels, dtype=np.float32)
        #transformed_bbox = torch.as_tensor(bounding_box_coordinates, dtype=torch.float32)
        #transformed_labels = torch.as_tensor(labels, dtype=torch.int64)

        #target = []

        temp_dict = {}
        temp_dict['boxes'] = transformed_bbox
        temp_dict['labels'] = transformed_labels
        target = temp_dict
        #target.append(temp_dict)

        #return transformed_image, bounding_box_coordinates
        #if(self.mode == 'train'):
        #  return transformed_image, target
        #else:
        #  return transformed_image, target, val_bbox_coordinates, image, image_height, image_width
        
        return transformed_image, target

        #raise NotImplementedError("CocoDataset::__getitem__()")


# coalate_fn is used to collate the data into batches
#def collate_fn(batch, mode = None):
def collate_fn(batch):
    images = []
    targets = []
    #val_bbox_coordinates = []
    #plain_images = []
    #image_heights = []
    #image_widths = []
    for item in batch:
        images.append(item[0])
        targets.append(item[1])
    images = torch.stack(images, 0)
    #if(mode == 'train'):
    return images, targets
    #else:
    #return images
    #return images, targets

