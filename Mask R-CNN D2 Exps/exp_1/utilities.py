"""
Something 
"""

# ===== Imported libraries =============================================== #
# ----- base libraries --------------------------------------------------- #
#import threading
#from turtle import width
#from cv2 import imread
#from detectron2 import data
#from detectron2.model_zoo.model_zoo import get
#import torch, torchvision
#from torch.utils.data import dataset

# ----- setting up detectron logger -------------------------------------- # 
#import detectron2
#from detectron2 import modeling
#from detectron2.data import build, datasets
#from detectron2.utils.logger import setup_logger
#from tqdm import tqdm
#setup_logger()

# ----- some common libraries -------------------------------------------- #
from cv2 import RETR_TREE
from matplotlib.pyplot import contour
import numpy as np
#import tqdm  
import os, json, cv2, random, copy, PIL

# ----- utilites from detectron2 ----------------------------------------- #
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
#from detectron2.utils.video_visualizer import VideoVisualizer
#from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, build_detection_train_loader 

# ----- for data registration and validation ----------------------------- #
from detectron2.structures import BoxMode
from detectron2.data.datasets import load_coco_json, register_coco_instances
#from detectron2.engine import DefaultTrainer
import random

# ----- for data augmentation -------------------------------------------- #
#from detectron2.data import detection_utils as utils
#import detectron2.data.transforms as T

# ------ for evaluation -------------------------------------------------- #
#from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# ===== Utility functions ================================================ #
# data loader
def data_loader(data_format, dataset_name, json_location, data_location, thing_classes_data):
    """
    details
    """
    # internal vgg to coco function
    def vgg_to_coco():
        """
        details
        """
        # open json file containing data annotations
        with open(json_location) as f:
            imgs_anns = json.load(f)
        
        # initialise list for dictionaries   
        dataset_dicts = []
        
        # itterate through data enteries in json
        for idx, v in enumerate(imgs_anns.values()):
            
            # init dict for data
            record = {}
            
            # retrieving file detials
            file_name = os.path.join(data_location, v['filename'])
            height, width = cv2.imread(file_name).shape[:2]
            
            # recording image details
            record["file_name"] = file_name
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width 
            
            # recording objects in image, looping through instances in data
            annos = v["regions"]
            objs = []
            for key in annos:
                for _, anno in key.items():                
                    if _ == "shape_attributes":

                        px = anno["all_points_x"]
                        py = anno["all_points_y"]
                        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                        poly = [p for x in poly for p in x]

                        obj = {
                            "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "segmentation": [poly],
                            "category_id": 0,
                        }
                        objs.append(obj)
            record["annotations"] = objs
            
            # appending image data to dataset dicts list
            dataset_dicts.append(record)
        
        # returning dataset dictionary
        return(dataset_dicts)
            
    # data formate is coco
    if data_format=="coco":
        register_coco_instances(dataset_name, {}, json_location, data_location)
    
    # data format is vgg    
    if data_format=="vgg":
        DatasetCatalog.register(dataset_name,  lambda: vgg_to_coco())
        
    # assigning and collecting meta data
    metadata = MetadataCatalog.get(dataset_name).set(thing_classes=thing_classes_data)
    dataset_dicts = DatasetCatalog.get(dataset_name)

    return(metadata)

# predictor implementation
def test_cfg_gen(model_path, test_threshold, model_file):
    """
    details
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.MODEL.WEIGHTS = model_path  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = test_threshold   # set a custom testing threshold
    return(cfg)

# mask to poly 
def mask_to_poly(mask):

    # internal function for point processing in mask point to poly logic
    def point_processing():

        # last point update
        x = int(last_point[2])
        y = int(last_point[1])
        x_list.append(x)
        y_list.append(y)
        segmentations.append(x)
        segmentations.append(y)

        # current point
        x = int(point[2])
        y = int(point[1])
        x_list.append(x)
        y_list.append(y)
        segmentations.append(x)
        segmentations.append(y)
    
    # get all non  zero co ordinates
    coords = np.column_stack(np.where(mask > 0))
    
    # prime last x, y, and point vals
    last_x = int(coords[0][2])
    last_y = int(coords[0][1])
    last_point = coords[0]
    
    # declare segmentation list
    segmentations = []

    # declaring x and y lists
    x_list = []
    y_list = []

    # loop through co-ordinates
    for i in range(coords.shape[0]):
        
        # current points
        point = coords[i]

                                
        if point[2] == last_x+1:        # if column is came
            pass  

        elif point[1] != last_y+1:      # if row point is not the same as the last plus 1
            point_processing()  

        elif point[2] != last_x:        # if point x is not the same as last point x 
            point_processing()
        
        else:                           # if other conditions pass
            pass
        
        # updating last points
        last_x = int(coords[i][2])
        last_y = int(coords[i][1])
        last_point = coords[i]


    # final x y points
    x = int(coords[-1][2])
    y = int(coords[-1][1])
    x_list.append(x)
    y_list.append(y)   
    segmentations.append(x)
    segmentations.append(y)

    top_left_x = min(x_list)
    top_left_y = min(y_list)
    width = max(x_list) - top_left_x
    height = max(y_list) - top_left_y
    bbox = [top_left_x, top_left_y, width, height]

    return segmentations, bbox