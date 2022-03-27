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
import torch, torchvision
#from torch.utils.data import dataset

# ----- setting up detectron logger -------------------------------------- # 
#import detectron2
#from detectron2 import modeling
#from detectron2.data import build, datasets
#from detectron2.utils.logger import setup_logger
#from tqdm import tqdm
#setup_logger()

# ----- some common libraries -------------------------------------------- #
#import numpy as np
#import tqdm  
import os, json, cv2, random, copy

# ----- utilites from detectron2 ----------------------------------------- #
#from detectron2 import model_zoo
#from detectron2.engine import DefaultPredictor
#from detectron2.config import get_cfg
#from detectron2.utils.video_visualizer import VideoVisualizer
#from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, build_detection_train_loader 

# ----- for data registration and validation ----------------------------- #
#from detectron2.structures import BoxMode
#from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.engine import DefaultTrainer
import random

# ----- for data augmentation -------------------------------------------- #
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T

# ------ for evaluation -------------------------------------------------- #
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# ===== Utility functions ================================================ #
# --- Custom traininer class --------------------------------------------- #
class CustomTrainer(DefaultTrainer):
    """
    detials
    """

    @classmethod
    def build_train_loader(cls, cfg):
        """
        detials
        """
        return build_detection_train_loader(cfg, mapper=augmentation)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        
        # if there is not a coco_eval folder
        if output_folder is None:
    
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

# ----- Augmentation class ----------------------------------------------- #
def augmentation(training_dict):
    """
    detials
    """
    aug_dict = copy.deepcopy(training_dict)
    image = utils.read_image(aug_dict["file_name"], format="RGB")
    transform_list = [T.RandomBrightness(0.8, 1.2),
                      T.RandomContrast(0.5, 1.5),
                      T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                      T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                      T.RandomCrop("absolute", (640, 640))
                      ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    aug_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in aug_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    aug_dict["instances"] = utils.filter_empty_instances(instances)
    return aug_dict