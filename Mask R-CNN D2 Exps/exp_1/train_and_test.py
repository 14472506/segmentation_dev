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
#import numpy as np
#import tqdm  
import os, json, cv2, random, copy

# ----- utilites from detectron2 ----------------------------------------- #
#from detectron2 import model_zoo
#from detectron2.engine import DefaultPredictor
#from detectron2.config import get_cfg
#from matplotlib.pyplot import step
#from torch import batch_norm
#from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, build_detection_train_loader 

# ----- for data registration and validation ----------------------------- #
#from detectron2.structures import BoxMode
#from detectron2.data.datasets import load_coco_json, register_coco_instances
#from detectron2.engine import DefaultTrainer
#import random

# ----- for data augmentation -------------------------------------------- #
#from detectron2.data import detection_utils as utils
#import detectron2.data.transforms as T

# ------ for evaluation -------------------------------------------------- #
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from augmentation import CustomTrainer

# ===== Model File Imports =============================================== #
# train_model
def model_train(cfg, TRAIN):
    """
    detials
    """
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    if TRAIN == True:
        trainer.train()
    return(trainer)

# model test
def model_test(cfg, predictor, testing_name, fun_output_dir):
    """
    details
    """    
    evaluator = COCOEvaluator(testing_name, cfg, False, output_dir=fun_output_dir)
    val_loader = build_detection_test_loader(cfg, testing_name)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
