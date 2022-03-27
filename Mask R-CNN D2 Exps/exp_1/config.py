"""
Something 
"""

# ===== Imported libraries =============================================== #
# ----- base libraries --------------------------------------------------- #
#import threading
#from turtle import width
#from cv2 import imread
#from detectron2 import data
from detectron2.model_zoo.model_zoo import get
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
#import os, json, cv2, random, copy

# ----- utilites from detectron2 ----------------------------------------- #
from detectron2 import model_zoo
#from detectron2.engine import DefaultPredictor
#from detectron2.config import get_cfg
#from detectron2.utils.video_visualizer import VideoVisualizer
#from detectron2.utils.visualizer import Visualizer, ColorMode
#from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, build_detection_train_loader 

# ----- for data registration and validation ----------------------------- #
#from detectron2.structures import BoxMode
#from detectron2.data.datasets import load_coco_json, register_coco_instances
#from detectron2.engine import DefaultTrainer
#import random

# ----- for data augmentation -------------------------------------------- #
#from detectron2.data import detection_utils as utils
#import detectron2.data.transforms as T

# ------ for evaluation -------------------------------------------------- #
#from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# ===== Model File Imports =============================================== #

def model_config(cfg, model_file, training_dataset_name, testing_dataset_name,
                output_dir, training_metadata, retrain_threshold):
    """
    detials
    """
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)
    cfg.DATASETS.TRAIN = training_dataset_name
    cfg.DATASETS.TEST = ()
    cfg.OUTPUT_DIR = output_dir
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(training_metadata.thing_classes)
    #cfg.MODEL.RETINANET.NUM_CLASSES = len(training_metadata.thing_classes)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = retrain_threshold
    return cfg

def solver_config(cfg, base_lr, gamma, steps, max_iter, warmup_iter, batch_ims):
    """
    details
    """
    cfg.SOLVER.BASE_LR = base_lr            
    cfg.SOLVER.GAMMA = gamma                 
    cfg.SOLVER.STEPS = steps     
    cfg.SOLVER.MAX_ITER = max_iter             
    cfg.SOLVER.WARMUP_ITERS = warmup_iter         
    cfg.SOLVER.IMS_PER_BATCH = batch_ims           
    return cfg