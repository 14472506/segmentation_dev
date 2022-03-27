"""
Details
"""
# ------ Imported libraries for code ------------------------------------- #
"""
Note, all libraries coppied over from bd2.py
"""
# base libraries
from curses import meta
from email.policy import default
import enum
from lzma import PRESET_DEFAULT
from pyexpat import model
import threading
from turtle import width
from cv2 import imread
from detectron2 import data
from detectron2.model_zoo.model_zoo import get
from importlib_metadata import metadata
import torch, torchvision
from torch.utils.data import dataset
import time

# setting up detectron logger
import detectron2
from detectron2 import modeling
from detectron2.data import build, datasets
from detectron2.utils.logger import setup_logger
from tqdm import tqdm
setup_logger()

# some common libraries
import numpy as np
import tqdm  
import os, json, cv2, random, copy

# utilites from detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, build_detection_train_loader 

# for data registration and validation
from detectron2.structures import BoxMode
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.engine import DefaultTrainer
import random

# for data augmentation
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T

# for evaluation
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# ------ functions ------------------------------------------------------- #
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

def predictor_init(model_file, model_path, test_threshold):
    """
    detials
    """
    # initialise config object
    cfg = get_cfg()
    
    # configure model
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.MODEL.WEIGHTS = model_path
    #cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = test_threshold
   
    
    # get predictor
    predictor = DefaultPredictor(cfg)
    
    # return cfg and predictor
    return(cfg, predictor)

# broken, same problem as bd2 but should mean when fixed it works?  
def testing(model_file, model_path, test_threshold, test_dataset_name, test_output_loc):
    """
    details
    """
    # collecting config and predictor
    cfg, predictor = predictor_init(model_file, model_path, test_threshold)
    
    # initialising evaluator and loader
    evaluator = COCOEvaluator(test_dataset_name, output_dir=test_output_loc)
    val_loader = build_detection_test_loader(cfg, test_dataset_name)
    
    # run print out
    print(inference_on_dataset(predictor, val_loader, evaluator))
            
def image_vis(model_file, model_path, test_threshold, image_path, input_metadata):
    """
    detials
    """
    # get predictor and image
    _, predictor = predictor_init(model_file, model_path, test_threshold)
    im = cv2.imread(image_path)
    
    # generate prediction
    output = predictor(im)
    
    # initialise visualiser
    v = Visualizer(im[:, :, ::-1], metadata=input_metadata, scale=0.8, instance_mode=ColorMode.IMAGE_BW)
    
    # predict and show
    v = v.draw_instance_predictions(output["instances"].to("cpu"))
    cv2.imshow("image", v.get_image()[:, :, ::-1])
    cv2.waitKey()  
    
def video_vis(video_path, model_file, model_path, test_threshold, input_metadata):
    """
    details
    """
    # Extract video properties
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    video_writer = cv2.VideoWriter('Data/videos/vid_prediction/base_2.mp4', fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=float(frames_per_second), frameSize=(width, height), isColor=True)

    # initialise predictor
    _, predictor = predictor_init(model_file, model_path, test_threshold)

    # Initialize visualizer
    v = VideoVisualizer(input_metadata, ColorMode.IMAGE)

    def runOnVideo(video, maxFrames):
        """ Runs the predictor on every frame in the video (unless maxFrames is given),
        and returns the frame with the predictions drawn.
        """

        readFrames = 0
        while True:
            hasFrame, frame = video.read()
            if not hasFrame:
                break

            # Get prediction results for this frame
            outputs = predictor(frame)

            # Make sure the frame is colored
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Draw a visualization of the predictions using the video visualizer
            visualization = v.draw_instance_predictions(frame, outputs["instances"].to("cpu"))

            # Convert Matplotlib RGB format to OpenCV BGR format
            visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)

            yield visualization

            readFrames += 1
            if readFrames > maxFrames:
                break

    # Create a cut-off for debugging
    num_frames = 1571

    # Enumerate the frames of the video
    for visualization in tqdm.tqdm(runOnVideo(video, num_frames), total=num_frames):

        # Write test image
        cv2.imwrite('Data/videos/vid_prediction/POSE detectron2.png', visualization)

        # Write to video file
        video_writer.write(visualization)

    # Release resources
    video.release()
    video_writer.release()
    cv2.destroyAllWindows()
    
def speed_benchmark(model_file, model_path, test_threshold, image_path):
    
    _, predictor = predictor_init(model_file, model_path, test_threshold)
    im = cv2.imread(image_path)
    times = []
    
    for i in range(20):
        start_time = time.time()
        outputs = predictor(im)
        delta = time.time() - start_time
        times.append(delta)
    mean_delta = np.array(times).mean()
    fps = 1 / mean_delta
    print("Average(sec):{:.2f},fps:{:.2f}".format(mean_delta, fps))

def prediction_producer():
    """
    details
    """
    
# ----- main function ---------------------------------------------------- #
def main(dataset_format, dataset_name, json_location, data_location, thing_classes_data, 
         model_file, model_path, test_threshold, test_output_loc, image_path):
    """
    detials
    """
        
    # load data set
    test_metadata = data_loader(dataset_format, dataset_name, json_location, data_location, thing_classes_data)
    # testing
    #testing(model_file, model_path, test_threshold, dataset_name, test_output_loc)
    
    # view image prediction
    #image_vis(model_file, model_path, test_threshold, image_path, test_metadata)
    
    # view video prediction
    #video_vis("Data/videos/test_vid2.webm", model_file, model_path, test_threshold, test_metadata)
    
    # speed benchmark
    speed_benchmark(model_file, model_path, test_threshold, image_path)   
# ----- initialisation --------------------------------------------------- #
# flags
init_test = True
    
# Dataset detials
dataset_format = "coco"
dataset_name = "jr_potatoes"
json_location = "Data/train2/mod_init_jr_3.json"
dataset_location = "Data/train2"
thing_classes_data = ["Jersey Royal", "Handle Bar"]

# Testing detials
#model_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
model_file = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
model_path = "outputs/resx_101_full/model_final.pth"
test_threshold = 0.9
test_output_loc = "outputs/resx_101_full/test_results"

# image prediction detials
image_path = "Data/val/image_25.jpg"

if __name__ == "__main__":
    """
    detials
    """    
    # call main
    main(dataset_format, dataset_name, json_location, dataset_location, thing_classes_data, 
         model_file, model_path, test_threshold, test_output_loc, image_path)