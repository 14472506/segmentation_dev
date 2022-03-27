# ------ Imported libraries for code ------------------------------------- #
# base libraries
from email.policy import default
from lzma import PRESET_DEFAULT
import threading
from turtle import width
from cv2 import imread
from detectron2 import data
from detectron2.model_zoo.model_zoo import get
import torch, torchvision
from torch.utils.data import dataset

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

# ----- Classes and Functions -------------------------------------------------------- #
# --- Classes --- #
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

# --- Data Loading --- #
def data_loading(dataset_name, json_file, data_dir):
    """
    detials
    """
    register_coco_instances(dataset_name, {}, json_file, data_dir)
    #data_dict = load_coco_json(json_file, data_dir, dataset_name=dataset_name)
    data_metadata = MetadataCatalog.get(dataset_name).set(thing_classes=["Jersey Royal", "Handle Bar"])
    return data_metadata

def data_loading_to_coco(data_dir, json_file):
    
    with open(json_file) as f:
        imgs_anns = json.load(f)
    
    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        file_name = os.path.join(data_dir, v['filename'])
        height, width = cv2.imread(file_name).shape[:2]
        
        record["file_name"] = file_name
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        
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
        dataset_dicts.append(record)
    
    return(dataset_dicts)       
   
def data_checker(data_dir, json_file, ds_metadata,rand_val=3):
    
    dataset_dicts = data_loading_to_coco(data_dir, json_file)
    for d in random.sample(dataset_dicts, rand_val):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=ds_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow("data_checker", out.get_image()[:, :, ::-1]) 
        cv2.waitKey()       

# --- Augmentation and config --- #        
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

def model_config(cfg, model_file, training_dataset_name, testing_dataset_name,
                output_dir, training_metadata, retrain_threshold):
    """
    detials
    """
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)
    cfg.DATASETS.TRAIN = (training_dataset_name, "jr_training_2",)
    cfg.DATASETS.TEST = ()
    cfg.OUTPUT_DIR = output_dir
    cfg.MODEL.ROI_HEADS.NUN_CLASSES = len(training_metadata.thing_classes)
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

# --- Training --- #
def model_train(cfg, TRAIN):
    """
    detials
    """
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    if TRAIN == True:
        trainer.train()
    return(trainer)

# --- Testing and Visualisation --- #
def predictor_gen(cfg, model_path, test_threshold):
    """
    details
    """
    cfg.MODEL.WEIGHTS = model_path  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = test_threshold   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    return(predictor)
    
def model_test(cfg, predictor, testing_name, fun_output_dir):
    """
    details
    """    
    evaluator = COCOEvaluator(testing_name, output_dir=fun_output_dir)
    val_loader = build_detection_test_loader(cfg, testing_name)
    print(inference_on_dataset(predictor, val_loader, evaluator))

def image_predict(file_name, predictor, testing_metadata):
   
    im = cv2.imread(file_name)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=testing_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("image", v.get_image()[:, :, ::-1])
    cv2.waitKey()  

def video_prediction(video_file, predictor, meta_data):
    """
    This need to be tested on a video
    """
    
    # Extract Video Properties
    video = cv2.VideoCapture(video_file)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # initialise video writer
    video_writer = cv2.VideoWriter(video_file, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=float(video_fps), frameSize=(width, height), isColor=True)
    
    # Initialise Visualiser
    vis = VideoVisualizer(meta_data, ColorMode.IMAGE)
    
    def runOnVideo(video, maxFrames):
        
        readFrames = 0
        while True:

            # get and check frame data
            hasFrame, frame = video.read()
            if not hasFrame:
                break
            
            # get prediction resultsfrom frame
            outputs = predictor(frame)

            # check frame colour
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Draw a visualisation of the predictions usin the visualiser
            visualisation = vis.draw_instance_predictions(frame, outputs["instance"].to("cpu"))

            # convert Matplotlib RGB to OpenCV BGR format
            visualisation = cv2.cvtColor(visualisation.get_image(), cv2.COLOR_BAYER_BG2BGR)

            yield visualisation

            readFrames += 1
            if readFrames > maxFrames:
                break
    
    # Create a cut-off for debugging
    cut_off_frames = 120
    
    # Enumerate the frames of the video
    for visualisation in tqdm.tqdm(runOnVideo(video, num_frames), total=num_frames):
        
        # Write test image
        cv2.imwrite('POSE detectron2.png', visualisation)
        
        # write to video file
        video_writer.write(visualisation)
    
    # release source
    video.release()
    video_writer.release()
    cv2.destroyAllWindows()
    
# ----- main ------------------------------------------------------------- #
def main(training_dataset_name, training_json_file, training_data_dir, 
        testing_dataset_name, testing_json_file, testing_data_dir,
        model_file, output_dir, retrain_threshold, base_lr, gamma, steps,
        max_iter, warmup_iter, batch_ims, model_path, test_threshold, TRAIN, TEST):

    # loading training & test data
    training_metadata = data_loading("jr_training_2", "/Data/train2/mod_init_jr_3.json", "train2")    
    testing_metadata = data_loading("jr_testing_2", "/Data/val2/init_jr_test.json", "val2")
    
    
    # training data
    DatasetCatalog.register(training_dataset_name, lambda: data_loading_to_coco(training_data_dir, training_json_file))
    MetadataCatalog.get(training_dataset_name).set(thing_classes=["Jersey Royal", "Handle Bar"])
    training_metadata = MetadataCatalog.get(training_dataset_name)

    # test data
    DatasetCatalog.register(testing_dataset_name, lambda: data_loading_to_coco(testing_data_dir, testing_json_file))
    MetadataCatalog.get(testing_dataset_name).set(thing_classes=["Jersey Royal", "Handle Bar"])
    testing_metadata = MetadataCatalog.get(testing_dataset_name)
    
    #data_checker(training_data_dir, training_json_file, training_metadata, 20)
    
    # configuring model and solver
    cfg = get_cfg()
    
    mod_cfg = model_config(cfg, 
                        model_file, 
                        training_dataset_name,
                        testing_dataset_name,
                        output_dir,
                        training_metadata,
                        retrain_threshold)
    
    mod_solv_cfg = solver_config(mod_cfg,
                                base_lr,
                                gamma,
                                steps,
                                max_iter,
                                warmup_iter,
                                batch_ims)
    
    
    trainer = model_train(mod_solv_cfg, TRAIN)
    
    predictor = predictor_gen(mod_solv_cfg, model_path, test_threshold)
    
    if TEST == True:
        output = model_test(mod_solv_cfg, predictor, testing_dataset_name, output_dir)
    
    #image_predict("pred_test/3.jpg", predictor, testing_metadata)
    
    #This needs to be tested
    video_prediction("Data/test_vid.mp4", predictor, testing_metadata)

if __name__=="__main__":

    # training and test data set config
    training_dataset_name = "jr_training"
    training_json_file = "Data/train/potato_d2_dataset.json" #"train/potato_d2_dataset.json"
    training_data_dir = "Data/train" 
    testing_dataset_name = "jr_testing"
    testing_json_file = "Data/val/potato_d2_dataset.json"
    testing_data_dir = "Data/val"

    # model config param
    model_file = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml" 
    output_dir = "output"
    retrain_threshold = 0.05 
    
    # solver config param
    base_lr = 0.00025
    gamma = 0.5
    steps = []
    max_iter = 1000
    warmup_iter = 100
    batch_ims = 1

    # test config
    model_path = "output/model_final.pth"
    test_threshold = 0.7

    # train test settings
    TRAIN = False
    TEST = False

    main(training_dataset_name, training_json_file, training_data_dir, 
        testing_dataset_name, testing_json_file, testing_data_dir,
        model_file, output_dir, retrain_threshold, base_lr, gamma, steps,
        max_iter, warmup_iter, batch_ims, model_path, test_threshold, TRAIN, TEST)
    
    ## address testing issues ##