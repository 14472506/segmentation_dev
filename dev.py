# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
import cv2

from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, DefaultPredictor, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
    inference_on_dataset,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger

from solov2.data.dataset_mapper import DatasetMapperWithBasis
from solov2.data.fcpose_dataset_mapper import FCPoseDatasetMapper
from solov2.config import get_cfg
from solov2.checkpoint import CustCheckpointer
#from solov2.evaluation import TextEvaluator

# Other stuff not from adet
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, build_detection_train_loader
 


class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader`/`resume_or_load` method.
    """
    def build_hooks(self):
        """
        Replace `DetectionCheckpointer` with `AdetCheckpointer`.

        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        """
        ret = super().build_hooks()
        for i in range(len(ret)):
            if isinstance(ret[i], hooks.PeriodicCheckpointer):
                self.checkpointer = CustCheckpointer(
                    self.model,
                    self.cfg.OUTPUT_DIR,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
                ret[i] = hooks.PeriodicCheckpointer(self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD)
        return ret
    
    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("adet.trainer")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            self.after_train()

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        #if cfg.MODEL.FCPOSE_ON:
        #    mapper = FCPoseDatasetMapper(cfg, True)
        #else:
        mapper = DatasetMapperWithBasis(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if evaluator_type == "text":
            return TextEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("adet.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
#    cfg.MODEL.FCPOSE_ON = False
    default_setup(cfg, args)


    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")

    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model) # d2 defaults.py
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()
# =============================================================================
# Loading Datasets
# =============================================================================
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

# =============================================================================
# Prediction Stuff
# =============================================================================
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
    evaluator = COCOEvaluator(testing_name, cfg, False, output_dir=fun_output_dir)
    val_loader = build_detection_test_loader(cfg, testing_name)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

# =============================================================================
# Other Stuff atm
# =============================================================================

format = "coco"
data_name = "test"
train_dir = "Data/train2"
train_json = "Data/train2/mod_init_jr_3.json"
thing_classes = ["Jersey Royal", "Handle Bar"]

test_met = data_loader(format, data_name, train_json, train_dir, thing_classes)

# =============================================================================
# Model Config
# =============================================================================
cfg = get_cfg()
cfg.DATASETS.TRAIN = (data_name,)
cfg.merge_from_file("configs/MS_R_50_2x.yaml")

# =============================================================================
# Model Training
# =============================================================================
trainer = Trainer(cfg)
#trainer.resume_or_load(resume=False)
#trainer.train()

format = "coco"
data_name = "val"
train_dir = "Data/val2"
train_json = "Data/val2/init_jr_test.json"
thing_classes = ["Jersey Royal", "Handle Bar"]
outdir = "test_output/"

val_met = data_loader(format, data_name, train_json, train_dir, thing_classes)

model_path = "output/SOLOv2/R_50_2x/model_final.pth"
cfg.MODEL.WEIGHTS = model_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

model_test(cfg, predictor, data_name, outdir)
