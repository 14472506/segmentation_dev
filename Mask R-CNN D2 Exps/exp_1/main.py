"""
Something 
"""

# ===== Imported libraries =============================================== #
from cv2 import imshow
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from pickle import TRUE
from detectron2.config import get_cfg
import cv2
import numpy

from utilities import data_loader, mask_to_poly
from config import model_config, solver_config
from train_and_test import model_train, model_test
from labeller import coco_labeller

# functions
def data_preprocesor(train_config, test_config, thing_classes):

    training_data_dict = {}
    for key in train_config.keys():
        data = train_config[key]
        training_data_dict[key] = data_loader(data[0], data[1], data[2], data[3], thing_classes)

    testing_data_dict = {}
    for key in test_config.keys():
        data = test_config[key]
        testing_data_dict[key] = data_loader(data[0], data[1], data[2], data[3], thing_classes)
    
    return training_data_dict, testing_data_dict

def main(model_conf_list, train_data_name, test_data_name, training_metadata,
        solver_config_list, train):

    cfg = get_cfg()
    model_cfg = model_config(cfg, model_conf_list[0], train_data_name, test_data_name,
                         model_conf_list[1], training_metadata, model_conf_list[2])
    sol_mod_cfg = solver_config(model_cfg, solver_config_list[0], solver_config_list[1],
                                solver_config_list[2], solver_config_list[3], solver_config_list[4],
                                solver_config_list[5])
    
    trainer = model_train(sol_mod_cfg, train)
    print(test_data_name)
    return sol_mod_cfg
    
# ===== Code Execution =================================================== # 
# dataset loading config
training_config_dict = {
    "train1": ["vgg", "train1", "Data/train/potato_d2_dataset.json", "Data/train"],
#    "train2": ["coco", "train2", "Data/train2/mod_init_jr_3.json", "Data/train2"]
}
testing_config_dict = {
    "test1": ["vgg", "test1", "Data/val/potato_d2_dataset.json", "Data/val"],
#    "test2": ["coco", "test2", "Data/val2/init_jr_test.json", "Data/val2"]
}
thing_classes = ["Jersey Royal", "Handle Bar"]

# model solver config
model_config_dict = {
    "R50_FPN": ["COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", "outputs/R50_FPN", 0.05, "test_outputs/R50_FPN"],
    "R101_FPN": ["COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml", "outputs/R101_FPN", 0.05, "test_outputs/R101_FPN"],
    "X101_FPN": ["COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml", "outputs/X101_FPN", 0.05, "test_outputs/X101_FPN"]
}
solver_config_dict = {
    "base": [0.00025, 0.5, [], 1000, 100, 1]
}

# Operation Config
TRAIN = False
TEST = False
LABEL = True


if __name__ == "__main__":
    
    # load training and get training dicitionaries
    training_data_dict, testing_data_dict = data_preprocesor(training_config_dict,
                                                    testing_config_dict, thing_classes)
    
    # execute main code (training optional)
    model_conf_list = model_config_dict["R50_FPN"]
    solver_config_list = solver_config_dict["base"]
    training_data_name = training_data_dict["train1"].name
    testing_data_name = testing_data_dict["test1"].name
    main_cfg = main(model_conf_list, training_data_name, testing_data_name, 
                    training_data_dict["train1"], solver_config_list, TRAIN)
    
    # prediction eval
    model_path = "outputs/R50_FPN/model_final.pth"
    main_cfg.MODEL.WEIGHTS = model_path
    main_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(main_cfg)    
    if TEST == True:
        output = model_test(main_cfg, predictor, testing_data_name, model_conf_list[3])

    if LABEL == True:
        coco_labeller(predictor, "Labeller_location/test_set/")

"""
############################################################################################
    # https://github.com/cocodataset/cocoapi/issues/131
    import numpy as np
    import cv2

    im_mask = output["instances"][0].pred_masks
    im_mask = im_mask.cpu().detach().numpy()
    im_mask = 1*im_mask
    

    points, bbox = mask_to_poly(im_mask)

    from PIL import Image, ImageDraw

    width = im_mask[0].shape[1]
    height = im_mask[0].shape[0]

    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(points, outline=1, fill=1)
    rec_mask = numpy.array(img)

    from matplotlib import pyplot as plt
    plt.imshow(rec_mask, interpolation='nearest')
    plt.show()
    
    print(bbox)
    #cv2.imshow("image", v.get_image()[:, :, ::-1])
    #cv2.waitKey(0)
"""