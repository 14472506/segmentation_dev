import os
import numpy as np
import cv2
import json
from datetime import date
from pycocotools import mask as cocomask
import io

from utilities import mask_to_poly, binary_mask_to_rle, binary_mask_to_poly

def coco_labeller(predictor, data_dir):


    info_dict = {
        "description": str("Test Coco JSON Production"),
        "url": str("https://non-maybeputmyemailhere"),
        "version": str("1.0"),       
        "year": int(2022),
        "contributor": str("Me"),
        "date_created": "02/11/1995"
    }

    licence_dict = [{
        "url": str("http://creativecommons.org/licenses/by-nc-sa/2.0/"),
        "id": int(1),
        "name": str("Attribution-NonCommercial-ShareAlike License")        
    }]
    
    category_dict = [{
        "name": str("Jersey Royal"),        
        "id": int(1),
        "supercategory": str("")
    },{
        "name": str("Handle Bar"),
        "id": int(2),
        "supercategory": str("")
    }]

    # loop initialisation
    img_count = 0
    anno_count = 0 
    image_dict = []
    anno_dict = []
    
    # looping through images in file
    for file in os.listdir(data_dir):

        # laoding image from file
        img_str = data_dir + str(file)
        img = cv2.imread(img_str)
        
        # making image dict entry
        im_dict = {
            "id": int(img_count),            
            "height": int(img.shape[0]),
            "width": int(img.shape[1]),
            "file_name": str(file),
            "license": int(1),
            "date_captured": "02/11/1995"
        }       
        
        # append to dict
        image_dict.append(im_dict)

        # generating predicition
        pred_data = predictor(img)
        
        # getting number of instances in image
        instance_nums = pred_data["instances"].pred_classes.cpu().detach().numpy().shape[0]
        
        # looping through instances in image
        for instance in range(instance_nums):
            
            cat_id = int(pred_data["instances"][instance].pred_classes.cpu().detach().numpy()[0] + 1)
            mask = pred_data["instances"][instance].pred_masks.cpu().detach().numpy()*1
            mask = mask[0].astype(np.uint8)
            
            """
            _, bbox = mask_to_poly(mask)
            rle = binary_mask_to_rle(mask[0]) 
            
            area = cocomask.area(rle)
            """
            
            segmentation, bbox, area = binary_mask_to_poly(mask) 
                       
            # current annotation dict
            ann_dict = {
                "id": int(anno_count),
                "image_id": int(img_count),
                "category_id": int(cat_id),
                "bbox": bbox,
                "segmentation": [segmentation],
                "area": float(area),
                "iscrowd": int(0) 
            }
            
            # append current dict
            anno_dict.append(ann_dict)
            
            # add 1 to annotation cound            
            anno_count += 1
            
        # add to dict count value
        img_count += 1
        
    coco_dict = {
        "info": info_dict,
        "licenses": licence_dict,
        "images": image_dict,
        "annotations": anno_dict,
        "categories": category_dict 
    }
    
    with open("labelled.json", "w") as outfile:        
        json.dump(coco_dict, outfile) 
        
#def coco_labeller_2(predictor, data_dir)      