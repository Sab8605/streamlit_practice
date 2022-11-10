try:
    import detectron2
except:
    import os 
    os.system('pip install git+https://github.com/facebookresearch/detectron2.git')

import streamlit as st



import numpy as np

# import cv2
# import requests
import torch
from PIL import Image
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
import os


def load_image(image_file):
	img = Image.open(image_file)
	return img

...
st.write("Instance Segmentation")
st.subheader("Image")
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

if image_file is not None:

    st.image(load_image(image_file),width=250)

model_path = "model_final_instance_segmentation.pth"

cfg = get_cfg()
# Force model to operate within CPU, erase if CUDA compatible devices ara available
if not torch.cuda.is_available():
    cfg.MODEL.DEVICE='cpu'
# Add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# Set threshold for this model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_path
# Initialize prediction model



from detectron2.data.datasets import register_coco_instances
register_coco_instances(f"Helmet_train1",{},  f"Train/_annotations.coco.json", f"Train")
Helmet_metadata = MetadataCatalog.get("Helmet_train1")
Helmet_metadata.thing_classes = ["Helmet"]


predictor = DefaultPredictor(cfg)

def inference(image):
    #im = cv2.imread(image)
    height = image.height

    img = np.array(image.resize((800, height)))
    outputs = predictor(img) 
    v = Visualizer(img[:, :, ::-1],
                   metadata= Helmet_metadata, 
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW  
    )
    #v = Visualizer(img, my_metadata, scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
    return out.get_image()

if st.button('Click for detection'):
  inference(image_file)
  st.image(processed_img, caption='Processed Image', use_column_width=True)
