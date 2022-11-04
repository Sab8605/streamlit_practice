Try:
    import detectron2
except:
    import os 
    os.system('pip install git+https://github.com/facebookresearch/detectron2.git')

from matplotlib.pyplot import axis
import gradio as gr
import requests
import numpy as np
from torch import nn
import cv2
import requests
import torch

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
import os

model_path = "model_final.pth"

cfg = get_cfg()
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = model_path

car_metadata = MetadataCatalog.get("train")
car_metadata.thing_classes = ["Damaged"]
if not torch.cuda.is_available():
    cfg.MODEL.DEVICE='cpu'

predictor = DefaultPredictor(cfg)

def inference(image):
    #im = cv2.imread(image)
    height = image.height

    img = np.array(image.resize((800, height)))
    outputs = predictor(img) 
    v = Visualizer(img[:, :, ::-1],
                   metadata=car_metadata, 
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW  
    )
    #v = Visualizer(img, my_metadata, scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    
    return out.get_image()

    
input = st.file_uploader(label, type=['png', 'jpg'])

if st.button('Click for detection'):
  inference(input)


