# USAGE
# python app.py --blur True

from facenet_pytorch import MTCNN
import facedetector
import argparse
import os
from classifier import model_module
import torch
from torchvision import models
import torch.nn as nn

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--blur", type=bool, default=True,
            choices=[True, False])
args = vars(ap.parse_args())

###### FOR SIMPLE CNN ########################################################
#initialize model instance
#classifier = model_module.Net()

# load the model's weights
#classifier.load_state_dict(torch.load('/Users/rosagradilla/Documents/summer20/face_detection/classifier/model.pth'))

###### FOR TRANSFER LEARNING NETWORK #########################################
classifier = models.resnet18(pretrained=True)
classifier.fc = nn.Linear(512, 2)

classifier.load_state_dict(torch.load('/Users/rosagradilla/Documents/summer20/face_detection/classifier/tl_model.pth'))
classifier.eval()

###### INITIATE NETWORKS #######################################################
mtcnn = MTCNN()
fcd = facedetector.FaceDetector(mtcnn, classifier, 'tl_model') # if running with simple CNN: 'simple_cnn'

if args["blur"] == True:
    fcd.run()
if args["blur"] == False:
    fcd.run(blur_setting=False)
