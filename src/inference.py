#!/usr/bin/env python3
"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import os
import time
import argparse
import logging
import numpy as np
import torch
import json

from timm.models import create_model
#from timm.data import ImageDataset, create_loader, resolve_data_config
#from timm.utils import AverageMeter, setup_default_logging
#import pdb
from torchvision import transforms
import PIL
from PIL import Image
from torch.autograd import Variable
import requests
from io import BytesIO





def image_loader(image_name):
    """load image, returns cuda tensor"""
    response = requests.get(image_name)
    img = Image.open(BytesIO(response.content))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_norm = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((224, 224)), transforms.Normalize(mean, std)])
    # get normalized image
    img_normalized = transform_norm(img).float()
    img_normalized.unsqueeze_(0)
    return img_normalized


parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--model', metavar='MODEL', default='resnet18', help='model architecture (default: resnet18)')
parser.add_argument('--image',metavar='DIR',help='path to dataset')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')



def main():

    args = parser.parse_args()
    # might as well try to do something useful...
    # create model
    model_path = '/workspace/emlo_v2_cifar10_hydra/logs/train/runs/2022-11-28_17-02-28/model.script.pt'
    model = torch.jit.load(model_path)


 

    #model = create_model(args.model,num_classes=args.num_classes, in_chans=3, pretrained=args.pretrained)
    

    image_src='https://www.wwf.org.uk/sites/default/files/styles/social_share_image/public/2018-10/Large_WW1113482.jpg?itok=Bluh496C'
    image = image_loader(image_src)
    with torch.no_grad():
        model.eval()
        labels = model(image)
        prob_labels = torch.softmax(labels, dim=1)
        value,index = torch.max(prob_labels,dim=1)
        with open("imagenet_classes.txt", "r") as f:
            labels = [s.strip() for s in f.readlines()]

        # with open('imagenet-simple-labels.json') as f:
        #     labels = json.load(f)

        def class_id_to_label(i):
            return labels[i]

        label = class_id_to_label(index.item())
        json_string = json.dumps({"predicted": label, "confidence": value.item()})
        print(json_string)
        return json_string



if __name__ == '__main__':
    main()