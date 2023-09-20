import torch

from utils.path import get_project_file_path
from model.ResNet_old import ResNet_old
from model.LinearNet import LinearNet
from model.DenseNet import DenseNet
from model.MSDNet import MSDNet
from model.EnergyNet import EnergyNet
from model.LinearNetConv import LinearNetConv
from model.PointNet import PointNet
from model.AngleNet import AngleNet,ResAngleNet,ResAngleNet2D
from model.ParticleNet import ParticleNet
from model.ResNet import ResNet18
from model.ResNet import ResNet34

def initializeModel(input_channel:int,input_size_x:int,input_size_y:int,output_size:int,model_type:str,input_info:int=None):
    if model_type=="ParticleNet":
        model=ParticleNet(input_channel,input_size_x,input_size_y,output_size)
    elif model_type=="EnergyNet":
        model=EnergyNet(input_size_x,input_size_y,input_info,output_size)
    elif model_type=="PointNet":
        model=PointNet(input_channel,input_size_x,input_size_y,output_size)
    elif model_type=="AngleNet":
        model=AngleNet(input_channel,input_size_x,input_size_y,output_size)

    elif model_type=="ResNet_old":
        model=ResNet_old(input_channel,input_size_x,input_size_y,output_size,init_weights=True)
    elif model_type=="LinearNet":
        model=LinearNet(input_channel,input_size_x,input_size_y,output_size,init_weights=True)
    elif model_type=="DenseNet":
        model=DenseNet(input_channel,input_size_x,input_size_y,output_size,init_weights=True)
    elif model_type=="MSDNet":
        model=MSDNet(input_channel,input_size_x=input_size_x,input_size_y=input_size_y,num_classes=output_size,init_weights=True)
    elif model_type=="LinearNetConv":
        model=LinearNetConv(1,input_size_x,input_size_y,input_info,4,16,output_size)
    elif model_type=="ResAngleNet":
        model=ResAngleNet(input_channel,input_size_x,input_size_y,output_size)
    elif model_type=="ResAngleNet2D":
        model=ResAngleNet2D(input_channel,input_size_x,input_size_y,output_size)
    elif model_type=="ResNet18":
        model=ResNet18(input_channel,input_size_x,output_size)
    elif model_type=="ResNet34":
        model=ResNet34(input_channel,input_size_x,output_size)
    else:
        raise Exception("invalid model type")
    
    return model
