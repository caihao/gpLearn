import torch

from utils.path import get_project_file_path
from model.ResNet import ResNet
from model.ResNet_0315 import ResNet_0315
from model.LinearNet import LinearNet
from model.DenseNet import DenseNet
from model.ResNet_34 import ResNet_34
from model.ResNet_34_Self import ResNet_34_Self
from model.ResNet_50 import ResNet_50,Bottleneck
from model.ResNet_0318 import ResNet_0318
from model.MSDNet import MSDNet
from model.DeepCore import DeepCore
from model.InceptionNet import InceptionNet
from model.LinearResNet import LinearResNet
from model.GoogLeNet import GoogLeNet
from model.LinearNetConv import LinearNetConv
from model.PointNet import PointNet
from model.AngleNet import AngleNet

def initializeModel(model_name:str,input_channel:int,input_size_x:int,input_size_y:int,output_size:int,model_type:str,other_init_keys:list=None,input_info:int=None):
    if model_type=="ResNet":
        model=ResNet(input_channel,input_size_x,input_size_y,output_size,init_weights=True)
    elif model_type=="LinearNet":
        model=LinearNet(input_channel,input_size_x,input_size_y,output_size,init_weights=True)
    elif model_type=="ResNet_0315":
        model=ResNet_0315(input_channel,input_size_x,input_size_y,output_size,init_weights=True)
    elif model_type=="DenseNet":
        model=DenseNet(input_channel,input_size_x,input_size_y,output_size,init_weights=True)
    elif model_type=="ResNet_34":
        model=ResNet_34(input_channel,input_size_x,input_size_y,output_size,init_weights=True)
    elif model_type=="ResNet_34_Self":
        model=ResNet_34_Self(input_channel,input_size_x,input_size_y,output_size,init_weights=True)
    elif model_type=="ResNet_50":
        model=ResNet_50(Bottleneck,[3,4,6,3],input_channal=input_channel,zero_init_residual=True)
    elif model_type=="ResNet_0318":
        model=ResNet_0318(input_channel,input_size_x,input_size_y,output_size,init_weights=True)
    elif model_type=="MSDNet":
        model=MSDNet(input_channel,input_size_x=input_size_x,input_size_y=input_size_y,num_classes=output_size,init_weights=True)
    elif model_type=="DeepCore":
        model=DeepCore()
    elif model_type=="InceptionNet":
        model=InceptionNet(input_channel,input_size_x,input_size_y,output_size)
    elif model_type=="LinearResNet":
        model=LinearResNet(input_channel,input_size_x,input_size_y,output_size)
    elif model_type=="GoogLeNet":
        model=GoogLeNet(input_size_x,input_size_y,input_info,output_size)
    elif model_type=="LinearNetConv":
        model=LinearNetConv(1,input_size_x,input_size_y,input_info,4,16,output_size)
    elif model_type=="PointNet":
        model=PointNet(input_channel,input_size_x,input_size_y,output_size)
    elif model_type=="AngleNet":
        model=AngleNet(input_channel,input_size_x,input_size_y)
    else:
        raise Exception("invalid model type")
    
    model_state={'model':model}
    for key in other_init_keys:
        model_state[key]=0
    if model_name[-3:]==".pt":
        torch.save(model_state,get_project_file_path("data/model/"+model_name))
    else:
        torch.save(model_state,get_project_file_path("data/model/"+model_name+".pt"))

