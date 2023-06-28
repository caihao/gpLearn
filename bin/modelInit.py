import torch

from utils.path import get_project_file_path
from model.ResNet import ResNet
from model.LinearNet import LinearNet
from model.DenseNet import DenseNet
from model.MSDNet import MSDNet
from model.GoogLeNet import GoogLeNet
from model.LinearNetConv import LinearNetConv
from model.PointNet import PointNet
from model.AngleNet import AngleNet
from model.ResNet_custom import ResNet_custom

def initializeModel(model_name:str,input_channel:int,input_size_x:int,input_size_y:int,output_size:int,model_type:str,other_init_keys:list=None,input_info:int=None):
    if model_type=="ResNet":
        model=ResNet(input_channel,input_size_x,input_size_y,output_size,init_weights=True)
    elif model_type=="LinearNet":
        model=LinearNet(input_channel,input_size_x,input_size_y,output_size,init_weights=True)
    elif model_type=="DenseNet":
        model=DenseNet(input_channel,input_size_x,input_size_y,output_size,init_weights=True)
    elif model_type=="MSDNet":
        model=MSDNet(input_channel,input_size_x=input_size_x,input_size_y=input_size_y,num_classes=output_size,init_weights=True)
    elif model_type=="GoogLeNet":
        model=GoogLeNet(input_size_x,input_size_y,input_info,output_size)
    elif model_type=="LinearNetConv":
        model=LinearNetConv(1,input_size_x,input_size_y,input_info,4,16,output_size)
    elif model_type=="PointNet":
        model=PointNet(input_channel,input_size_x,input_size_y,output_size)
    elif model_type=="AngleNet":
        model=AngleNet(input_channel,input_size_x,input_size_y)
    elif model_type=="ResNet_custom":
        model=ResNet_custom(input_channel,input_size_x,input_size_y,output_size)
    else:
        raise Exception("invalid model type")
    
    model_state={'model':model}
    for key in other_init_keys:
        model_state[key]=0
    if model_name[-3:]==".pt":
        torch.save(model_state,get_project_file_path("data/model/"+model_name))
    else:
        torch.save(model_state,get_project_file_path("data/model/"+model_name+".pt"))

