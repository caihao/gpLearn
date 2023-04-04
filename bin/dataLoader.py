import json
import os
import torch

from utils.path import get_project_file_path
from utils.locationTransform import coordinate_transform
from utils.log import Log

def load_data(particle:str,energy:int,total_number:int,allow_pic_number_list:list=[4,3,2,1],allow_min_pix_number:int=None,ignore_number:int=0,pic_size:int=64,mode:str="joint",centering:bool=True,use_weight:bool=False,label=None,label_dtype=None,log:Log=None):
    jsonFileName=get_project_file_path("data/origin/"+particle+"_"+str(energy)+".json")
    with open(jsonFileName,'r') as f:
        jsonData=json.load(f)
        f.close()
    
    data_tensor=None
    current_number=0
    current_ignore=0
    is_break=False
    for pic in allow_pic_number_list:
        if is_break:
            break
        data_all_list=jsonData["c"+str(pic)]
        for pic_item in data_all_list:
            if current_ignore<ignore_number:
                current_ignore=current_ignore+1
                continue
            if allow_min_pix_number:
                pic_number=len(pic_item["1"])+len(pic_item["2"])+len(pic_item["3"])+len(pic_item["4"])
                if pic_number<allow_min_pix_number:
                    continue
            temp_tensor=[torch.zeros((pic_size,pic_size),dtype=torch.float32),torch.zeros((pic_size,pic_size),dtype=torch.float32),torch.zeros((pic_size,pic_size),dtype=torch.float32),torch.zeros((pic_size,pic_size),dtype=torch.float32)]
            for i in range(4):
                points_list=coordinate_transform(pic_item[str(i+1)],400,pic_size,centering,weighted_average=use_weight)
                for item in points_list:
                    temp_tensor[i][item[0]][item[1]]=item[2]
            if mode=="overlay":
                fin=torch.cat((temp_tensor[0].reshape(1,pic_size,pic_size),temp_tensor[1].reshape(1,pic_size,pic_size),temp_tensor[2].reshape(1,pic_size,pic_size),temp_tensor[3].reshape(1,pic_size,pic_size)))
            elif mode=="joint":
                fin=torch.cat((torch.cat((temp_tensor[0],temp_tensor[1]),dim=1),torch.cat((temp_tensor[2],temp_tensor[3]),dim=1)))
            else:
                raise Exception("invalid_mode")
            if data_tensor==None:
                if mode=="overlay":
                    data_tensor=fin.reshape(1,4,pic_size,pic_size)
                elif mode=="joint":
                    data_tensor=fin.reshape(1,1,2*pic_size,2*pic_size)
            else:
                if mode=="overlay":
                    data_tensor=torch.cat((data_tensor,fin.reshape(1,4,pic_size,pic_size)))
                elif mode=="joint":
                    data_tensor=torch.cat((data_tensor,fin.reshape(1,1,2*pic_size,2*pic_size)))
            current_number=current_number+1
            if current_number>=total_number:
                is_break=True
                break
    print(particle+str(energy)+" loading finish with length: "+str(len(data_tensor)))
    if log!=None:
        log.write(particle+str(energy)+" loading finish with length: "+str(len(data_tensor)))
    if label!=None:
        return data_tensor,load_label(label,len(data_tensor),label_dtype)
    else:
        return data_tensor,None

def load_label(label,length:int,dtype):
    label_list=[label for _ in range(length)]
    return torch.tensor(label_list,dtype=dtype)
