import json
import os
import torch
import math

from utils.path import get_project_file_path
from utils.locationTransform import coordinate_transform

def load_data(particle:str,energy:int,total_number:int,allow_pic_number_list:list=[4,3,2,1],limit_min_pix_number:bool=False,ignore_number:int=0,pic_size:int=64,train_type:str="particle",centering:bool=True,use_weight:bool=False,label=None,label_dtype=None):
    jsonFileName=get_project_file_path("data/origin/"+particle+"_"+str(energy)+".json")
    with open(jsonFileName,'r') as f:
        jsonData=json.load(f)
        f.close()

    min_pix=0
    if limit_min_pix_number:
        with open(get_project_file_path("settings.json"),"r") as f:
            settings=json.load(f)
            f.close()
        if particle[0:5]=="gamma":
            par="gamma"
        elif particle[0:6]=="proton":
            par="proton"
        else:
            raise Exception("invalid particle type")
        if settings["loading_min_pix"]["uniformThreshold"]:
            min_pix=settings["loading_min_pix"][par+"Uniform"]
        else:
            if str(energy) in settings["loading_min_pix"][par].keys():
                min_pix=settings["loading_min_pix"][par][str(energy)]

    data_tensor=None
    label_tensor=None
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
            if train_type=="energy":
                # if pic_item["info"]["legal"]==False:
                #     continue
                fin_energy=torch.tensor([
                    [
                        pic_item["info"]["_position"][str(i+1)][0]/100,pic_item["info"]["_position"][str(i+1)][1]/100,
                        # pic_item["info"]["_position_rebuild"][str(i+1)][0],pic_item["info"]["_position_rebuild"][str(i+1)][1],
                        pic_item["info"]["coreX"]/100,pic_item["info"]["coreY"]/100
                    ] for i in range(4)
                ],dtype=torch.float32)
                fin_energy=torch.unsqueeze(fin_energy,dim=0)

            if min_pix!=0:
                pic_number=len(pic_item["1"])+len(pic_item["2"])+len(pic_item["3"])+len(pic_item["4"])
                if pic_number<min_pix:
                    continue
            if train_type=="angle":
                temp_tensor=torch.zeros((pic_size,pic_size),dtype=torch.float32)
                points_list=coordinate_transform(pic_item["1"]+pic_item["2"]+pic_item["3"]+pic_item["4"],400,pic_size,centering,weighted_average=use_weight)
                for item in points_list:
                    temp_tensor[item[0]][item[1]]=item[2]
            else:
                temp_tensor=[torch.zeros((pic_size,pic_size),dtype=torch.float32),torch.zeros((pic_size,pic_size),dtype=torch.float32),torch.zeros((pic_size,pic_size),dtype=torch.float32),torch.zeros((pic_size,pic_size),dtype=torch.float32)]
                for i in range(4):
                    points_list=coordinate_transform(pic_item[str(i+1)],400,pic_size,centering,weighted_average=use_weight)
                    for item in points_list:
                        temp_tensor[i][item[0]][item[1]]=item[2]
            
            if train_type=="energy":
                fin=torch.cat((temp_tensor[0].reshape(1,pic_size,pic_size),temp_tensor[1].reshape(1,pic_size,pic_size),temp_tensor[2].reshape(1,pic_size,pic_size),temp_tensor[3].reshape(1,pic_size,pic_size)))
                # fin=temp_tensor[0]+temp_tensor[1]+temp_tensor[2]+temp_tensor[3]
            elif train_type=="particle":
                fin=torch.cat((torch.cat((temp_tensor[0],temp_tensor[1]),dim=1),torch.cat((temp_tensor[2],temp_tensor[3]),dim=1)))
            elif train_type=="position":
                fin=torch.cat((torch.cat((temp_tensor[0],temp_tensor[1]),dim=1),torch.cat((temp_tensor[2],temp_tensor[3]),dim=1)))
                # fin=torch.cat((temp_tensor[0].reshape(1,pic_size,pic_size),temp_tensor[1].reshape(1,pic_size,pic_size),temp_tensor[2].reshape(1,pic_size,pic_size),temp_tensor[3].reshape(1,pic_size,pic_size)))
                fin_label=torch.tensor([
                    pic_item["info"]["coreX"]/100,pic_item["info"]["coreY"]/100
                ],dtype=torch.float32)
            elif train_type=="angle":
                fin=torch.cat((temp_tensor[0].reshape(1,pic_size,pic_size),temp_tensor[1].reshape(1,pic_size,pic_size),temp_tensor[2].reshape(1,pic_size,pic_size),temp_tensor[3].reshape(1,pic_size,pic_size)))
                # fin=temp_tensor.reshape(1,pic_size,pic_size)
                
                # if not -10<math.tan((pic_item["info"]["priPhi"]-180)*math.pi/180)<10:
                #     continue
                if pic_item["info"]["Cdelta"]==1e3:
                    continue
                fin_label=torch.tensor([
                    # math.tan((pic_item["info"]["priPhi"]-180)*math.pi/180)
                    pic_item["info"]["Cdelta"]/1000,pic_item["info"]["Cr"]/1000
                ],dtype=torch.float32)
                
            else:
                raise Exception("invalid train type")
            
            if data_tensor==None:
                if train_type=="energy":
                    data_tensor=[fin.reshape(1,4,pic_size,pic_size),fin_energy]
                    # data_tensor=fin.reshape(1,1,pic_size,pic_size)
                elif train_type=="particle":
                    data_tensor=fin.reshape(1,1,2*pic_size,2*pic_size)
                elif train_type=="position":
                    data_tensor=fin.reshape(1,1,2*pic_size,2*pic_size)
                    # data_tensor=[fin.reshape(1,4,pic_size,pic_size),fin_energy]
                    label_tensor=fin_label.reshape(1,2)
                elif train_type=="angle":
                    data_tensor=fin.reshape(1,4,pic_size,pic_size)
                    label_tensor=fin_label.reshape(1,2)
            else:
                if train_type=="energy":
                    data_tensor[0]=torch.cat((data_tensor[0],fin.reshape(1,4,pic_size,pic_size)))
                    data_tensor[1]=torch.cat((data_tensor[1],fin_energy))
                    # data_tensor=torch.cat((data_tensor,fin.reshape(1,4,pic_size,pic_size)))
                elif train_type=="particle":
                    data_tensor=torch.cat((data_tensor,fin.reshape(1,1,2*pic_size,2*pic_size)))
                elif train_type=="position":
                    # data_tensor[0]=torch.cat((data_tensor[0],fin.reshape(1,4,pic_size,pic_size)))
                    # data_tensor[1]=torch.cat((data_tensor[1],fin_energy))
                    data_tensor=torch.cat((data_tensor,fin.reshape(1,1,2*pic_size,2*pic_size)))
                    label_tensor=torch.cat((label_tensor,fin_label.reshape(1,2)))
                elif train_type=="angle":
                    data_tensor=torch.cat((data_tensor,fin.reshape(1,4,pic_size,pic_size)))
                    label_tensor=torch.cat((label_tensor,fin_label.reshape(1,2)))
            current_number=current_number+1
            if current_number>=total_number:
                is_break=True
                break
    # print(particle+str(energy)+" loading finish with length: "+str(current_number))
    # if log!=None:
    #     log.write(particle+str(energy)+" loading finish with length: "+str(current_number))
    
    if train_type=="position":
        return data_tensor,label_tensor
    elif train_type=="angle":
        return data_tensor,label_tensor
    else:
        if label!=None:
            return data_tensor,load_label(label,current_number,label_dtype),current_number
        else:
            return data_tensor,None,current_number

def load_label(label,length:int,dtype):
    label_list=[label for _ in range(length)]
    return torch.tensor(label_list,dtype=dtype)
