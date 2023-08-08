import json
import os
import torch
import math
import pickle
import re

from utils.path import get_project_file_path
from utils.locationTransform import coordinate_transform,coordinate_transform_angle
from utils.log import Log    

def load_data(particle:str,energy:int,total_number:int,allow_pic_number_list:list=[4,3,2,1],limit_min_pix_settings:json=None,ignore_number:int=0,pic_size:int=64,train_type:str="particle",centering:bool=True,use_weight:bool=False,label=None,label_dtype=None,settings:json=None,log:Log=None):
    jsonFileName=get_project_file_path("data/origin/"+particle+"_"+str(energy)+".json")
    with open(jsonFileName,'r') as f:
        jsonData=json.load(f)
        f.close()

    min_pix=0
    if limit_min_pix_settings:
        # with open(get_project_file_path("settings.json"),"r") as f:
        #     settings=json.load(f)
        #     f.close()
        if particle[0:5]=="gamma":
            par="gamma"
        elif particle[0:6]=="proton":
            par="proton"
        else:
            raise Exception("invalid particle type")
        if limit_min_pix_settings["loading_min_pix"]["uniformThreshold"]:
            min_pix=limit_min_pix_settings["loading_min_pix"][par+"Uniform"]
        else:
            if str(energy) in limit_min_pix_settings["loading_min_pix"][par].keys():
                min_pix=limit_min_pix_settings["loading_min_pix"][par][str(energy)]

    data_file_name=None
    old_file_name=None
    if settings:
        for load_path in [get_project_file_path("data/temp")]+settings["tempData"]["loadPath"]:
            for file_name in os.listdir(load_path):
                pattern="(.*)\((\d+)\).data"
                matches=re.findall(pattern,file_name)
                if len(matches)==0:
                    continue
                if matches[0][0]==(particle+"_"+str(energy)+"_"+str(allow_pic_number_list)+"_"+str(min_pix)+"_"+str(pic_size)+"_"+str(train_type)+"_"+str(centering)+"_"+str(use_weight)):
                    if total_number<=int(matches[0][1]):
                        data_file_name=os.path.join(load_path,file_name)
                    else:
                        old_file_name=os.path.join(load_path,file_name)
            if data_file_name!=None:
                break
    
    index_info=None
    if data_file_name:
        with open(data_file_name,"rb") as f:
            pcl=pickle.load(f)
            f.close()  
        if train_type=="energy":
            data_tensor=[torch.tensor(pcl["data"][0][:total_number],dtype=torch.float32),torch.tensor(pcl["data"][1][:total_number],dtype=torch.float32)]
            data_tensor=torch.tensor(pcl["label"][:total_number],dtype=torch.float32)
        elif train_type=="particle":
            data_tensor=torch.tensor(pcl["data"][:total_number],dtype=torch.float32)
            label_tensor=torch.tensor(pcl["label"][:total_number],dtype=torch.int64)
        else:
            data_tensor=torch.tensor(pcl["data"][:total_number],dtype=torch.float32)
            label_tensor=torch.tensor(pcl["label"][:total_number],dtype=torch.float32)
        
        print(particle+"_"+str(energy)+" was loaded from the cache file: "+data_file_name)
        if log:
            log.write(particle+"_"+str(energy)+" was loaded from the cache file: "+data_file_name)
        
        return data_tensor,label_tensor,total_number,min_pix

    elif old_file_name:
        with open(old_file_name,"rb") as f:
            pcl=pickle.load(f)
            f.close()  
        if train_type=="energy":
            data_tensor=[torch.tensor(pcl["data"][0][:total_number],dtype=torch.float32),torch.tensor(pcl["data"][1][:total_number],dtype=torch.float32)]
            data_tensor=torch.tensor(pcl["label"][:total_number],dtype=torch.float32)
        elif train_type=="particle":
            data_tensor=torch.tensor(pcl["data"][:total_number],dtype=torch.float32)
            label_tensor=torch.tensor(pcl["label"][:total_number],dtype=torch.int64)
        else:
            data_tensor=torch.tensor(pcl["data"][:total_number],dtype=torch.float32)
            label_tensor=torch.tensor(pcl["label"][:total_number],dtype=torch.float32)
        index_info=pcl["index_info"]
        current_number=int(matches[0][1])

        print(particle+"_"+str(energy)+" cache file ("+old_file_name+") updating...")
        if log:
            log.write(particle+"_"+str(energy)+" cache file ("+old_file_name+") updating...")

    else:
        data_tensor=None
        label_tensor=None
        current_number=0

    current_index_info={"1":0,"2":0,"3":0,"4":0}
    is_break=False

    for pic in allow_pic_number_list:
        if is_break:
            break
        data_all_list=jsonData["c"+str(pic)]
        end_index=len(data_all_list)
        # 判断开始序号
        if index_info:
            if index_info[str(pic)]>=(end_index-1):
                # 超出最大范围
                continue
            else:
                start_index=index_info[str(pic)]+1
        elif ignore_number!=0:
            if ignore_number>=(end_index-1):
                continue
            else:
                start_index=ignore_number+1
        else:
            start_index=0
        for m in range(start_index,end_index):
            pic_item=data_all_list[m]
        # for pic_item in data_all_list:
            # if current_ignore<ignore_number:
            #     current_ignore=current_ignore+1
            #     continue
            # if index_info and pic_item["index"]<=index_info[str(pic)]:
            #     continue

            if train_type=="energy":
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
                temp_tensor=coordinate_transform_angle(pic_item,pic_size,use_weight,use_out1_weight=False)
            else:
                temp_tensor=[torch.zeros((pic_size,pic_size),dtype=torch.float32),torch.zeros((pic_size,pic_size),dtype=torch.float32),torch.zeros((pic_size,pic_size),dtype=torch.float32),torch.zeros((pic_size,pic_size),dtype=torch.float32)]
                center_info=[]
                for i in range(4):
                    points_list,center_x,center_y=coordinate_transform(pic_item[str(i+1)],400,pic_size,centering,weighted_average=use_weight)
                    center_info.append([center_x,center_y])
                    for item in points_list:
                        if isinstance(item[0],int) and isinstance(item[1],int):
                            temp_tensor[i][item[0]][item[1]]=temp_tensor[i][item[0]][item[1]]+item[2]
                        else:
                            temp_tensor[i][int(item[0])][int(item[1])]=temp_tensor[i][int(item[0])][int(item[1])]+item[2]
                # 绝对位置信息归一化
                center_info[0]=[(center_info[0][0]/1000+100)/100,(center_info[0][1]/1000)/100]
                center_info[1]=[(center_info[1][0]/1000)/100,(center_info[1][1]/1000)/100]
                center_info[2]=[(center_info[2][0]/1000+100)/100,(center_info[2][1]/1000+100)/100]
                center_info[3]=[(center_info[3][0]/1000)/100,(center_info[3][1]/1000+100)/100]

            # points_list,center_x,center_y=coordinate_transform(pic_item,400,pic_size,centering,weighted_average=use_weight)
            # for i in range(4):
            #     for item in points_list[str(i+1)]:
            #         temp_tensor[i][item[0]][item[1]]=temp_tensor[i][item[0]][item[1]]+item[2]
            #         mean=torch.mean(temp_tensor[i])
            #         std=torch.std(temp_tensor[i])
            #         temp_tensor[i]=(temp_tensor[i]-mean)/std
            
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
                    # pic_item["info"]["angle_center_x"]/1000,pic_item["info"]["angle_center_y"]/1000
                ],dtype=torch.float32)
            elif train_type=="angle":
                fin=temp_tensor.reshape(1,4,pic_size,pic_size,4)
                
                # fin=torch.cat((torch.cat((temp_tensor[0],temp_tensor[1]),dim=1),torch.cat((temp_tensor[2],temp_tensor[3]),dim=1)),dim=0)
                fin_label=torch.tensor([
                    math.cos(pic_item["info"]["priPhi"]*math.pi/180),math.sin(pic_item["info"]["priPhi"]*math.pi/180)
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
                    data_tensor=fin

                    # data_tensor=fin.reshape(1,4,128,128)
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
                    data_tensor=torch.cat((data_tensor,fin))
                    
                    # data_tensor=torch.cat((data_tensor,fin.reshape(1,4,128,128)))
                    label_tensor=torch.cat((label_tensor,fin_label.reshape(1,2)))
            
            current_number=current_number+1
            current_index_info[str(pic)]=pic_item["index"]
            if current_number>=total_number:
                is_break=True
                break

    if train_type=="particle" or train_type=="energy":
        label_tensor=load_label(label,current_number,label_dtype)
    
    if settings["tempData"]["autoSave"]:
        if current_index_info!={"1":0,"2":0,"3":0,"4":0}:
            if os.path.exists(settings["tempData"]["savePath"]):
                data_file_name=particle+"_"+str(energy)+"_"+str(allow_pic_number_list)+"_"+str(min_pix)+"_"+str(pic_size)+"_"+str(train_type)+"_"+str(centering)+"_"+str(use_weight)+"("+str(current_number)+")"+".data"
                data_full_path=os.path.join(settings["tempData"]["savePath"],data_file_name)

                with open(data_full_path,"wb") as f:
                    if train_type=="energy":
                        pickle.dump({"index_info":current_index_info,"data":[data_tensor[0].tolist(),data_tensor[1].tolist()],"label":label_tensor.tolist()},f)
                    else:
                        pickle.dump({"index_info":current_index_info,"data":data_tensor.tolist(),"label":label_tensor.tolist()},f)
                    f.close()
                
                # 删除旧文件
                if old_file_name:
                    os.remove(old_file_name)
                    print("old file has been deleted: "+old_file_name)
                    if log:
                        log.write("old file has been deleted: "+old_file_name)
                
                print(particle+"_"+str(energy)+" has been cached to: "+data_full_path)
                if log:
                    log.write(particle+"_"+str(energy)+" has been cached to: "+data_full_path)
            else:
                print("File cache failure (reason: invalid path "+settings["tempData"]["savePath"]+")")
                if log:
                    log.write("File cache failure (reason: invalid path "+settings["tempData"]["savePath"]+")")
        else:
            if ignore_number!=0:
                print("File cache update failure (reason: all origin data ignored)")
                if log:
                    log.write("File cache update failure (reason: all origin data ignored)")
            else:
                print("File cache update failure (reason: all origin data cached)")
                if log:
                    log.write("File cache update failure (reason: all origin data cached)")

    return data_tensor,label_tensor,current_number,min_pix

def load_label(label,length:int,dtype):
    label_list=[label for _ in range(length)]
    return torch.tensor(label_list,dtype=dtype)
