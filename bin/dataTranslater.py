import os
import shutil
import time
import json
import re

from utils.path import get_project_file_path
from utils.locationTransform import location_transform

# particle="gamma"
# energy=100

def load_from_root_to_text(filePathList:list,filePathHead:str=None,fileNameHead:str="gp_nonoise_pix5pe8pmt5_CER",startNum:int=100000,endNum:int=100501):

    # filePathHead='/home/cay/rootdata/codeproject/data_origin'
    # filePathList=['/1500/CR_20deg_700m_1500GeV/']
    # fileNameHead='gp_nonoise_pix5pe8pmt5_CER'
    # startNum=100000
    # endNum=100500

    tempPath=os.path.join("data/temp/",str(int(time.time())))
    os.makedirs(get_project_file_path(tempPath))

    currentNumber=0
    for n in filePathList:
        if filePathHead!=None:
            filePath=os.path.join(filePathHead,n)
            for i in range(startNum,endNum):
                try:
                    os.rename(os.path.join(filePath,fileNameHead+str(i)+'.data'),os.path.join(tempPath,str(currentNumber)+".txt"))
                    currentNumber=currentNumber+1
                except FileNotFoundError:
                    continue
    
    return tempPath,currentNumber


def load_from_text_to_json(particle:str,energy:int,tempPath:str,text_file_save:bool=False):
    jsonFileName=get_project_file_path("data/origin/"+particle+"_"+str(energy)+".json")
    if os.path.exists(jsonFileName):
        with open(jsonFileName,"r") as f:
            particle_dict=json.load(f)
            f.close()
    else:
        particle_dict={
            "c1":[
            # {"1":[],"2":[],"3":[],"4":[]},{"1":[],"2":[],"3":[],"4":[]}
            ],
            "c2":[],
            "c3":[],
            "c4":[]
        }
    i=0
    while True:
        now_number=0
        x_list=[[],[],[],[]]
        y_list=[[],[],[],[]]
        value_list=[[],[],[],[]]
        filePath=get_project_file_path(os.path.join(tempPath,str(i)+".txt"))
        if not os.path.exists(filePath):
            break
        with open(filePath,'r') as f:
        # with open(get_project_file_path("data/temp/"+str(i)+".txt"),'r') as f:
            for line in f:
                items=re.search(r'(.*) (\d) (.*?) (.*)',line)
                num=int(items.group(1))
                picture=int(items.group(2))
                location=int(items.group(3))
                value=float(items.group(4))
                x,y=location_transform(location)
                if now_number==num:
                    x_list[picture].append(x)
                    y_list[picture].append(y)
                    value_list[picture].append(value)
                else:
                    now_number=num
                    # 判断激发探测器数量同时添加数据
                    activate_pic=0
                    particle_item={"1":[],"2":[],"3":[],"4":[]}
                    for m in range(4):
                        assert len(x_list[m])==len(y_list[m]) and len(y_list[m])==len(value_list[m])
                        if len(x_list[m])>0:
                            activate_pic=activate_pic+1
                            for n in range(len(x_list[m])):
                                particle_item[str(m+1)].append((x_list[m][n],y_list[m][n],value_list[m][n]))
                    if activate_pic==0:
                        continue
                    particle_dict["c"+str(activate_pic)].append(particle_item)
                    x_list=[[],[],[],[]]
                    y_list=[[],[],[],[]]
                    value_list=[[],[],[],[]]
            f.close()
        i=i+1
        
    with open(jsonFileName,"w") as f:
        json.dump(particle_dict,f)
        f.close()
    # os.removedirs(get_project_file_path(tempPath))
    if not text_file_save:
        shutil.rmtree(get_project_file_path(tempPath))
