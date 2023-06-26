import os
import shutil
import time
import json
import re
# import ROOT

from utils.path import get_project_file_path
from utils.locationTransform import location_transform


def load_from_root_to_text(filePathList:list,filePathHead:str=None,fileNameHead:str="gp_nonoise_pix5pe8pmt5_CER",startNum:int=100000,endNum:int=100501):
    # 已弃用！！！
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
    # 已弃用！！！
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

def translate_data(particle:str,energy:int,filePathList:list,filePathHead:str=None,fileNameHead:str="gp_nonoise_pix5pe8pmt5_CER",startNum:int=100000,endNum:int=100501,fileLoadDelete:bool=False):
    jsonFileName=get_project_file_path("data/origin/"+particle+"_"+str(energy)+".json")
    if os.path.exists(jsonFileName):
        with open(jsonFileName,"r") as f:
            particle_dict=json.load(f)
            f.close()
        ii=len(particle_dict["c4"])
    else:
        particle_dict={
            "c1":[
            # {"1":[],"2":[],"3":[],"4":[]},{"1":[],"2":[],"3":[],"4":[]}
            ],
            "c2":[],
            "c3":[],
            "c4":[]
        }
        ii=0
    
    is_break=False
    for n in filePathList:
        if is_break:
            break
        if filePathHead!=None:
            filePath=os.path.join(filePathHead,n)
        else:
            filePath=n
        for i in range(startNum,endNum):
            if is_break:
                break
            root_file_name=os.path.join(filePath,fileNameHead+str(i)+'.root')
            if not os.path.exists(root_file_name):
                continue
            
            particle_all=[]

            root_file=ROOT.TFile(root_file_name)
            root_tree=root_file.Get('ana')
            single_number=0
            for _ in root_tree:
                single_number=single_number+1

            for m in range(single_number):
                particle_item={"1":[],"2":[],"3":[],"4":[],"info":{"_position":{},"_position_rebuild":{}}}
                for n in range(4):
                    if m==0:
                        hist=root_file.Get('h'+str(n))
                    else:
                        hist=root_file.Get('h'+str(m)+str(n))
                    x_sum=0
                    y_sum=0
                    w_sum=0

                    for a in range(1, hist.GetNbinsX() + 1):
                        for b in range(1, hist.GetNbinsY() + 1):
                            bin_content = hist.GetBinContent(a, b)
                            x_sum=x_sum+a*bin_content
                            y_sum=y_sum+b*bin_content
                            w_sum=w_sum+bin_content
                            if bin_content!=0:
                                particle_item[str(n+1)].append([a,b,bin_content])

                    if w_sum==0:
                        assert x_sum==0
                        assert y_sum==0
                        w_sum=1
                    particle_item["info"]["_position"][str(n+1)]=[x_sum/w_sum,y_sum/w_sum]
                    if n==0:
                        particle_item["info"]["_position_rebuild"]["1"]=[0.05*(x_sum/w_sum-200),0.05*(y_sum/w_sum-200)+100]
                    elif n==1:
                        particle_item["info"]["_position_rebuild"]["2"]=[0.05*(x_sum/w_sum-200)+100,0.05*(y_sum/w_sum-200)+100]
                    elif n==2:
                        particle_item["info"]["_position_rebuild"]["3"]=[0.05*(x_sum/w_sum-200),0.05*(y_sum/w_sum-200)]
                    elif n==3:
                        particle_item["info"]["_position_rebuild"]["4"]=[0.05*(x_sum/w_sum-200)+100,0.05*(y_sum/w_sum-200)]
                    
                particle_all.append(particle_item)

            for j,_ in enumerate(root_tree):
                for branch in root_tree.GetListOfBranches():
                    particle_all[j]["info"][branch.GetName()]=branch.GetLeaf(branch.GetName()).GetValue()
            

            for item in particle_all:
                pic_num=0
                for z in range(4):
                    if len(item[str(z+1)])>0:
                        pic_num=pic_num+1
                if pic_num==0:
                    continue
                if pic_num>0:
                    particle_dict["c"+str(pic_num)].append(item)
                if pic_num==4:
                    ii=ii+1

                print(ii)
            print("file "+str(i)+" finish")
            if fileLoadDelete:
                os.remove(root_file_name)
    
    with open(jsonFileName,"w") as f:
        json.dump(particle_dict,f)
        f.close()
    print(particle+str(energy)+" loading finish with total number: "+str(ii))

def add_data(particle:str,energy:int):
    jsonFileName=get_project_file_path("data/origin/"+particle+"_"+str(energy)+".json")
    if not os.path.exists(jsonFileName):
        raise Exception("file_not_found")
    with open(jsonFileName,"r") as f:
        json_data=json.load(f)
        f.close()
    for n in range(1,5):
        for i in range(len(json_data["c"+str(n)])):
            if "_position" not in json_data["c"+str(n)][i]["info"]:
                json_data["c"+str(n)][i]["info"]["_position"]={}
            else:
                continue
            if "_position_rebuild" not in json_data["c"+str(n)][i]["info"]:
                json_data["c"+str(n)][i]["info"]["_position_rebuild"]={}
            else:
                continue
            for index in range(1,5):
                x=0
                y=0
                w=0
                for item in json_data["c"+str(n)][i][str(index)]:
                    x=x+item[0]*item[2]
                    y=y+item[1]*item[2]
                    w=w+item[2]
                if w==0:
                    assert x==0 and y==0
                    w=1
                    json_data["c"+str(n)][i]["info"]["_position"][str(index)]=[0,0]
                else:
                    json_data["c"+str(n)][i]["info"]["_position"][str(index)]=[x/w,y/w]

                if index==1:
                    json_data["c"+str(n)][i]["info"]["_position_rebuild"]["1"]=[0.05*(x/w-200),0.05*(y/w-200)+100]
                elif index==2:
                    json_data["c"+str(n)][i]["info"]["_position_rebuild"]["2"]=[0.05*(x/w-200)+100,0.05*(y/w-200)+100]
                elif index==3:
                    json_data["c"+str(n)][i]["info"]["_position_rebuild"]["3"]=[0.05*(x/w-200),0.05*(y/w-200)]
                elif index==4:
                    json_data["c"+str(n)][i]["info"]["_position_rebuild"]["4"]=[0.05*(x/w-200)+100,0.05*(y/w-200)]

    with open(jsonFileName,"w") as f:
        json.dump(json_data,f)
        f.close()
