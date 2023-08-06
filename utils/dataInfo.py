import time
import pickle

from utils.path import get_project_file_path

class DataInfo(object):
    def __init__(self,init_time:int,train_type:str,train_info_size:int=64):
        self.init_time=init_time
        self.train_type=train_type
        self.train_batch=[]
        self.train_info_size=train_info_size
        self.train_temp=[]
        self.test_temp={}
        if train_type=="particle":
            self.key_list=["loss","acc"]
            self.info={
                "info":{},
                "train":[],
                "test":[],
                "result":{
                    "gamma":{"energy":[],"r":[],"std":[]},
                    "proton":{"energy":[],"r":[],"std":[]},
                    "q":{"energy":[],"r":[],"std":[]}
                }
            }
        elif train_type=="energy":
            self.key_list=["loss"]
            self.info={
                "info":{},
                "train":[],
                "test":[],
                "result":{
                    "gamma":{"energy":[],"loss":[],"std":[]},
                    "proton":{"energy":[],"loss":[],"std":[]}
                }
            }
        elif train_type=="position":
            self.key_list=["loss","loss_0","loss_1"]
            self.info={
                "info":{},
                "train":[],
                "test":[],
                "result":{
                    "gamma":{"energy":[],"loss":[],"loss_0":[],"loss_1":[],"std":[],"std_0":[],"std_1":[],"angle":[],"std_angle":[]},
                    "proton":{"energy":[],"loss":[],"loss_0":[],"loss_1":[],"std":[],"std_0":[],"std_1":[],"angle":[],"std_angle":[]}
                }
            }
        elif train_type=="angle":
            self.key_list=["loss"]
            self.info={
                "info":{},
                "train":[],
                "test":[],
                "result":{
                    "gamma":{"energy":[],"loss":[],"std":[],"angle":[],"std_angle":[]},
                    "proton":{"energy":[],"loss":[],"std":[],"angle":[],"std_angle":[]}
                }
            }
        else:
            raise Exception("invalid train type")

    def set_info(self,key:str,value):
        self.info["info"][key]=value

    def add_train_info(self,info_dict:dict):
        info_dict.update({"time":int(time.time())})
        self.train_batch.append(info_dict)
        if len(self.train_batch)>=self.train_info_size:
            self.add_batch()
    
    def add_batch(self):
        t={"batchSize":0,"time":0}
        for key in self.key_list:
            t[key]=0
        for item in self.train_batch:
            t["batchSize"]=t["batchSize"]+1
            for key in self.key_list:
                t[key]=t[key]+item[key]
        for key in self.key_list:
            t[key]=t[key]/t["batchSize"]
        t["time"]=int(time.time())
        self.train_temp.append(t)
        self.train_batch.clear()
    
    def add_test_info(self,key:str,info_dict:dict):
        info_dict.update({"time":int(time.time())})
        self.test_temp[key]=info_dict
        # self.test_temp.append(info_dict.update({"time":int(time.time())}))

    def finish_train_info(self):
        if len(self.train_batch)>0:
            self.add_batch()
        self.info["train"].append(self.train_temp)
        self.train_temp.clear()
    
    def finish_test_info(self):
        self.info["test"].append(self.test_temp)
        self.test_temp.clear()

    def add_result_particle(self,energy:int,r:float,std:float,info_type:str):
        if info_type in ["gamma","proton","q"]:
            self.info["result"][info_type]["energy"].append(energy)
            self.info["result"][info_type]["r"].append(r)
            self.info["result"][info_type]["std"].append(std)
        else:
            raise Exception("invalid info type")
        
    def add_result_energy(self,energy:int,loss:float,std:float,info_type:str):
        if info_type in ["gamma","proton"]:
            self.info["result"][info_type]["energy"].append(energy)
            self.info["result"][info_type]["loss"].append(loss)
            self.info["result"][info_type]["std"].append(std)
        else:
            raise Exception("invalid info type")
    
    def add_result_position(self,energy:int,loss:list,std:list,info_type:str):
        if info_type in ["gamma","proton"]:
            self.info["result"][info_type]["energy"].append(energy)
            self.info["result"][info_type]["loss"].append(loss[0])
            self.info["result"][info_type]["loss_0"].append(loss[1])
            self.info["result"][info_type]["loss_1"].append(loss[2])
            self.info["result"][info_type]["angle"].append(loss[3])
            self.info["result"][info_type]["std"].append(std[0])
            self.info["result"][info_type]["std_0"].append(std[1])
            self.info["result"][info_type]["std_1"].append(std[2])
            self.info["result"][info_type]["std_angle"].append(std[3])
        else:
            raise Exception("invalid info type")
    
    def add_result_angle(self,energy:int,loss:float,std:float,angle:float,std_angle:float,info_type:str):
        if info_type in ["gamma","proton"]:
            self.info["result"][info_type]["energy"].append(energy)
            self.info["result"][info_type]["loss"].append(loss)
            self.info["result"][info_type]["std"].append(std)
            self.info["result"][info_type]["angle"].append(angle)
            self.info["result"][info_type]["std_angle"].append(std_angle)
        else:
            raise Exception("invalid info type")
    
    def save(self):
        with open(get_project_file_path("data/info/"+self.init_time+".data"),"wb") as f:
            pickle.dump(self.info,f)
            f.close()
        return "data/info/"+self.init_time+".data"
