import time
import pickle

from utils.path import get_project_file_path

class DataInfo(object):
    def __init__(self,init_time:int,train_type:str):
        self.init_time=init_time
        self.train_type=train_type
        self.train_temp=[]
        self.test_temp={}
        if train_type=="particle":
            self.info={
                "train":[],
                "test":[],
                "result":{
                    "gamma":{"energy":[],"r":[]},
                    "proton":{"energy":[],"r":[]},
                    "q":{"energy":[],"r":[]}
                }
            }
        elif train_type=="energy":
            self.info={
                "train":[],
                "test":[],
                "result":{
                    "gamma":{"energy":[],"loss":[]},
                    "proton":{"energy":[],"loss":[]}
                }
            }
        elif train_type=="position":
            self.info={
                "train":[],
                "test":[],
                "result":{
                    "gamma":{"energy":[],"loss":[],"loss_0":[],"loss_1":[]},
                    "proton":{"energy":[],"loss":[],"loss_0":[],"loss_1":[]}
                }
            }
        elif train_type=="angle":
            self.info={
                "train":[],
                "test":[],
                "result":{
                    "gamma":{"energy":[],"loss":[]},
                    "proton":{"energy":[],"loss":[]}
                }
            }
        else:
            raise Exception("invalid train type")
        
    def add_train_info(self,info_dict:dict):
        self.train_temp.append(info_dict.update({"time":int(time.time())}))
    
    def add_test_info(self,key:str,info_dict:dict):
        self.test_temp[key]=info_dict.update({"time":int(time.time())})
        # self.test_temp.append(info_dict.update({"time":int(time.time())}))

    def finish_train_info(self):
        self.info["train"].append(self.train_temp)
        self.train_temp.clear()
    
    def finish_test_info(self):
        self.info["test"].append(self.test_temp)
        self.test_temp.clear()

    def add_result_particle(self,energy:int,r:float,info_type:str):
        if info_type in ["gamma","proton","q"]:
            self.info["result"][info_type]["energy"].append(energy)
            self.info["result"][info_type]["r"].append(r)
        else:
            raise Exception("invalid info type")
        
    def add_result_energy(self,energy:int,loss:float,info_type:str):
        if info_type in ["gamma","proton"]:
            self.info["result"][info_type]["energy"].append(energy)
            self.info["result"][info_type]["loss"].append(loss)
        else:
            raise Exception("invalid info type")
    
    def add_result_position(self,energy:int,loss:list,info_type:str):
        if info_type in ["gamma","proton"]:
            self.info["result"][info_type]["energy"].append(energy)
            self.info["result"][info_type]["loss"].append(loss[0])
            self.info["result"][info_type]["loss_0"].append(loss[1])
            self.info["result"][info_type]["loss_1"].append(loss[2])
        else:
            raise Exception("invalid info type")
    
    def add_result_angle(self,energy:int,loss:float,info_type:str):
        if info_type in ["gamma","proton"]:
            self.info["result"][info_type]["energy"].append(energy)
            self.info["result"][info_type]["loss"].append(loss)
        else:
            raise Exception("invalid info type")
    
    def save(self):
        with open(get_project_file_path("data/info/"+self.init_time+".data"),"wb") as f:
            pickle.dump(self.info,f)
            f.close()
        return "data/info/"+self.init_time+".data"