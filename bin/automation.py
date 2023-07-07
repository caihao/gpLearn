import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import math
import os
import multiprocessing
import json

from bin.dataTranslater import load_from_text_to_json
from bin.modelInit import initializeModel
from bin.dataLoader import load_data
from bin.train import train
from bin.test import test
from utils.path import get_project_file_path
from utils.log import Log
from utils.leftTime import LeftTime
from utils.dataInfo import DataInfo

def multi_process_load_data(particle:str,energy:int,particle_number:int,allow_pic_number_list:list,limit_min_pix_number:int,ignore_head_number:int,pic_size:int,centering:bool,train_type:str):
    print(particle,energy,"start...")
    if particle=="gamma":
        particle_label_number=0
    elif particle=="proton":
        particle_label_number=1
    else:
        raise Exception("invalid particle type")
    if train_type=="energy":
        data,label=load_data(particle,energy,particle_number,allow_pic_number_list,limit_min_pix_number,ignore_head_number,pic_size,centering,energy/100,torch.float32)
    elif train_type=="particle":
        data,label=load_data(particle,energy,particle_number,allow_pic_number_list,limit_min_pix_number,ignore_head_number,pic_size,centering,particle_label_number,torch.int64)
    else:
        raise Exception("invalid train type")
    return data,label,particle_label_number,energy

class GPDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class GPDataset_energy(Dataset):
    def __init__(self, data1, data2, targets):
        self.data1 = data1
        self.data2 = data2
        self.targets = targets
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx], self.targets[idx]

class Au(object):
    def __init__(self,
            gamma_energy_list:list,proton_energy_list:list,
            particle_number_gamma:int,particle_number_proton:int,
            allow_pic_number_list:list,
            limit_min_pix_number:bool=False,
            ignore_head_number:int=0,
            interval:float=0.8,
            pic_size:int=64,
            batch_size:int=1,
            use_data_type:str=None,
            centering:bool=True,
            use_weight:bool=False,
            train_type:str="particle",
            need_data_info:bool=False,
            use_loading_process:int=None,
            current_file_name:str="main.py"
    ):
        log=Log(current_file_name,"alpha-0.5.0")
        self.timeStamp=log.timeStamp
        log.info("pid",os.getpid())
        log.method("Au.__init__")

        log.info("gamma_energy_list",gamma_energy_list)
        log.info("proton_energy_list",proton_energy_list)
        log.info("particle_number_gamma",particle_number_gamma)
        log.info("particle_number_proton",particle_number_proton)
        log.info("allow_pic_number_list",allow_pic_number_list)
        log.info("limit_min_pix_number",limit_min_pix_number)
        log.info("ignore_head_number",ignore_head_number)
        log.info("interval",interval)
        log.info("pic_size",pic_size)
        log.info("batch_size",batch_size)
        self.batch_size=batch_size
        log.info("use_data_type",use_data_type)
        log.info("centering",centering)
        log.info("use_weight",use_weight)
        log.info("train_type",train_type)
        log.info("need_data_info",need_data_info)
        if need_data_info:
            self.data_info=DataInfo(self.timeStamp,train_type)
        else:
            self.data_info=None

        with open(get_project_file_path("settings.json"),"r") as f:
            self.settings=json.load(f)
            f.close()

        log.info("USE MULTIPLE GPU",self.settings["GPU"]["multiple"])
        log.info("SET CUDA DEVICE",self.settings["GPU"]["mainGPUIndex"])
        if self.settings["GPU"]["multiple"]:
            if self.settings["GPU"]["mainGPUNumber"]>=torch.cuda.device_count():
                raise Exception("GPU index out of range")
            for index in self.settings["GPU"]["multipleGPUIndex"]:
                if index >=torch.cuda.device_count():
                    raise Exception("Multiple GPU index out of range")
            log.info("MULTIPLE GPU INDEX",self.settings["GPU"]["multipleGPUIndex"])

        if use_loading_process==None:
            log.info("use_loading_process","No_Multi_Process")
        else:
            log.info("use_loading_process",use_loading_process)

        self.lt=LeftTime()
        self.lt.startLoading(len(gamma_energy_list)*particle_number_gamma+len(proton_energy_list)*particle_number_proton)

        self.train_type=train_type
        self.pic_size=pic_size
        train_data=None
        train_label=None
        self.test_list={"test_data_list":[],"test_label_list":[],"test_type_list":[],"test_energy_list":[]}
        
        for i in gamma_energy_list:
            if train_type=="particle":
                data,label,final_length=load_data("gamma" if use_data_type==None else "gamma_"+use_data_type,i,particle_number_gamma,allow_pic_number_list,limit_min_pix_number,ignore_head_number,pic_size,train_type,centering,use_weight,0,torch.int64)
            elif train_type=="energy":
                data,label,final_length=load_data("gamma" if use_data_type==None else "gamma_"+use_data_type,i,particle_number_gamma,allow_pic_number_list,limit_min_pix_number,ignore_head_number,pic_size,train_type,centering,use_weight,i/100,torch.float32)
            elif train_type=="position":
                data,label,final_length=load_data("gamma" if use_data_type==None else "gamma_"+use_data_type,i,particle_number_gamma,allow_pic_number_list,limit_min_pix_number,ignore_head_number,pic_size,train_type,centering,use_weight,None,None)
            elif train_type=="angle":
                data,label,final_length=load_data("gamma" if use_data_type==None else "gamma_"+use_data_type,i,particle_number_gamma,allow_pic_number_list,limit_min_pix_number,ignore_head_number,pic_size,train_type,centering,use_weight,None,None)
            else:
                raise Exception("invalid train type")

            if train_data==None:
                if train_type=="particle":
                    train_data=data[:int(interval*len(data))]
                elif train_type=="energy":
                    train_data=[data[0][:int(interval*len(data[0]))],data[1][:int(interval*len(data[1]))]]
                elif train_type=="position":
                    train_data=data[:int(interval*len(data))]
                    # train_data=[data[0][:int(interval*len(data[0]))],data[1][:int(interval*len(data[1]))]]
                elif train_type=="angle":
                    train_data=data[:int(interval*len(data))]
            else:
                if train_type=="particle":
                    train_data=torch.cat((train_data,data[:int(interval*len(data))]))
                elif train_type=="energy":
                    train_data=[
                        torch.cat((train_data[0],data[0][:int(interval*len(data[0]))])),
                        torch.cat((train_data[1],data[1][:int(interval*len(data[1]))]))
                    ]
                elif train_type=="position":
                    train_data=torch.cat((train_data,data[:int(interval*len(data))]))
                    # train_data=[
                    #     torch.cat((train_data[0],data[0][:int(interval*len(data[0]))])),
                    #     torch.cat((train_data[1],data[1][:int(interval*len(data[1]))]))
                    # ]
                elif train_type=="angle":
                    train_data=torch.cat((train_data,data[:int(interval*len(data))]))
            
            if train_label==None:
                train_label=label[:int(interval*len(label))]
            else:
                train_label=torch.cat((train_label,label[:int(interval*len(label))]))
            
            if train_type=="particle":
                self.test_list["test_data_list"].append(data[int(interval*len(data)):])
            elif train_type=="energy":
                self.test_list["test_data_list"].append([data[0][int(interval*len(data[0])):],data[1][int(interval*len(data[1])):]])
            elif train_type=="position":
                self.test_list["test_data_list"].append(data[int(interval*len(data)):])
                # self.test_list["test_data_list"].append([data[0][int(interval*len(data[0])):],data[1][int(interval*len(data[1])):]])
            elif train_type=="angle":
                self.test_list["test_data_list"].append(data[int(interval*len(data)):])
            self.test_list["test_label_list"].append(label[int(interval*len(data)):])
            self.test_list["test_type_list"].append(0)
            self.test_list["test_energy_list"].append(i)

            hms,da,completion=self.lt.loadLeftTime(final_length,particle_number_gamma)
            particle="gamma" if use_data_type==None else "gamma_"+use_data_type
            print(particle+" "+str(i)+" loading finish with length: "+str(final_length)+" (pre set: "+str(particle_number_gamma)+")")
            print("Loading ("+completion+"%): estimated remaining time: "+hms+", estimated completion time: "+da)
            log.write(particle+" "+str(i)+" loading finish with length: "+str(final_length)+" (pre set: "+str(particle_number_gamma)+")")
            log.write("Loading ("+completion+"%): estimated remaining time: "+hms+", estimated completion time: "+da)


        for i in proton_energy_list:
            if train_type=="particle":
                data,label,final_length=load_data("proton" if use_data_type==None else "proton_"+use_data_type,i,particle_number_proton,allow_pic_number_list,limit_min_pix_number,ignore_head_number,pic_size,train_type,centering,use_weight,1,torch.int64)
            elif train_type=="energy":
                data,label,final_length=load_data("proton" if use_data_type==None else "proton_"+use_data_type,i,particle_number_proton,allow_pic_number_list,limit_min_pix_number,ignore_head_number,pic_size,train_type,centering,use_weight,i/100,torch.float32)
            elif train_type=="position":
                data,label,final_length=load_data("proton" if use_data_type==None else "proton_"+use_data_type,i,particle_number_gamma,allow_pic_number_list,limit_min_pix_number,ignore_head_number,pic_size,train_type,centering,use_weight,None,None)
            elif train_type=="angle":
                data,label,final_length=load_data("proton" if use_data_type==None else "proton_"+use_data_type,i,particle_number_gamma,allow_pic_number_list,limit_min_pix_number,ignore_head_number,pic_size,train_type,centering,use_weight,None,None)
            
            if train_data==None:
                if train_type=="particle":
                    train_data=data[:int(interval*len(data))]
                elif train_type=="energy":
                    train_data=[data[0][:int(interval*len(data[0]))],data[1][:int(interval*len(data[1]))]]
                elif train_type=="position":
                    train_data=data[:int(interval*len(data))]
                    # train_data=data[data[0][:int(interval*len(data[0]))],data[1][:int(interval*len(data[1]))]]
                elif train_type=="angle":
                    train_data=data[:int(interval*len(data))]
            else:
                if train_type=="particle":
                    train_data=torch.cat((train_data,data[:int(interval*len(data))]))
                elif train_type=="energy":
                    train_data=[
                        torch.cat((train_data[0],data[0][:int(interval*len(data[0]))])),
                        torch.cat((train_data[1],data[1][:int(interval*len(data[1]))]))
                    ]
                elif train_type=="position":
                    train_data=torch.cat((train_data,data[:int(interval*len(data))]))
                    # train_data=[
                    #     torch.cat((train_data[0],data[0][:int(interval*len(data[0]))])),
                    #     torch.cat((train_data[1],data[1][:int(interval*len(data[1]))]))
                    # ]
                elif train_type=="angle":
                    train_data=torch.cat((train_data,data[:int(interval*len(data))]))
            
            if train_label==None:
                train_label=label[:int(interval*len(label))]
            else:
                train_label=torch.cat((train_label,label[:int(interval*len(label))]))
            
            if train_type=="particle":
                self.test_list["test_data_list"].append(data[int(interval*len(data)):])
            elif train_type=="energy":
                self.test_list["test_data_list"].append([data[0][int(interval*len(data[0])):],data[1][int(interval*len(data[1])):]])
            elif train_type=="position":
                self.test_list["test_data_list"].append(data[int(interval*len(data)):])
                # self.test_list["test_data_list"].append([data[0][int(interval*len(data[0])):],data[1][int(interval*len(data[1])):]])
            elif train_type=="angle":
                self.test_list["test_data_list"].append(data[int(interval*len(data)):])
            self.test_list["test_label_list"].append(label[int(interval*len(data)):])
            self.test_list["test_type_list"].append(1)
            self.test_list["test_energy_list"].append(i)

            hms,da,completion=self.lt.loadLeftTime(final_length,particle_number_proton)
            particle="proton" if use_data_type==None else "proton_"+use_data_type
            print(particle+" "+str(i)+" loading finish with length: "+str(final_length)+" (pre set: "+str(particle_number_proton)+")")
            print("Loading ("+completion+"%): estimated remaining time: "+hms+", estimated completion time: "+da)
            log.write(particle+" "+str(i)+" loading finish with length: "+str(final_length)+" (pre set: "+str(particle_number_proton)+")")
            log.write("Loading ("+completion+"%): estimated remaining time: "+hms+", estimated completion time: "+da)

        # 构建训练集DataLoader
        if train_type=="energy":
            gp_dataset=GPDataset_energy(train_data[0],train_data[1],train_label)
        else:
            gp_dataset=GPDataset(train_data,train_label)
        self.dataLoader=DataLoader(gp_dataset,batch_size=batch_size,shuffle=True)

        self.lt.endLoading()
        print("data loading finish")
        log.write("data loading finish")
        self.log=log

    def load_model(self,modelName:str,model_type:str,modelInit:bool=False):
        self.log.method("Au.load_model")
        self.log.info("modelName",modelName)
        self.log.info("model_type",model_type)
        self.log.info("is_modelInit",modelInit)
        if modelInit:
            self.log.write("model initializing...")
            if self.train_type=="particle":
                initializeModel(modelName,1,self.pic_size*2,self.pic_size*2,2,model_type,["acc","q"])
                # initializeModel(modelName,4,self.pic_size,self.pic_size,2,model_type,["acc","q"])
            elif self.train_type=="energy":
                initializeModel(modelName,4,self.pic_size,self.pic_size,1,model_type,["l"],input_info=4)
                # initializeModel(modelName,1,self.pic_size,self.pic_size,1,model_type,["l"])
            elif self.train_type=="position":
                initializeModel(modelName,1,self.pic_size*2,self.pic_size*2,2,model_type,["l","l_0","l_1"],input_info=2)
            elif self.train_type=="angle":
                initializeModel(modelName,4,self.pic_size,self.pic_size,2,model_type,["l"],input_info=2)
            else:
                self.log.error("invalid train type")
                raise Exception("invalid train type")
            self.log.write("model initializing finish")
        self.log.write("model loading...")
        
        self.modelfile=get_project_file_path("data/model/"+modelName)
        state=torch.load(self.modelfile)

        # self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model=state['model'].to(self.device)
        # self.optimizer=torch.optim.AdamW(self.model.parameters(),8e-6)
        # self.model=nn.DataParallel(self.model)

        if self.settings["GPU"]["multiple"]:
            self.device=torch.device("cuda:"+str(self.settings["GPU"]["mainGPUIndex"]))
            self.model=nn.DataParallel(state['model'].to(self.device),device_ids=self.settings["GPU"]["multipleGPUIndex"],output_device=self.settings["GPU"]["mainGPUIndex"])
        else:
            self.device=torch.device("cuda:"+str(self.settings["GPU"]["mainGPUIndex"]) if torch.cuda.is_available() else 'cpu')
            self.model=state['model'].to(self.device)

        if self.train_type=="particle":
            self.acc=state['acc']
            self.q=state['q']
            print("model acc:"+str(self.acc)+", q:"+str(self.q))
            self.log.write("model acc:"+str(self.acc)+", q:"+str(self.q))
            self.loss_function=nn.CrossEntropyLoss(reduction="sum")
            self.log.info("loss_function","nn.CrossEntropyLoss()")
        elif self.train_type=="energy":
            self.l=state['l']
            print("model l:"+str(self.l))
            self.log.write("model l:"+str(self.l))
            # class EnergyLoss(nn.Module):
            #     def __init__(self):
            #         super(EnergyLoss,self).__init__()

            #     def forward(self,inputs,targets):
            #         loss=torch.mean((inputs-targets)**2)/targets**2
            #         return loss

            self.loss_function=nn.MSELoss(reduction="sum")
            # self.loss_function=EnergyLoss()
            self.log.info("loss_function","nn.MSELoss(reduction=sum)")
        elif self.train_type=="position":
            self.l=state['l']
            self.l_0=state['l_0']
            self.l_1=state['l_1']
            print("model l:"+str(self.l)+", l_0:"+str(self.l_0)+", l_1:"+str(self.l_1))
            self.log.write("model l:"+str(self.l)+", l_0:"+str(self.l_0)+", l_1:"+str(self.l_1))
            self.loss_function=nn.MSELoss(reduction="sum")
            self.log.info("loss_function","nn.MSELoss(reduction=sum)")
        elif self.train_type=="angle":
            self.l=state['l']
            print("model l:"+str(self.l))
            self.log.write("model l:"+str(self.l))
            self.loss_function=nn.MSELoss(reduction="sum")
            self.log.info("loss_function","nn.MSELoss(reduction=sum)")
        self.log.write("model loading finish")
        self.set_optimizer()
    
    def set_optimizer(self,lr:float=6e-6):
        self.log.method("Au.set_optimizer")
        self.log.info("optimizer_lr",lr)
        self.log.write("optimizer setting...")
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr)
        # self.model=nn.DataParallel(self.model)
        self.log.write("optimizer setting finish")

    def train_step(self,epoch_step_list:list,lr_step_list:list=None):
        self.log.method("Au.train_step")
        self.log.info("epoch_step_list",epoch_step_list)
        if lr_step_list!=None:
            self.log.info("lr_step_list",lr_step_list)
        else:
            self.log.info("lr_step_list","default")

        # self.log.info("need_data_info",need_data_info)
        # if need_data_info:
        #     data_info=[]

        self.lt.startTraining(len(self.dataLoader)*self.batch_size,sum(epoch_step_list))
        for i in range(len(epoch_step_list)):
            if lr_step_list!=None:
                self.set_optimizer(lr_step_list[i])
                se_lr=lr_step_list[i]
            else:
                se_lr=6e-6
            
            self.log.write("training on step:"+str(i+1)+" with lr:"+str(se_lr)+" ...")
            print("training on step:"+str(i+1)+" with lr:"+str(se_lr)+" ...")
            if self.train_type=="particle":
                self.model,return_result=train(self.dataLoader,self.batch_size,epoch_step_list[i],self.model,self.device,self.optimizer,self.loss_function,self.train_type,self.test_list,{"acc":self.acc,"q":self.q,"model_file":self.modelfile},self.log,self.data_info,self.lt)
                self.acc=return_result["acc"]
                self.q=return_result["q"]
            elif self.train_type=="energy":
                self.model,return_result=train(self.dataLoader,self.batch_size,epoch_step_list[i],self.model,self.device,self.optimizer,self.loss_function,self.train_type,self.test_list,{"l":self.l,"model_file":self.modelfile},self.log,self.data_info,self.lt)
                self.l=return_result["l"]
            elif self.train_type=="position":
                self.model,return_result=train(self.dataLoader,self.batch_size,epoch_step_list[i],self.model,self.device,self.optimizer,self.loss_function,self.train_type,self.test_list,{"l":self.l,"l_0":self.l_0,"l_1":self.l_1,"model_file":self.modelfile},self.log,self.data_info,self.lt)
                self.l=return_result["l"]
                self.l_0=return_result["l_0"]
                self.l_1=return_result["l_1"]
            elif self.train_type=="angle":
                self.model,return_result=train(self.dataLoader,self.batch_size,epoch_step_list[i],self.model,self.device,self.optimizer,self.loss_function,self.train_type,self.test_list,{"l":self.l,"model_file":self.modelfile},self.log,self.data_info,self.lt)
                self.l=return_result["l"]
            
            self.log.write("step "+str(i+1)+" finish")
            print("step "+str(i+1)+" finish")
            # if need_data_info:
            #     data_info.append(data_info_item)

        hms,da=self.lt.endTraining()
        print("Training finish! Used time: "+hms+", end time: "+da)
        self.log.write("Training finish! Used time: "+hms+", end time: "+da)
       
        # if need_data_info:
        #     with open(get_project_file_path("data/info/"+self.timeStamp+".data"),"wb") as f:
        #         pickle.dump(data_info,f)
        #         f.close()

    def test(self):
        self.log.method("test")
        if self.train_type=="particle":
            gamma_eff,gamma_total,proton_eff,proton_total=test(self.test_list["test_data_list"],self.test_list["test_label_list"],self.test_list["test_type_list"],self.test_list["test_energy_list"],self.model,self.device,self.train_type,True,self.log,None,self.data_info)
            acc_g=gamma_eff/gamma_total
            acc_p=proton_eff/proton_total
            acc=(gamma_eff+proton_eff)/(gamma_total+proton_total)
            try:
                q=acc_g/math.sqrt(1-acc_p)
            except:
                q=acc_g/math.sqrt(1-acc_p+0.00001)
            self.log.write("test result with acc_g:"+str(acc_g)+", acc_p:"+str(acc_p)+", acc:"+str(acc)+", q:"+str(q))
            print("test result with acc_g:"+str(acc_g)+", acc_p:"+str(acc_p)+", acc:"+str(acc)+", q:"+str(q))
        elif self.train_type=="energy":
            gamma_eff,gamma_total,proton_eff,proton_total=test(self.test_list["test_data_list"],self.test_list["test_label_list"],self.test_list["test_type_list"],self.test_list["test_energy_list"],self.model,self.device,self.train_type,True,self.log,None,self.data_info)
            l_g=gamma_eff/(gamma_total+0.0001)
            l_p=proton_eff/(proton_total+0.0001)
            l=(gamma_eff+proton_eff)/(gamma_total+proton_total+0.0001)
            self.log.write("test with loss_g:"+str(l_g)+", loss_p:"+str(l_p)+", loss:"+str(l))
            print("test with loss_g:"+str(l_g)+", loss_p:"+str(l_p)+", loss:"+str(l))
        elif self.train_type=="position":
            gamma_eff,gamma_loss_0,gamma_loss_1,gamma_total,proton_eff,proton_loss_0,proton_loss_1,proton_total=test(self.test_list["test_data_list"],self.test_list["test_label_list"],self.test_list["test_type_list"],self.test_list["test_energy_list"],self.model,self.device,self.train_type,True,self.log,None,self.data_info)
            l_g=gamma_eff/(gamma_total+0.0001)
            l_g_0=gamma_loss_0/(gamma_total+0.0001)
            l_g_1=gamma_loss_1/(gamma_total+0.0001)
            l_p=proton_eff/(proton_total+0.0001)
            l_p_0=proton_loss_0/(proton_total+0.0001)
            l_p_1=proton_loss_1/(proton_total+0.0001)
            l=(gamma_eff+proton_eff)/(gamma_total+proton_total+0.0001)
            l_0=(gamma_loss_0+proton_loss_0)/(gamma_total+proton_total+0.0001)
            l_1=(gamma_loss_1+proton_loss_1)/(gamma_total+proton_total+0.0001)
            res_log="test with l_g:%f, l_g_0:%f, l_g_1:%f, l_p:%f, l_p_0:%f, l_p_1:%f, l:%f, l_0:%f, l_1:%f" % (l_g,l_g_0,l_g_1,l_p,l_p_0,l_p_1,l,l_0,l_1)
            res_print="test with l_g:%f, l_g_0:%f, l_g_1:%f,\nl_p:%f, l_p_0:%f, l_p_1:%f,\nl:%f, l_0:%f, l_1:%f" % (l_g,l_g_0,l_g_1,l_p,l_p_0,l_p_1,l,l_0,l_1)
            self.log.write(res_log)
            print(res_print)
        elif self.train_type=="angle":
            gamma_eff,gamma_total,proton_eff,proton_total=test(self.test_list["test_data_list"],self.test_list["test_label_list"],self.test_list["test_type_list"],self.test_list["test_energy_list"],self.model,self.device,self.train_type,True,self.log,None,self.data_info)
            l_g=gamma_eff/(gamma_total+0.0001)
            l_p=proton_eff/(proton_total+0.0001)
            l=(gamma_eff+proton_eff)/(gamma_total+proton_total+0.0001)
            self.log.write("test with loss_g:"+str(l_g)+", loss_p:"+str(l_p)+", loss:"+str(l))
            print("test with loss_g:"+str(l_g)+", loss_p:"+str(l_p)+", loss:"+str(l))
    
    def finish(self):
        if self.data_info:
            info_path=self.data_info.save()
            self.log.write("data info has been saved in '"+info_path+"'")
            print("data info has been saved in '"+info_path+"'")
        
        self.log.method("finish")
        self.log.write("job finish")
        print("job finish")
        self.log.close()
