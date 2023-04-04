import torch
import torch.nn as nn
import math
import os
import pickle
import multiprocessing

from bin.dataTranslater import load_from_text_to_json
from bin.modelInit import initializeModel
from bin.dataLoader import load_data
from bin.train import train
from bin.test import test
from utils.path import get_project_file_path
from utils.log import Log

def multi_process_load_data(particle:str,energy:int,particle_number:int,allow_pic_number_list:list,allow_min_pix_number:int,ignore_head_number:int,pic_size:int,centering:bool,train_type:str):
    print(particle,energy,"start...")
    if particle=="gamma":
        particle_label_number=0
    elif particle=="proton":
        particle_label_number=1
    else:
        raise Exception("invalid particle type")
    if train_type=="energy":
        data,label=load_data(particle,energy,particle_number,allow_pic_number_list,allow_min_pix_number,ignore_head_number,pic_size,centering,energy/100,torch.float32)
    elif train_type=="particle":
        data,label=load_data(particle,energy,particle_number,allow_pic_number_list,allow_min_pix_number,ignore_head_number,pic_size,centering,particle_label_number,torch.int64)
    else:
        raise Exception("invalid train type")
    return data,label,particle_label_number,energy

class Au(object):
    def __init__(
            self,gamma_energy_list:list,proton_energy_list:list,
            particle_number_gamma:int,particle_number_proton:int,
            allow_pic_number_list:list,
            allow_min_pix_number:int,
            ignore_head_number:int=0,
            interval:float=0.8,
            pic_size:int=64,
            use_data_type:str=None,
            centering:bool=True,
            use_weight:bool=False,
            train_type:str="particle",
            use_loading_process:int=4,
            use_gpu_number:int=None,
            current_file_name:str="main.py"
    ):
        log=Log(current_file_name)
        self.timeStamp=log.timeStamp
        log.info("pid",os.getpid())
        log.method("Au.__init__")

        log.info("gamma_energy_list",gamma_energy_list)
        log.info("proton_energy_list",proton_energy_list)
        log.info("particle_number_gamma",particle_number_gamma)
        log.info("particle_number_proton",particle_number_proton)
        log.info("allow_pic_number_list",allow_pic_number_list)
        log.info("allow_min_pix_number",allow_min_pix_number)
        log.info("ignore_head_number",ignore_head_number)
        log.info("interval",interval)
        log.info("pic_size",pic_size)
        log.info("centering",centering)
        log.info("train_type",train_type)

        if use_gpu_number!=None:
            os.environ["CUDA_VISIBLE_DEVICES"]=str(use_gpu_number)
            log.info("SET CUDA DEVICE",use_gpu_number)
        else:
            log.info("SET CUDA DEVICE","default")
        log.info("use_loading_process",use_loading_process)

        self.train_type=train_type
        self.pic_size=pic_size
        self.train_data=None
        self.train_label=None
        self.test_list={"test_data_list":[],"test_label_list":[],"test_type_list":[],"test_energy_list":[]}
        
        # log.write("using multi-process")
        # pool=multiprocessing.Pool(use_loading_process)
        # rets=[]
        # for i in gamma_energy_list:
        #     p=pool.apply_async(multi_process_load_data,args=("gamma",i,particle_number_gamma,allow_pic_number_list,allow_min_pix_number,ignore_head_number,pic_size,centering,train_type))
        #     rets.append(p)
        # for i in proton_energy_list:
        #     p=pool.apply_async(multi_process_load_data,args=("proton",i,particle_number_proton,allow_pic_number_list,allow_min_pix_number,ignore_head_number,pic_size,centering,train_type))
        #     rets.append(p)
        # pool.close()
        # pool.join()
        
        # log.write("data loading...")
        # for p in rets:
        #     data,label,particle_label_num,ene=p.get()
        #     if particle_label_num==0:
        #         print("gamma",ene,"loading finish")
        #         log.write("gamma"+str(ene)+" loading finish")
        #     elif particle_label_num==1:
        #         print("proton",ene,"loading finish")
        #         log.write("proton"+str(ene)+" loading finish")
        #     else:
        #         log.error("invalid particle label number")
        #         raise Exception("invalid particle label number")

        #     if self.train_data==None:
        #         self.train_data=data[:int(interval*len(data))]
        #     else:
        #         self.train_data=torch.cat((self.train_data,data[:int(interval*len(data))]))
        #     if self.train_label==None:
        #         self.train_label=label[:int(interval*len(label))]
        #     else:
        #         self.train_label=torch.cat((self.train_label,label[:int(interval*len(label))]))
            
        #     self.test_list["test_data_list"].append(data[int(interval*len(data)):])
        #     self.test_list["test_label_list"].append(label[int(interval*len(data)):])
        #     self.test_list["test_type_list"].append(particle_label_num)
        #     self.test_list["test_energy_list"].append(ene)
        
        
        for i in gamma_energy_list:
            print(i)
            if train_type=="energy":
                data,label=load_data("gamma" if use_data_type==None else "gamma_"+use_data_type,i,particle_number_gamma,allow_pic_number_list,allow_min_pix_number,ignore_head_number,pic_size,"overlay",centering,use_weight,i/100,torch.float32,log)
            elif train_type=="particle":
                data,label=load_data("gamma" if use_data_type==None else "gamma_"+use_data_type,i,particle_number_gamma,allow_pic_number_list,allow_min_pix_number,ignore_head_number,pic_size,"joint",centering,use_weight,0,torch.int64,log)
                # data,label=load_data("gamma" if use_data_type==None else "gamma_"+use_data_type,i,particle_number_gamma,allow_pic_number_list,allow_min_pix_number,ignore_head_number,pic_size,"overlay",centering,use_weight,0,torch.int64,log)
            else:
                raise Exception("invalid train_type")
            
            if self.train_data==None:
                self.train_data=data[:int(interval*len(data))]
            else:
                self.train_data=torch.cat((self.train_data,data[:int(interval*len(data))]))
            if self.train_label==None:
                self.train_label=label[:int(interval*len(label))]
            else:
                self.train_label=torch.cat((self.train_label,label[:int(interval*len(label))]))
            self.test_list["test_data_list"].append(data[int(interval*len(data)):])
            self.test_list["test_label_list"].append(label[int(interval*len(data)):])
            self.test_list["test_type_list"].append(0)
            self.test_list["test_energy_list"].append(i)
    
        for i in proton_energy_list:
            print(i)
            if train_type=="energy":
                data,label=load_data("proton" if use_data_type==None else "proton_"+use_data_type,i,particle_number_proton,allow_pic_number_list,allow_min_pix_number,ignore_head_number,pic_size,"overlay",centering,use_weight,i/100,torch.float32,log)
            elif train_type=="particle":
                data,label=load_data("proton" if use_data_type==None else "proton_"+use_data_type,i,particle_number_proton,allow_pic_number_list,allow_min_pix_number,ignore_head_number,pic_size,"joint",centering,use_weight,1,torch.int64,log)
                # data,label=load_data("proton" if use_data_type==None else "proton_"+use_data_type,i,particle_number_proton,allow_pic_number_list,allow_min_pix_number,ignore_head_number,pic_size,"overlay",centering,use_weight,1,torch.int64,log)
            else:
                raise Exception("invalid train_type")
            if self.train_data==None:
                self.train_data=data[:int(interval*len(data))]
            else:
                self.train_data=torch.cat((self.train_data,data[:int(interval*len(data))]))
            if self.train_label==None:
                self.train_label=label[:int(interval*len(label))]
            else:
                self.train_label=torch.cat((self.train_label,label[:int(interval*len(label))]))
            self.test_list["test_data_list"].append(data[int(interval*len(data)):])
            self.test_list["test_label_list"].append(label[int(interval*len(data)):])
            self.test_list["test_type_list"].append(1)
            self.test_list["test_energy_list"].append(i)

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
                initializeModel(modelName,4,self.pic_size,self.pic_size,1,model_type,["l"])
            else:
                self.log.error("invalid train type")
                raise Exception("invalid train type")
            self.log.write("model initializing finish")
        self.log.write("model loading...")
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.modelfile=get_project_file_path("data/model/"+modelName)
        state=torch.load(self.modelfile)
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
            self.loss_function=nn.MSELoss(reduction="sum")
            self.log.info("loss_function","nn.MSELoss()")
        self.log.write("model loading finish")
        self.set_optimizer()
    
    def set_optimizer(self,lr:float=6e-6):
        self.log.method("Au.set_optimizer")
        self.log.info("optimizer_lr",lr)
        self.log.write("optimizer setting...")
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr)
        self.log.write("optimizer setting finish")

    def train_step(self,epoch_step_list:list,lr_step_list:list=None,batch_size:int=1,need_data_info:bool=False):
        self.log.method("Au.train_step")
        self.log.info("epoch_step_list",epoch_step_list)
        if lr_step_list!=None:
            assert len(epoch_step_list)==len(lr_step_list)
            self.log.info("lr_step_list",lr_step_list)
        else:
            self.log.info("lr_step_list","default")
        self.log.info("batch_size",batch_size)
        # self.log.info("auto_save",auto_save)

        if need_data_info:
            data_info=[]

        for i in range(len(epoch_step_list)):
            if lr_step_list!=None:
                self.set_optimizer(lr_step_list[i])
                se_lr=lr_step_list[i]
            else:
                se_lr=6e-6
            
            self.log.write("training on step:"+str(i+1)+" with lr:"+str(se_lr)+" ...")
            print("training on step:"+str(i+1)+" with lr:"+str(se_lr)+" ...")
            if self.train_type=="particle":
                self.model,return_result,data_info_item=train(self.train_data,self.train_label,batch_size,epoch_step_list[i],self.model,self.device,self.optimizer,self.loss_function,True,self.test_list,{"acc":self.acc,"q":self.q,"model_file":self.modelfile},self.log,need_data_info)
                self.acc=return_result["acc"]
                self.q=return_result["q"]
                # acc_g,acc_p=test(self.test_list["test_data_list"],self.test_list["test_label_list"],self.test_list["test_type_list"],self.test_list["test_energy_list"],self.model,self.device,True,self.log)
                # acc=(acc_g+acc_p)/2
                # try:
                #     q=acc_g/math.sqrt(1-acc_p)
                # except:
                #     q=acc_g/math.sqrt(1-acc_p+0.00001)
                # res="acc_g:%f, acc_p:%f, acc:%f, q:%f" % (acc_g,acc_p,acc,q)
                self.log.write("step "+str(i+1)+" finish")
                # self.log.write(res)
                print("step "+str(i+1)+" finish")
                # print(res)
            elif self.train_type=="energy":
                self.model,return_result,data_info_item=train(self.train_data,self.train_label,batch_size,epoch_step_list[i],self.model,self.device,self.optimizer,self.loss_function,False,self.test_list,{"l":self.l,"model_file":self.modelfile},self.log,need_data_info)
                self.l=return_result["l"]
                # l_g,l_p=test(self.test_list["test_data_list"],self.test_list["test_label_list"],self.test_list["test_type_list"],self.test_list["test_energy_list"],self.model,self.device,False,self.log)
                # l=(l_g+l_p)/2
                # res="l_g:%f, l_p:%f, l:%f" % (l_g,l_p,l)
                self.log.write("step "+str(i+1)+" finish")
                # self.log.write(res)
                print("step "+str(i+1)+" finish")
                # print(res)
            
            if need_data_info:
                data_info.append(data_info_item)

            # if self.train_type=="particle":
            #     print("test result with acc_g:"+str(acc_g)+", acc_p:"+str(acc_p)+", acc:"+str(acc)+", q:"+str(q))
            #     if acc>=self.acc and q>=self.q:
            #         torch.save({'model':self.model,'acc':acc,'q':q},self.modelfile)
            #         self.acc=acc
            #         self.q=q
            #         self.log.write("model update")
            #         print("model update")
            #     else:
            #         self.log.write("model continue...")
            #         print("model continue...")
            # elif self.train_type=="energy":
            #     print("test with loss:"+str(l))
            #     if l<=self.l or self.l==0:
            #         torch.save({'model':self.model,'l':l},self.modelfile)
            #         self.l=l
            #         self.log.write("model update")
            #         print("model update")
            #     else:
            #         self.log.write("model continue...")
            #         print("model continue...")
            # if auto_save:
            #     
            # else:
            #     self.log.write("auto_save switch is off")
            #     print("auto_save switch is off")

        if need_data_info:
            with open(get_project_file_path("data/info/"+self.timeStamp+".data"),"wb") as f:
                pickle.dump(data_info,f)
                f.close()
            self.log.write("data info has been saved in '"+"data/info/"+self.timeStamp+".data"+"'")
            print("data info has been saved in '"+"data/info/"+self.timeStamp+".data"+"'")

    def test(self):
        self.log.method("test")
        if self.train_type=="particle":
            gamma_eff,gamma_total,proton_eff,proton_total=test(self.test_list["test_data_list"],self.test_list["test_label_list"],self.test_list["test_type_list"],self.test_list["test_energy_list"],self.model,self.device,True,self.log)
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
            gamma_eff,gamma_total,proton_eff,proton_total=test(self.test_list["test_data_list"],self.test_list["test_label_list"],self.test_list["test_type_list"],self.test_list["test_energy_list"],self.model,self.device,False,self.log)
            l_g=gamma_eff/(gamma_total+0.0001)
            l_p=proton_eff/(proton_total+0.0001)
            l=(gamma_eff+proton_eff)/(gamma_total+proton_total)
            self.log.write("test with loss_g:"+str(l_g)+", loss_p:"+str(l_p)+", loss:"+str(l))
            print("test with loss_g:"+str(l_g)+", loss_p:"+str(l_p)+", loss:"+str(l))
    
    def finish(self):
        self.log.method("finish")
        self.log.write("job finish")
        print("job finish")
        self.log.close()
