import torch
import torch.nn as nn
import math
import json

from bin.test import test
from utils.tensorShuffle import tensor_shuffle
from utils.log import Log
from utils.leftTime import LeftTime
from utils.dataInfo import DataInfo

def softmax(X):
    X_exp=torch.exp(X)
    partition=X_exp.sum(1,keepdim=True)
    return X_exp/partition

def Norm(X:torch.tensor):
    n=torch.norm(X)
    if n==0:
        return X
    else:
        return X/n

def train(dataLoader,batchSize:int,epoch:int,model:nn.Module,device,optimizer,lossFunction,train_type:str,test_list:dict=None,current_result:dict=None,settings:json=None,log:Log=None,data_info:DataInfo=None,leftTime:LeftTime=None):
    model.train()
    if train_type=="particle":
        self_acc=current_result["acc"]
        self_q=current_result["q"]
    elif train_type=="energy":
        self_l=current_result["l"]
    elif train_type=="position":
        self_l=current_result["l"]
        self_l_0=current_result["l_0"]
        self_l_1=current_result["l_1"]
    elif train_type=="angle":
        self_l=current_result["l"]
    # if need_data_info:
    #     data_info_list=[]
    # else:
    #     data_info_list=None
    for t in range(epoch):
        # data_train,label_train=tensor_shuffle(train_type,dataTensor,labelTensor)
        # if need_data_info:
        #     data_info_item={
        #         "train":[],
        #         "test":[]
        #     }
        if log!=None:
            log.write('training on epoch '+str(t+1)+'/'+str(epoch))
        print('training on epoch '+str(t+1)+'/'+str(epoch))
        if leftTime!=None:
            leftTime.startEpochTraining()
        
        current_index=0
        valid_number=0
        if train_type=="position":
            valid_number_0=0
            valid_number_1=0
        data_length=len(dataLoader)*batchSize
        
        if train_type=="energy":
            for data1,data2,label in dataLoader:
                if current_index%5000==0:
                    if current_index==0:
                        p="Training: epoch current progress: "+str(current_index)+'/'+str(data_length)
                        print(p)
                        log.write(p)
                    else:
                        state,hms,da,total_completion,epoch_completion=leftTime.trainLeftTime(current_index)
                        if state==0:
                            p="Training ("+total_completion+"%): epoch current progress: "+str(current_index)+'/'+str(data_length)+", estimated total remaining time: "+hms+", estimated completion time: "+da
                        else:
                            p="Training ("+total_completion+"%): epoch current progress: "+str(current_index)+'/'+str(data_length)+", estimated total remaining time: >"+hms+", estimated completion time: >"+da
                        print(p)
                        log.write(p)
                
                data1,data2,label=data1.to(device),data2.to(device),label.to(device)
                optimizer.zero_grad()
                if train_type=="energy":
                    y_hat=model(data1,data2)
                    loss=lossFunction(y_hat.reshape(1),label)
                    valid_number=valid_number+loss.sum().item()
                # elif train_type=="angle":
                #     y_hat=Norm(model(data1,data2))
                #     loss=lossFunction(y_hat,label)
                #     valid_number=valid_number+loss.sum().item()

                loss.mean().backward()
                optimizer.step()
                # 进行损失以及正确率记录
                if data_info:
                    if train_type=="energy":
                        data_info.add_train_info({"batchSize":batchSize,"loss":loss.sum().item()})
                    # elif train_type=="angle":
                    #     data_info.add_train_info({"batchSize":batchSize,"loss":loss.sum().item()})
                # if need_data_info:
                #     with torch.no_grad():
                #         data_info_item["train"].append({"batchSize":batchSize,"loss":loss.sum().item()})

                current_index=current_index+batchSize

        else:
            for data,label in dataLoader:
                if current_index%5000==0:
                    # if log!=None:
                    #     log.write(str(current_index)+'/'+str(data_length))
                    # print(str(current_index)+'/'+str(data_length))
                    if current_index==0:
                        p="Training: epoch current progress: "+str(current_index)+'/'+str(data_length)
                        print(p)
                        log.write(p)
                    else:
                        state,hms,da,total_completion,epoch_completion=leftTime.trainLeftTime(current_index)
                        if state==0:
                            p="Training ("+total_completion+"%): epoch current progress: "+str(current_index)+'/'+str(data_length)+", estimated total remaining time: "+hms+", estimated completion time: "+da
                        else:
                            p="Training ("+total_completion+"%): epoch current progress: "+str(current_index)+'/'+str(data_length)+", estimated total remaining time: >"+hms+", estimated completion time: >"+da
                        print(p)
                        log.write(p)
                
                data,label=data.to(device),label.to(device)
                optimizer.zero_grad()
                if train_type=="particle":
                    y_hat=softmax(model(data))
                    m=torch.argmax(y_hat,dim=1)
                    s=(m==label).int()
                    valid_number=valid_number+s.sum().item()
                elif train_type=="position":
                    y_hat=model(data)
                elif train_type=="angle":
                    y_hat=Norm(model(data))
                
                if train_type=="particle":
                    loss=lossFunction(y_hat,label)
                elif train_type=="position":
                    loss=lossFunction(y_hat,label)
                    # print(y_hat,current_label,loss)
                    valid_number=valid_number+loss.sum().item()
                    y_hat_0,y_hat_1=torch.split(y_hat,1,dim=1)
                    current_label_0,current_label_1=torch.split(label,1,dim=1)
                    with torch.no_grad():
                        loss_0=lossFunction(y_hat_0,current_label_0)
                        loss_1=lossFunction(y_hat_1,current_label_1)
                        valid_number_0=valid_number_0+loss_0.sum().item()
                        valid_number_1=valid_number_1+loss_1.sum().item()
                    # print(loss.sum().item(),loss_0.sum().item(),loss_1.sum().item())
                    # print(y_hat,current_label)
                elif train_type=="angle":
                    loss=lossFunction(y_hat,label)
                    valid_number=valid_number+loss.sum().item()
                    
                loss.mean().backward()
                optimizer.step()
                # 进行损失以及正确率记录
                if data_info:
                    with torch.no_grad():
                        if train_type=="particle":
                            data_info.add_train_info({"batchSize":batchSize,"loss":loss.item(),"acc":s.sum().item()})
                        elif train_type=="position":
                            data_info.add_train_info({"batchSize":batchSize,"loss":loss.sum().item(),"loss_0":loss_0.sum().item(),"loss_1":loss_1.sum().item()})
                        elif train_type=="angle":
                            data_info.add_train_info({"batchSize":batchSize,"loss":loss.sum().item()})

                current_index=current_index+batchSize
        
        if data_info:
            data_info.finish_train_info()
        leftTime.endEpochTrainin()

        if train_type=="particle":
            if log!=None:
                log.write("training accuracy: "+str(valid_number/current_index))
            print("training accuracy: "+str(valid_number/current_index))
        elif train_type=="energy":
            if log!=None:
                log.write("training loss: "+str(valid_number/current_index))
            print("training loss: "+str(valid_number/current_index))
        elif train_type=="position":
            if log!=None:
                log.write("training loss: "+str(valid_number/current_index))
                log.write("training loss_0: "+str(valid_number_0/current_index))
                log.write("training loss_1: "+str(valid_number_1/current_index))
            print("training loss: "+str(valid_number/current_index))
            print("training loss_0: "+str(valid_number_0/current_index))
            print("training loss_1: "+str(valid_number_1/current_index))
        elif train_type=="angle":
            if log!=None:
                log.write("training loss: "+str(valid_number/current_index))
            print("training loss: "+str(valid_number/current_index))
            # print(valid_number,current_index)
    
        if train_type=="particle":
            gamma_eff,gamma_total,proton_eff,proton_total=test(test_list["test_data_list"],test_list["test_label_list"],test_list["test_type_list"],test_list["test_energy_list"],model,device,train_type,False,log,leftTime,data_info)
            acc_g=gamma_eff/gamma_total
            acc_p=proton_eff/proton_total
            acc=(gamma_eff+proton_eff)/(gamma_total+proton_total)
            try:
                q=acc_g/math.sqrt(1-acc_p)
            except:
                q=acc_g/math.sqrt(1-acc_p+0.00001)
            
            if data_info:
                data_info.add_test_info("all",{"acc_g":acc_g,"acc_p":acc_p,"acc":acc,"q":q})
                # data_info_item["test"].append({"acc_g":acc_g,"acc_p":acc_p,"acc":acc,"q":q})

            res="acc_g:%f, acc_p:%f, acc:%f, q:%f" % (acc_g,acc_p,acc,q)
            log.write("epoch "+str(t+1)+" finish")
            log.write(res)
            print("epoch "+str(t+1)+" finish")
            print(res)

            if q>=self_q:
                if settings["GPU"]["multiple"]:
                    torch.save({'model':model.module.state_dict(),'acc':acc,'q':q},current_result["model_file"])
                else:
                    torch.save({'model':model.state_dict(),'acc':acc,'q':q},current_result["model_file"])
                self_acc=acc
                self_q=q
                log.write("model update")
                print("model update")
            else:
                # _model=torch.load(current_result["model_file"])["model"]
                log.write("model continue...")
                print("model continue...")
        
        elif train_type=="energy":
            gamma_eff,gamma_total,proton_eff,proton_total=test(test_list["test_data_list"],test_list["test_label_list"],test_list["test_type_list"],test_list["test_energy_list"],model,device,train_type,False,log,leftTime,data_info)
            l_g=gamma_eff/(gamma_total+0.0001)
            l_p=proton_eff/(proton_total+0.0001)
            l=(gamma_eff+proton_eff)/(gamma_total+proton_total)

            if data_info:
                data_info.add_test_info("all",{"l_g":l_g,"l_p":l_p,"l":l})
                # data_info_item["test"].append({"l_g":l_g,"l_p":l_p,"l":l})

            res="l_g:%f, l_p:%f, l:%f" % (l_g,l_p,l)
            log.write("epoch "+str(t+1)+" finish")
            log.write(res)
            print("epoch "+str(t+1)+" finish")
            print(res)

            if l<=self_l or self_l==0:
                if settings["GPU"]["multiple"]:
                    torch.save({'model':model.module.state_dict(),'l':l},current_result["model_file"])
                else:
                    torch.save({'model':model.state_dict(),'l':l},current_result["model_file"])
                self_l=l
                log.write("model update")
                print("model update")
            else:
                # _model=torch.load(current_result["model_file"])["model"]
                log.write("model continue...")
                print("model continue...")
        
        elif train_type=="position":
            gamma_eff,gamma_loss_0,gamma_loss_1,gamma_total,proton_eff,proton_loss_0,proton_loss_1,proton_total=test(test_list["test_data_list"],test_list["test_label_list"],test_list["test_type_list"],test_list["test_energy_list"],model,device,train_type,False,log,leftTime,data_info)
            l_g=gamma_eff/(gamma_total+0.0001)
            l_g_0=gamma_loss_0/(gamma_total+0.0001)
            l_g_1=gamma_loss_1/(gamma_total+0.0001)
            l_p=proton_eff/(proton_total+0.0001)
            l_p_0=proton_loss_0/(proton_total+0.0001)
            l_p_1=proton_loss_1/(proton_total+0.0001)
            l=(gamma_eff+proton_eff)/(gamma_total+proton_total+0.0001)
            l_0=(gamma_loss_0+proton_loss_0)/(gamma_total+proton_total+0.0001)
            l_1=(gamma_loss_1+proton_loss_1)/(gamma_total+proton_total+0.0001)

            if data_info:
                data_info.add_test_info("all",{
                    "l_g":l_g,"l_g_0":l_g_0,"l_g_1":l_g_1,
                    "l_p":l_p,"l_p_0":l_p_0,"l_p_1":l_p_1,
                    "l":l,"l_0":l_0,"l_1":l_1
                })
                # data_info_item["test"].append({
                #     "l_g":l_g,"l_g_0":l_g_0,"l_g_1":l_g_1,
                #     "l_p":l_p,"l_p_0":l_p_0,"l_p_1":l_p_1,
                #     "l":l,"l_0":l_0,"l_1":l_1
                # })

            res_log="l_g:%f, l_g_0:%f, l_g_1:%f, l_p:%f, l_p_0:%f, l_p_1:%f, l:%f, l_0:%f, l_1:%f" % (l_g,l_g_0,l_g_1,l_p,l_p_0,l_p_1,l,l_0,l_1)
            res_print="l_g:%f, l_g_0:%f, l_g_1:%f,\nl_p:%f, l_p_0:%f, l_p_1:%f,\nl:%f, l_0:%f, l_1:%f" % (l_g,l_g_0,l_g_1,l_p,l_p_0,l_p_1,l,l_0,l_1)
            log.write("epoch "+str(t+1)+" finish")
            log.write(res_log)
            print("epoch "+str(t+1)+" finish")
            print(res_print)

            if l<=self_l or self_l==0:
                if settings["GPU"]["multiple"]:
                    torch.save({'model':model.module.state_dict(),'l':l,'l_0':l_0,'l_1':l_1},current_result["model_file"])
                else:
                    torch.save({'model':model.state_dict(),'l':l,'l_0':l_0,'l_1':l_1},current_result["model_file"])
                self_l=l
                self_l_0=l_0
                self_l_1=l_1
                log.write("model update")
                print("model update")
            else:
                # _model=torch.load(current_result["model_file"])["model"]
                log.write("model continue...")
                print("model continue...")

        elif train_type=="angle":
            gamma_eff,gamma_total,proton_eff,proton_total=test(test_list["test_data_list"],test_list["test_label_list"],test_list["test_type_list"],test_list["test_energy_list"],model,device,train_type,False,log,leftTime,data_info)
            l_g=gamma_eff/(gamma_total+0.0001)
            l_p=proton_eff/(proton_total+0.0001)
            l=(gamma_eff+proton_eff)/(gamma_total+proton_total)

            if data_info:
                data_info.add_test_info("all",{"l_g":l_g,"l_p":l_p,"l":l})
                # data_info_item["test"].append({"l_g":l_g,"l_p":l_p,"l":l})

            res="l_g:%f, l_p:%f, l:%f" % (l_g,l_p,l)
            log.write("epoch "+str(t+1)+" finish")
            log.write(res)
            print("epoch "+str(t+1)+" finish")
            print(res)

            if l<=self_l or self_l==0:
                if settings["GPU"]["multiple"]:
                    torch.save({'model':model.module.state_dict(),'l':l},current_result["model_file"])
                else:
                    torch.save({'model':model.state_dict(),'l':l},current_result["model_file"])
                self_l=l
                log.write("model update")
                print("model update")
            else:
                # _model=torch.load(current_result["model_file"])["model"]
                log.write("model continue...")
                print("model continue...")

        if data_info:
            data_info.finish_test_info()

    if train_type=="particle":
        return model,{"acc":self_acc,"q":self_q}
    elif train_type=="energy":
        return model,{"l":self_l}
    elif train_type=="position":
        return model,{"l":self_l,"l_0":self_l_0,"l_1":self_l_1}
    elif train_type=="angle":
        return model,{"l":self_l}
    else:
        raise Exception("invalid train type")
