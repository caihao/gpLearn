import torch
import torch.nn as nn
import math

from utils.log import Log
from utils.leftTime import LeftTime

def softmax(X):
    X_exp=torch.exp(X)
    partition=X_exp.sum(1,keepdim=True)
    return X_exp/partition

def test(test_data_list:list,test_label_list:list,test_type_list:list,test_energy_list:list,model,device,train_type:str,log:Log=None,leftTime:LeftTime=None):
    assert len(test_data_list)==len(test_label_list)==len(test_type_list)==len(test_energy_list)
    batchSize=1
    if log!=None:
        log.write("testing...")
    print("testing...")
    if leftTime!=None:
        leftTime.startEpochTesting()
    
    if train_type=="particle":
        loss_function=nn.CrossEntropyLoss(reduction="sum")
        # 对于背景抑制工作，针对同能量的光子/质子正确率，需要计算品质因子
        acc_info={"gamma":{},"proton":{}}
    elif train_type=="energy":
        loss_function=nn.MSELoss(reduction="sum")
    elif train_type=="position":
        loss_function=nn.MSELoss(reduction="sum")
    elif train_type=="angle":
        loss_function=nn.MSELoss(reduction="sum")
    
    gamma_eff=0
    gamma_total=0
    proton_eff=0
    proton_total=0
    if train_type=="position":
        gamma_loss_0=0
        gamma_loss_1=0
        proton_loss_0=0
        proton_loss_1=0
    for i in range(len(test_data_list)):
        item_eff=0
        item_total=0
        if train_type=="position":
            loss_0=0
            loss_1=0
        
        if train_type=="particle":
            for j in range(len(test_data_list[i])):
                y_hat=softmax(model(test_data_list[i][j:j+batchSize].to(device)))
                m=torch.argmax(y_hat,dim=1)
                s=(m==test_label_list[i][j:j+batchSize].to(device)).int()
                item_eff=item_eff+s.sum().item()
                item_total=item_total+len(y_hat)
        elif train_type=="energy":
            for j in range(len(test_data_list[i][0])):
                y_hat=model(test_data_list[i][0][j:j+batchSize].to(device),test_data_list[i][1][j:j+batchSize].to(device))
                # print(y_hat)
                # y_hat=model(test_data_list[i][0][j:j+batchSize].to(device))
                item_eff=item_eff+loss_function(y_hat.reshape(1),test_label_list[i][j:j+batchSize].to(device)).sum().item()
                item_total=item_total+len(y_hat)
        elif train_type=="position":
            for j in range(len(test_data_list[i])):
                y_hat=model(test_data_list[i][j:j+batchSize].to(device))
                # y_hat=model(test_data_list[i][0][j:j+batchSize].to(device),test_data_list[i][1][j:j+batchSize].to(device))
                item_eff=item_eff+loss_function(y_hat,test_label_list[i][j:j+batchSize].to(device)).sum().item()
                item_total=item_total+len(y_hat)
                y_hat_0,y_hat_1=torch.split(y_hat,1,dim=1)
                current_label_0,current_label_1=torch.split(test_label_list[i][j:j+batchSize].to(device),1,dim=1)
                with torch.no_grad():
                    loss_0=loss_0+loss_function(y_hat_0,current_label_0).sum().item()
                    loss_1=loss_1+loss_function(y_hat_1,current_label_1).sum().item()
        elif train_type=="angle":
            for j in range(len(test_data_list[i])):
                y_hat=model(test_data_list[i][j:j+batchSize].to(device))
                item_eff=item_eff+loss_function(y_hat,test_label_list[i][j:j+batchSize].to(device)).sum().item()
                item_total=item_total+len(y_hat)

        if test_type_list[i]==0:
            gamma_eff=gamma_eff+item_eff
            gamma_total=gamma_total+item_total
            if train_type=="particle":
                acc_info["gamma"][str(test_energy_list[i])]=item_eff/item_total
                # if log!=None:
                #     log.write("gamma"+str(test_energy_list[i])+' accuracy '+str(item_eff/item_total))
                # print("gamma"+str(test_energy_list[i])+' accuracy '+str(item_eff/item_total))
            elif train_type=="energy":
                if log!=None:
                    log.write("gamma"+str(test_energy_list[i])+' loss '+str(item_eff/item_total))
                print("gamma"+str(test_energy_list[i])+' loss '+str(item_eff/item_total))
            elif train_type=="position":
                gamma_loss_0=gamma_loss_0+loss_0
                gamma_loss_1=gamma_loss_1+loss_1
                if log!=None:
                    log.write("gamma"+str(test_energy_list[i])+" loss:"+str(item_eff/item_total)+", loss_0:"+str(loss_0/item_total)+", loss_1:"+str(loss_1/item_total))
                print("gamma"+str(test_energy_list[i])+" loss:"+str(item_eff/item_total)+", loss_0:"+str(loss_0/item_total)+", loss_1:"+str(loss_1/item_total))
            elif train_type=="angle":
                if log!=None:
                    log.write("gamma"+str(test_energy_list[i])+' loss '+str(item_eff/item_total))
                print("gamma"+str(test_energy_list[i])+' loss '+str(item_eff/item_total))
        
        elif test_type_list[i]==1:
            proton_eff=proton_eff+item_eff
            proton_total=proton_total+item_total
            if train_type=="particle":
                acc_info["proton"][str(test_energy_list[i])]=item_eff/item_total
                # if log!=None:
                #     log.write("proton"+str(test_energy_list[i])+' accuracy '+str(item_eff/item_total))
                # print("proton"+str(test_energy_list[i])+' accuracy '+str(item_eff/item_total))
            elif train_type=="energy":
                if log!=None:
                    log.write("proton"+str(test_energy_list[i])+' loss '+str(item_eff/item_total))
                print("proton"+str(test_energy_list[i])+' loss '+str(item_eff/item_total))
            elif train_type=="position":
                proton_loss_0=proton_loss_0+loss_0
                proton_loss_1=proton_loss_1+loss_1
                if log!=None:
                    log.write("proton"+str(test_energy_list[i])+" loss:"+str(item_eff/item_total)+", loss_0:"+str(loss_0/item_total)+", loss_1:"+str(loss_1/item_total))
                print("proton"+str(test_energy_list[i])+" loss:"+str(item_eff/item_total)+", loss_0:"+str(loss_0/item_total)+", loss_1:"+str(loss_1/item_total))
            elif train_type=="angle":
                if log!=None:
                    log.write("proton"+str(test_energy_list[i])+' loss '+str(item_eff/item_total))
                print("proton"+str(test_energy_list[i])+' loss '+str(item_eff/item_total))
        else:
            if log!=None:
                log.error("invalid type")
            raise Exception("invalid type")
    
    if train_type=="particle":
        gamma_set=set(acc_info["gamma"].keys())
        proton_set=set(acc_info["proton"].keys())
        union=list(gamma_set.union(proton_set))
        for z in range(len(union)):
            union[z]=int(union[z])
        union.sort()
        intersection=list(gamma_set.intersection(proton_set))
        for en in union:
            if str(en) in intersection:
                # 光子/质子同能量点数据均存在
                a_g=acc_info["gamma"][str(en)]
                a_p=acc_info["proton"][str(en)]
                q=a_g/math.sqrt(1-a_p)
                if log!=None:
                    log.write("calculation at energy "+str(en)+" gets Q: "+str(q)+" (with acc_gamma: "+str(a_g)+" and acc_proton: "+str(a_p)+")")
                print("calculation at energy "+str(en)+" gets Q: "+str(q)+" (with acc_gamma: "+str(a_g)+" and acc_proton: "+str(a_p)+")")
            else:
                if str(en) in list(gamma_set):
                    # 仅存在光子数据点
                    if log!=None:
                        log.write("missing data for calculation Q (only with acc_gamma: "+str(acc_info["gamma"][str(en)])+")")
                    print("missing data for calculation Q (only with acc_gamma: "+str(acc_info["gamma"][str(en)])+")")
                else:
                    # 仅存在质子数据点
                    if log!=None:
                        log.write("missing data for calculation Q (only with acc_proton: "+str(acc_info["proton"][str(en)])+")")
                    print("missing data for calculation Q (only with acc_proton: "+str(acc_info["proton"][str(en)])+")")
    
    if leftTime!=None:
        leftTime.endEpochTesting()
        
    if train_type=="position":
        return gamma_eff,gamma_loss_0,gamma_loss_1,gamma_total,proton_eff,proton_loss_0,proton_loss_1,proton_total
    else:
        return gamma_eff,gamma_total,proton_eff,proton_total
