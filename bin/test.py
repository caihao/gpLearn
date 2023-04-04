import torch
import torch.nn as nn

from utils.log import Log

def softmax(X):
    X_exp=torch.exp(X)
    partition=X_exp.sum(1,keepdim=True)
    return X_exp/partition

def test(test_data_list:list,test_label_list:list,test_type_list:list,test_energy_list:list,model,device,useSoftmax:bool=True,log:Log=None):
    batchSize=1
    if log!=None:
        log.write("testing...")
    print("testing...")
    if not useSoftmax:
        loss_function=nn.MSELoss()
    assert len(test_data_list)==len(test_label_list) and len(test_label_list)==len(test_type_list) and len(test_type_list)==len(test_energy_list)
    gamma_eff=0
    gamma_total=0
    proton_eff=0
    proton_total=0
    for i in range(len(test_data_list)):
        item_eff=0
        item_total=0
        for j in range(len(test_data_list[i])):
            if useSoftmax:
                y_hat=softmax(model(test_data_list[i][j:j+batchSize].to(device)))
                m=torch.argmax(y_hat,dim=1)
                s=(m==test_label_list[i][j:j+batchSize].to(device)).int()
                item_eff=item_eff+s.sum().item()
                item_total=item_total+len(y_hat)
            else:
                y_hat=model(test_data_list[i][j:j+batchSize].to(device))
                item_eff=item_eff+loss_function(y_hat.reshape(1),test_label_list[i][j:j+batchSize].to(device)).sum().item()
                item_total=item_total+len(y_hat)
        if test_type_list[i]==0:
            gamma_eff=gamma_eff+item_eff
            gamma_total=gamma_total+item_total
            if useSoftmax:
                if log!=None:
                    log.write("gamma"+str(test_energy_list[i])+' accuracy '+str(item_eff/item_total))
                print("gamma"+str(test_energy_list[i])+' accuracy '+str(item_eff/item_total))
            else:
                if log!=None:
                    log.write("gamma"+str(test_energy_list[i])+' loss '+str(item_eff/item_total))
                print("gamma"+str(test_energy_list[i])+' loss '+str(item_eff/item_total))
        elif test_type_list[i]==1:
            proton_eff=proton_eff+item_eff
            proton_total=proton_total+item_total
            if useSoftmax:
                if log!=None:
                    log.write("proton"+str(test_energy_list[i])+' accuracy '+str(item_eff/item_total))
                print("proton"+str(test_energy_list[i])+' accuracy '+str(item_eff/item_total))
            else:
                if log!=None:
                    log.write("proton"+str(test_energy_list[i])+' loss '+str(item_eff/item_total))
                print("proton"+str(test_energy_list[i])+' loss '+str(item_eff/item_total))
        else:
            if log!=None:
                log.error("invalid type")
            raise Exception("invalid type")
    return gamma_eff,gamma_total,proton_eff,proton_total


def test_jupyter(test_data_list:list,test_label_list:list,test_type_list:list,test_energy_list:list,model,device,useSoftmax:bool=True,log:Log=None):
    if log!=None:
        log.write("testing...")
    print("testing...")
    loss_total=0
    if useSoftmax:
        loss_function=nn.CrossEntropyLoss()
    else:
        loss_function=nn.MSELoss()
    assert len(test_data_list)==len(test_label_list) and len(test_label_list)==len(test_type_list) and len(test_type_list)==len(test_energy_list)
    gamma_eff=0
    gamma_total=0
    proton_eff=0
    proton_total=0
    loss_total=0
    for i in range(len(test_data_list)):
        item_eff=0
        item_total=0
        for j in range(len(test_data_list[i])):
            if useSoftmax:
                y_hat=softmax(model(test_data_list[i][j:j+1].to(device)))
                m=torch.argmax(y_hat,dim=1)
                s=(m==test_label_list[i][j:j+1].to(device)).int()
                loss_total=loss_total+loss_function(y_hat,test_label_list[i][j:j+1].to(device))
                item_eff=item_eff+s.sum().item()
                item_total=item_total+len(y_hat)
            else:
                y_hat=model(test_data_list[i][j:j+1].to(device))
                item_eff=item_eff+loss_function(y_hat.reshape(1),test_label_list[i][j:j+1].to(device)).sum().item()
                item_total=item_total+len(y_hat)
        if test_type_list[i]==0:
            gamma_eff=gamma_eff+item_eff
            gamma_total=gamma_total+item_total
            if useSoftmax:
                if log!=None:
                    log.write("gamma"+str(test_energy_list[i])+' accuracy '+str(item_eff/item_total))
                print("gamma"+str(test_energy_list[i])+' accuracy '+str(item_eff/item_total))
            else:
                if log!=None:
                    log.write("gamma"+str(test_energy_list[i])+' loss '+str(item_eff/item_total))
                print("gamma"+str(test_energy_list[i])+' loss '+str(item_eff/item_total))
        elif test_type_list[i]==1:
            proton_eff=proton_eff+item_eff
            proton_total=proton_total+item_total
            if useSoftmax:
                if log!=None:
                    log.write("proton"+str(test_energy_list[i])+' accuracy '+str(item_eff/item_total))
                print("proton"+str(test_energy_list[i])+' accuracy '+str(item_eff/item_total))
            else:
                if log!=None:
                    log.write("proton"+str(test_energy_list[i])+' loss '+str(item_eff/item_total))
                print("proton"+str(test_energy_list[i])+' loss '+str(item_eff/item_total))
        else:
            if log!=None:
                log.error("invalid type")
            raise Exception("invalid type")
    if useSoftmax:
        return gamma_eff/gamma_total,proton_eff/proton_total,loss_total/(gamma_total+proton_total)
    return gamma_eff/gamma_total,proton_eff/proton_total,(gamma_eff+proton_eff)/(gamma_total+proton_total)

