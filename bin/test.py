import torch
import torch.nn as nn
import math
import time
import numpy as np
import json

from utils.std import std_q
from utils.log import Log
from utils.leftTime import LeftTime
from utils.dataInfo import DataInfo

# define temp g/p particle defination
# particle_gamma_list=[]
# particle_proton_list=[]

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

def angle(priPoint:list,recPoint:list):
    Deg2Rad = math.pi / 180
    Rad2Deg = 180 / math.pi
    priTheta = 20
    recTheta = 20
    priPhi = math.atan(priPoint[1] / priPoint[0]) * Rad2Deg
    recPhi = math.atan(recPoint[1] / recPoint[0]) * Rad2Deg
    pripx = math.sin(priTheta * Deg2Rad) * math.sin(priPhi * Deg2Rad)
    pripy = math.sin(priTheta * Deg2Rad) * math.cos(priPhi * Deg2Rad)
    pripz = -math.cos(priTheta * Deg2Rad)
    recpx = math.sin(recTheta * Deg2Rad) * math.sin(recPhi * Deg2Rad)
    recpy = math.sin(recTheta * Deg2Rad) * math.cos(recPhi * Deg2Rad)
    recpz = -math.cos(recTheta * Deg2Rad)
    return math.acos(pripx * recpx + pripy * recpy + pripz * recpz) * Rad2Deg

def test(test_data_list:list,test_label_list:list,test_type_list:list,test_energy_list:list,model,device,train_type:str,final_test:bool=False,log:Log=None,leftTime:LeftTime=None,data_info:DataInfo=None):
    model.eval()
    assert len(test_data_list)==len(test_label_list)==len(test_type_list)==len(test_energy_list)
    batchSize=1
    if log!=None:
        log.write("testing...")
    print("testing...")
    if not final_test and leftTime!=None:
        leftTime.startEpochTesting()
    
    gamma_info={}
    proton_info={}
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
        result_list=[]
        if train_type=="position":
            result_0_list=[]
            result_1_list=[]
            result_angle_list=[]
        elif train_type=="angle":
            result_angle_list=[]

        # ss=[]
        # sso=[]

        if train_type=="particle":
            for j in range(len(test_data_list[i])):
                y_hat=softmax(model(test_data_list[i][j:j+batchSize].to(device)))
                m=torch.argmax(y_hat,dim=1)
                s=(m==test_label_list[i][j:j+batchSize].to(device)).int()
                # item_eff=item_eff+s.sum().item()
                # item_total=item_total+len(y_hat)

                # if test_label_list[i][j:j+batchSize].item()==0:#g
                #     particle_gamma_list.append(y_hat[0][1].item())
                # elif test_label_list[i][j:j+batchSize].item()==1:#p
                #     particle_proton_list.append(y_hat[0][1].item())

                result_list.append(s.sum().item()/len(y_hat))
        elif train_type=="energy":
            for j in range(len(test_data_list[i][0])):
                y_hat=model(test_data_list[i][0][j:j+batchSize].to(device),test_data_list[i][1][j:j+batchSize].to(device))
                # item_eff=item_eff+loss_function(y_hat.reshape(1),test_label_list[i][j:j+batchSize].to(device)).sum().item()
                # item_total=item_total+len(y_hat)
                result_list.append(loss_function(y_hat.reshape(1),test_label_list[i][j:j+batchSize].to(device)).sum().item()/len(y_hat))
        elif train_type=="position":
            for j in range(len(test_data_list[i])):
                y_hat=model(test_data_list[i][j:j+batchSize].to(device))
                result_angle_list.append(angle(test_label_list[i][j:j+batchSize].to(device).tolist()[0],y_hat.tolist()[0]))
                # import time
                # time.sleep(10000)
                # item_eff=item_eff+loss_function(y_hat,test_label_list[i][j:j+batchSize].to(device)).sum().item()
                # item_total=item_total+len(y_hat)
                result_list.append(loss_function(y_hat,test_label_list[i][j:j+batchSize].to(device)).sum().item()/len(y_hat))
                y_hat_0,y_hat_1=torch.split(y_hat,1,dim=1)
                current_label_0,current_label_1=torch.split(test_label_list[i][j:j+batchSize].to(device),1,dim=1)
                with torch.no_grad():
                    # loss_0=loss_0+loss_function(y_hat_0,current_label_0).sum().item()
                    # loss_1=loss_1+loss_function(y_hat_1,current_label_1).sum().item()
                    result_0_list.append(loss_function(y_hat_0,current_label_0).sum().item()/len(y_hat))
                    result_1_list.append(loss_function(y_hat_1,current_label_1).sum().item()/len(y_hat))
        elif train_type=="angle":
            for j in range(len(test_data_list[i])):
                y_hat=Norm(model(test_data_list[i][j:j+batchSize].to(device)))
                # y_hat=model(test_data_list[i][j:j+batchSize].to(device))
                result_list.append(loss_function(y_hat,test_label_list[i][j:j+batchSize].to(device)).sum().item()/len(y_hat))
                priUnvt=test_label_list[i][j:j+batchSize].tolist()[0]
                recUnvt=y_hat.tolist()[0]
                ea=angle(priUnvt,recUnvt)
                # ea=abs(priUnvt[0]-recUnvt[0])
                result_angle_list.append(ea)
                
            #     if ea<5:
            #         ss.append(ea)
            #         sso.append(test_label_list[i][j:j+batchSize][0][2].item())
            # ss.sort()
            # sso.sort()
            # print(ss)
            # print(sso)

            # print(ss[int(len(ss)*0.68)])
            # print(sso[int(len(sso)*0.68)])

        if test_type_list[i]==0:
            # gamma_eff=gamma_eff+item_eff
            # gamma_total=gamma_total+item_total
            gamma_eff=gamma_eff+sum(result_list)
            gamma_total=gamma_total+len(result_list)
            if train_type=="particle":
                # acc_info["gamma"][str(test_energy_list[i])]=item_eff/item_total
                # gamma_info[str(test_energy_list[i])]={"acc":item_eff/item_total,"time":int(time.time())}
                avg=sum(result_list)/len(result_list)
                std=np.std(result_list)
                acc_info["gamma"][str(test_energy_list[i])]=avg
                gamma_info[str(test_energy_list[i])]={"acc":avg,"std":std,"time":int(time.time())}
            elif train_type=="energy":
                # gamma_info[str(test_energy_list[i])]={"loss":item_eff/item_total,"time":int(time.time())}
                avg=sum(result_list)/len(result_list)
                std=np.std(result_list)
                gamma_info[str(test_energy_list[i])]={"loss":avg,"std":std,"time":int(time.time())}
                if log!=None:
                    log.write("gamma"+str(test_energy_list[i])+" loss "+str(avg))
                print("gamma"+str(test_energy_list[i])+" loss "+str(avg))
            elif train_type=="position":
                # gamma_loss_0=gamma_loss_0+loss_0
                # gamma_loss_1=gamma_loss_1+loss_1
                gamma_loss_0=gamma_loss_0+sum(result_0_list)
                gamma_loss_1=gamma_loss_1+sum(result_1_list)
                # gamma_info[str(test_energy_list[i])]={"loss":item_eff/item_total,"loss_0":loss_0/item_total,"loss_1":loss_1/item_total,"time":int(time.time())}
                avg=sum(result_list)/len(result_list)
                avg_0=sum(result_0_list)/len(result_list)
                avg_1=sum(result_1_list)/len(result_list)
                std=np.std(result_list)
                std_0=np.std(result_0_list)
                std_1=np.std(result_1_list)

                result_angle_list.sort()
                result_angle_list=result_angle_list[:int(len(result_angle_list)*0.68)]
                angle_err=result_angle_list[-1]
                std_angle=np.std(result_angle_list)
                gamma_info[str(test_energy_list[i])]={"loss":avg,"loss_0":avg_0,"loss_1":avg_1,"std":std,"std_0":std_0,"std_1":std_1,"angle":angle_err,"std_angle":std_angle,"time":int(time.time())}
                
                # if log!=None:
                #     log.write("gamma"+str(test_energy_list[i])+" loss:"+str(item_eff/item_total)+", loss_0:"+str(loss_0/item_total)+", loss_1:"+str(loss_1/item_total))
                # print("gamma"+str(test_energy_list[i])+" loss:"+str(item_eff/item_total)+", loss_0:"+str(loss_0/item_total)+", loss_1:"+str(loss_1/item_total))
                if log!=None:
                    log.write("gamma"+str(test_energy_list[i])+" loss:"+str(avg)+", loss_0:"+str(avg_0)+", loss_1:"+str(avg_1)+", angle:"+str(angle_err))
                print("gamma"+str(test_energy_list[i])+" loss:"+str(avg)+", loss_0:"+str(avg_0)+", loss_1:"+str(avg_1)+", angle:"+str(angle_err))
            elif train_type=="angle":
                avg=sum(result_list)/len(result_list)
                std=np.std(result_list)

                result_angle_list.sort()
                result_angle_list=result_angle_list[:int(len(result_angle_list)*0.68)]
                angle_err=result_angle_list[-1]
                std_angle=np.std(result_angle_list)

                # gamma_info[str(test_energy_list[i])]={"loss":item_eff/item_total,"time":int(time.time())}
                gamma_info[str(test_energy_list[i])]={"loss":avg,"std":std,"angle":angle_err,"std_angle":std_angle,"time":int(time.time())}
                # if log!=None:
                #     log.write("gamma"+str(test_energy_list[i])+' loss '+str(item_eff/item_total))
                # print("gamma"+str(test_energy_list[i])+' loss '+str(item_eff/item_total))
                if log!=None:
                    log.write("gamma"+str(test_energy_list[i])+' loss '+str(avg)+", angle: "+str(angle_err))
                print("gamma"+str(test_energy_list[i])+' loss '+str(avg)+", angle: "+str(angle_err))
        
        elif test_type_list[i]==1:
            # proton_eff=proton_eff+item_eff
            # proton_total=proton_total+item_total
            proton_eff=proton_eff+sum(result_list)
            proton_total=proton_total+len(result_list)
            if train_type=="particle":
                # acc_info["proton"][str(test_energy_list[i])]=item_eff/item_total
                # proton_info[str(test_energy_list[i])]={"acc":item_eff/item_total,"time":int(time.time())}
                avg=sum(result_list)/len(result_list)
                std=np.std(result_list)
                acc_info["proton"][str(test_energy_list[i])]=avg
                proton_info[str(test_energy_list[i])]={"acc":avg,"std":std,"time":int(time.time())}
            elif train_type=="energy":
                # proton_info[str(test_energy_list[i])]={"loss":item_eff/item_total,"time":int(time.time())}
                avg=sum(result_list)/len(result_list)
                std=np.std(result_list)
                proton_info[str(test_energy_list[i])]={"loss":avg,"std":std,"time":int(time.time())}
                # if log!=None:
                #     log.write("proton"+str(test_energy_list[i])+' loss '+str(item_eff/item_total))
                # print("proton"+str(test_energy_list[i])+' loss '+str(item_eff/item_total))
                if log!=None:
                    log.write("proton"+str(test_energy_list[i])+' loss '+str(avg))
                print("proton"+str(test_energy_list[i])+' loss '+str(avg))
            elif train_type=="position":
                # proton_loss_0=proton_loss_0+loss_0
                # proton_loss_1=proton_loss_1+loss_1
                proton_loss_0=proton_loss_0+sum(result_0_list)
                proton_loss_1=proton_loss_1+sum(result_1_list)
                # proton_info[str(test_energy_list[i])]={"loss":item_eff/item_total,"loss_0":loss_0/item_total,"loss_1":loss_1/item_total,"time":int(time.time())}
                avg=sum(result_list)/len(result_list)
                avg_0=sum(result_0_list)/len(result_list)
                avg_1=sum(result_1_list)/len(result_list)
                std=np.std(result_list)
                std_0=np.std(result_0_list)
                std_1=np.std(result_1_list)

                result_angle_list.sort()
                result_angle_list=result_angle_list[:int(len(result_angle_list)*0.68)]
                angle_err=result_angle_list[-1]
                std_angle=np.std(result_angle_list)
                proton_info[str(test_energy_list[i])]={"loss":avg,"loss_0":avg_0,"loss_1":avg_1,"std":std,"std_0":std_0,"std_1":std_1,"angle":angle_err,"std_angle":std_angle,"time":int(time.time())}

                # if log!=None:
                #     log.write("proton"+str(test_energy_list[i])+" loss:"+str(item_eff/item_total)+", loss_0:"+str(loss_0/item_total)+", loss_1:"+str(loss_1/item_total))
                # print("proton"+str(test_energy_list[i])+" loss:"+str(item_eff/item_total)+", loss_0:"+str(loss_0/item_total)+", loss_1:"+str(loss_1/item_total))
                if log!=None:
                    log.write("proton"+str(test_energy_list[i])+" loss:"+str(avg)+", loss_0:"+str(avg_0)+", loss_1:"+str(avg_1)+", angle:"+str(angle_err))
                print("proton"+str(test_energy_list[i])+" loss:"+str(avg)+", loss_0:"+str(avg_0)+", loss_1:"+str(avg_1)+", angle:"+str(angle_err))
            elif train_type=="angle":
                # proton_info[str(test_energy_list[i])]={"loss":item_eff/item_total,"time":int(time.time())}
                avg=sum(result_list)/len(result_list)
                std=np.std(result_list)

                result_angle_list.sort()
                result_angle_list=result_angle_list[:int(len(result_angle_list)*0.68)]
                angle_err=result_angle_list[-1]
                std_angle=np.std(result_angle_list)

                proton_info[str(test_energy_list[i])]={"loss":avg,"std":std,"angle":angle_err,"std_angle":std_angle,"time":int(time.time())}
                # if log!=None:
                #     log.write("proton"+str(test_energy_list[i])+' loss '+str(item_eff/item_total))
                # print("proton"+str(test_energy_list[i])+' loss '+str(item_eff/item_total))
                if log!=None:
                    log.write("proton"+str(test_energy_list[i])+" loss "+str(avg)+", angle: "+str(angle_err))
                print("proton"+str(test_energy_list[i])+" loss "+str(avg)+", angle: "+str(angle_err))
        else:
            if log!=None:
                log.error("invalid type")
            raise Exception("invalid type")
    
    if data_info:
        if not final_test:
            data_info.add_test_info("gamma",gamma_info)
            data_info.add_test_info("proton",proton_info)
        else:
            gamma_info_keys=[int(x) for x in gamma_info.keys()]
            gamma_info_keys.sort()
            for k in gamma_info_keys:
                if train_type=="particle":
                    data_info.add_result_particle(k,gamma_info[str(k)]["acc"],gamma_info[str(k)]["std"],"gamma")
                elif train_type=="energy":
                    data_info.add_result_energy(k,gamma_info[str(k)]["loss"],gamma_info[str(k)]["std"],"gamma")
                elif train_type=="position":
                    data_info.add_result_position(k,[gamma_info[str(k)]["loss"],gamma_info[str(k)]["loss_0"],gamma_info[str(k)]["loss_1"],gamma_info[str(k)]["angle"]],[gamma_info[str(k)]["std"],gamma_info[str(k)]["std_0"],gamma_info[str(k)]["std_1"],gamma_info[str(k)]["std_angle"]],"gamma")
                elif train_type=="angle":
                    data_info.add_result_angle(k,gamma_info[str(k)]["loss"],gamma_info[str(k)]["std"],gamma_info[str(k)]["angle"],gamma_info[str(k)]["std_angle"],"gamma")
            proton_info_keys=[int(x) for x in proton_info.keys()]
            proton_info_keys.sort()
            for k in proton_info_keys:
                if train_type=="particle":
                    data_info.add_result_particle(k,proton_info[str(k)]["acc"],proton_info[str(k)]["std"],"proton")
                elif train_type=="energy":
                    data_info.add_result_energy(k,proton_info[str(k)]["loss"],proton_info[str(k)]["std"],"proton")
                elif train_type=="position":
                    data_info.add_result_position(k,[proton_info[str(k)]["loss"],proton_info[str(k)]["loss_0"],proton_info[str(k)]["loss_1"],proton_info[str(k)]["angle"]],[proton_info[str(k)]["std"],proton_info[str(k)]["std_0"],proton_info[str(k)]["std_1"],proton_info[str(k)]["std_angle"]],"proton")
                elif train_type=="angle":
                    data_info.add_result_angle(k,proton_info[str(k)]["loss"],proton_info[str(k)]["std"],proton_info[str(k)]["angle"],proton_info[str(k)]["std_angle"],"proton")
        
    if train_type=="particle":
        q_info={}
        gamma_set=set([int(i) for i in acc_info["gamma"].keys()])
        proton_set=set([int(int(i)/3) for i in acc_info["proton"].keys()])
        union=list(gamma_set.union(proton_set))
        union.sort()
        intersection=list(gamma_set.intersection(proton_set))

        for en in union:
            if en in intersection:
                # 光子/质子同能量点数据均存在
                a_g=acc_info["gamma"][str(en)]
                a_p=acc_info["proton"][str(en*3)]
                q=a_g/math.sqrt(1-a_p+0.000001)
                # q_info[str(en)]={"q":q,"time":int(time.time())}
                q_info[str(en)]={"q":q,"std":std_q(gamma_info[str(en)]["acc"],gamma_info[str(en)]["std"],proton_info[str(int(en*3))]["acc"],proton_info[str(int(en*3))]["std"],num_samples=20000),"time":int(time.time())}
                if log!=None:
                    log.write("calculation at energy "+str(en)+" gets Q: "+str(q)+" (with acc_gamma: "+str(a_g)+" and acc_proton: "+str(a_p)+")")
                print("calculation at energy "+str(en)+" gets Q: "+str(q)+" (with acc_gamma: "+str(a_g)+" and acc_proton: "+str(a_p)+")")
            else:
                if en in list(gamma_set):
                    # 仅存在光子数据点
                    if log!=None:
                        log.write("missing data at energy "+str(en)+" for calculation Q (only with acc_gamma: "+str(acc_info["gamma"][str(en)])+")")
                    print("missing data at energy "+str(en)+" for calculation Q (only with acc_gamma: "+str(acc_info["gamma"][str(en)])+")")
                else:
                    # 仅存在质子数据点
                    if log!=None:
                        log.write("missing data at energy "+str(en*3)+" for calculation Q (only with acc_proton: "+str(acc_info["proton"][str(en*3)])+")")
                    print("missing data at energy "+str(en*3)+" for calculation Q (only with acc_proton: "+str(acc_info["proton"][str(en*3)])+")")

        if data_info:
            if not final_test:
                data_info.add_test_info("q",q_info)
                data_info.finish_test_info()
            else:
                q_info_keys=[int(x) for x in q_info.keys()]
                q_info_keys.sort()
                for k in q_info_keys:
                    data_info.add_result_particle(k,q_info[str(k)]["q"],q_info[str(k)]["std"],"q")

    if not final_test and leftTime!=None:
        leftTime.endEpochTesting()
    
    # with open("temp_particle_gp.json","w") as f:
    #     json.dump({"gamma":particle_gamma_list,"proton":particle_proton_list},f)
    #     f.close()

    if train_type=="position":
        return gamma_eff,gamma_loss_0,gamma_loss_1,gamma_total,proton_eff,proton_loss_0,proton_loss_1,proton_total
    else:
        return gamma_eff,gamma_total,proton_eff,proton_total
