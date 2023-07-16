import torch
import os
import pickle
import json
import matplotlib.pyplot as plt

from bin.dataLoader import load_data
from utils.path import get_project_file_path

def generate_info_data(data_file_path_relative:str,epoch_step:int=10,train_type:str="particle"):
    with open(get_project_file_path(data_file_path_relative),"rb") as f:
        data_info=pickle.load(f)
        f.close()
    
    x=[0]
    if train_type=="particle":
        train_loss=[]
        train_acc=[]
        test_acc_g=[0]
        test_acc_p=[0]
        test_acc=[0]
        test_q=[0]
    elif train_type=="energy":
        train_loss=[]
        test_l_g=[0]
        test_l_p=[0]
        test_l=[0]
    else:
        raise Exception("invalid train type")

    for item in data_info:
        for m in range(len(item)):
            x=x+[x[-1]+(i+1)/epoch_step for i in range(epoch_step)]
            # print(x)
            train_length=len(item[m]["train"])
            train_gap_points=[int(train_length*(i+1)/epoch_step) for i in range(epoch_step)]

            gap_loss=0
            gap_acc=0
            gap_total=0
            for i in range(train_length):
                if (i+1) not in train_gap_points:
                    if train_type=="particle":
                        gap_loss=gap_loss+item[m]["train"][i]["loss"]
                        gap_acc=gap_acc+item[m]["train"][i]["acc"]
                        gap_total=gap_total+item[m]["train"][i]["batchSize"]
                    elif train_type=="energy":
                        gap_loss=gap_loss+item[m]["train"][i]["loss"]
                        gap_total=gap_total+item[m]["train"][i]["batchSize"]
                else:
                    if train_type=="particle":
                        train_loss.append(gap_loss/(gap_total+0.00001))
                        train_acc.append(gap_acc/(gap_total+0.00001))
                    elif train_type=="energy":
                        train_loss.append(gap_loss/(gap_total+0.00001))
                        # print("###")
                    gap_total=0
                    gap_loss=0
                    gap_acc=0

            if train_type=="particle":
                k_acc_g=(item[m]["test"][0]["acc_g"]-test_acc_g[-1])/epoch_step
                for i in range(epoch_step):
                    test_acc_g.append(test_acc_g[-1]+k_acc_g)

                k_acc_p=(item[m]["test"][0]["acc_p"]-test_acc_p[-1])/epoch_step
                for i in range(epoch_step):
                    test_acc_p.append(test_acc_p[-1]+k_acc_p)

                k_acc=(item[m]["test"][0]["acc"]-test_acc[-1])/epoch_step
                for i in range(epoch_step):
                    test_acc.append(test_acc[-1]+k_acc)

                k_q=(item[m]["test"][0]["q"]-test_q[-1])/epoch_step
                for i in range(epoch_step):
                    test_q.append(test_q[-1]+k_q)
            
            elif train_type=="energy":
                k_l_g=(item[m]["test"][0]["l_g"]-test_l_g[-1])/epoch_step
                for i in range(epoch_step):
                    test_l_g.append(test_l_g[-1]+k_l_g)

                k_l_p=(item[m]["test"][0]["l_p"]-test_l_p[-1])/epoch_step
                for i in range(epoch_step):
                    test_l_p.append(test_l_p[-1]+k_l_p)

                k_l=(item[m]["test"][0]["l"]-test_l[-1])/epoch_step
                for i in range(epoch_step):
                    test_l.append(test_l[-1]+k_l)

    x=x[1:]
    if train_type=="particle":
        test_acc_g=test_acc_g[1:]
        test_acc_p=test_acc_p[1:]
        test_acc=test_acc[1:]
        test_q=test_q[1:]
        return x,train_loss,train_acc,test_acc_g,test_acc_p,test_acc,test_q
    elif train_type=="energy":
        test_l_g=test_l_g[1:]
        test_l_p=test_l_p[1:]
        test_l=test_l[1:]
        return x,train_loss,test_l_g,test_l_p,test_l
    
def generate_info_picture(data_file_path_relative:str,epoch_step:int=10,train_type:str="particle",picture_title:str=None):
    if train_type=="particle":
        x,train_loss,train_acc,test_acc_g,test_acc_p,test_acc,test_q=generate_info_data(data_file_path_relative,epoch_step,train_type)
    elif train_type=="energy":
        x,train_loss,test_l_g,test_l_p,test_l=generate_info_data(data_file_path_relative,epoch_step,train_type)
    else:
        raise Exception("invalid train type")
    
    if train_type=="particle":
        fig, axs = plt.subplots(2, 3)
        axs[0, 0].plot(x, train_loss, label='train_loss')
        axs[0, 1].plot(x, train_acc, label='train_acc')
        axs[0, 2].plot(x, test_acc_g, label='test_acc_gamma')
        axs[1, 0].plot(x, test_acc_p, label='test_acc_proton')
        axs[1, 1].plot(x, test_acc, label='test_acc')
        axs[1, 2].plot(x, test_q, label='test_q')

        for ax in axs.flat:
            ax.legend()
            ax.set_title(picture_title if picture_title!=None else 'particle')
        plt.show()
    
    elif train_type=="energy":
        fig, axs = plt.subplots(2, 2)
        print(len(x),len(train_loss),len(test_l_g),len(test_l_p),len(test_l))
        axs[0, 0].plot(x, train_loss, label='train_loss')
        axs[0, 1].plot(x, test_l_g, label='test_loss_gamma')
        axs[1, 0].plot(x, test_l_p, label='test_loss_proton')
        axs[1, 1].plot(x, test_l, label='test_loss')

        for ax in axs.flat:
            ax.legend()
            ax.set_title(picture_title if picture_title!=None else 'energy')
        plt.show()

def energy_distribution(particle:str,energy:int,model_file:str,total_number:int,allow_pic_number_list:list=[4,3,2,1],allow_min_pix_number:int=None,ignore_number:int=0,centering:bool=True,ran:int=500,epo:int=5):
    data,_=load_data(particle,energy,total_number,allow_pic_number_list,allow_min_pix_number,ignore_number,64,"energy",centering,None,None)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state=torch.load(get_project_file_path("data/model/"+model_file))
    model=state['model'].to(device)

    points=[]
    for i in range(len(data[0])):
        points.append(model(data[0][i:i+1].to(device),data[1][i:i+1].to(device)).reshape(1).item())
        # points.append(model(data[0][i:i+1].to(device)).reshape(1).item())
    x=[]
    y=[]
    def p(x1,x2,points):
        num=0
        for i in points:
            if i*100>=x1 and i*100<x2:
                num=num+1
        return num/len(points)
    for i in range(energy-ran,energy+ran,epo):
        x.append(i)
        y.append(p(i,i+epo,points))

    # return x,y,points
    plt.plot(x, y, 'b-', alpha=0.5, linewidth=1, label=particle+" "+str(energy)+'GeV')
    plt.legend()
    plt.xlabel('Energy(GeV)')
    plt.ylabel('Normalized Event')
    
    plt.xlim(energy-ran-100 if (energy-ran-100)>=0 else 0,energy+ran+100)
    plt.ylim(0,max(y))
    plt.show()
    return x,y

def energy_error(particle:str,energy:int,model_file:str,total_number:int,allow_pic_number_list:list=[4],allow_min_pix_number:int=None,ignore_number:int=0,centering:bool=True):
    data,_=load_data(particle,energy,total_number,allow_pic_number_list,allow_min_pix_number,ignore_number,64,"energy",centering,None,None)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state=torch.load(get_project_file_path("data/model/"+model_file))
    model=state['model'].to(device)

    points=[]
    for i in range(len(data[0])):
        points.append(model(data[0][i:i+1].to(device),data[1][i:i+1].to(device)).reshape(1).item())

    error=0
    print(points)
    for i in points:
        error=(energy/1000-i)**2+error
        # error=abs(energy-i*1000)+error
    error=error/total_number
    return error
    # return error/energy

def show_particle_picture(tensor_data,index:int,single_pic_size:int=64,_type:str="single"):
    if _type=="single":
        data=tensor_data[index].reshape(2*single_pic_size,2*single_pic_size)
        fig, axs = plt.subplots(figsize=(5,5))
        axs.imshow(data.numpy())
    elif _type=="multi":
        data=tensor_data[index]
        fig, axs = plt.subplots(2, 2, figsize=(5,5))
        axs[0, 0].imshow(data[0].numpy())
        axs[0, 1].imshow(data[1].numpy())
        axs[1, 0].imshow(data[2].numpy())
        axs[1, 1].imshow(data[3].numpy())
    else:
        raise Exception("invalid _type")
    plt.show()

def result_genereation(info_name:str):
    info_path=get_project_file_path(os.path.join("/data/info",info_name))
    if not os.path.exists(info_path):
        info_path=get_project_file_path(info_name)
        if not os.path.exists(info_path):
            raise Exception("info not found")
    with open(info_path,"rb") as f:
        d=pickle.load(f)
        f.close()
    with open(get_project_file_path("settings.json"),"r") as f:
        settings=json.load(f)["drawing"]["result"]
        f.close()
    train_type=d["info"]["train_type"]
    # train_type="particle"
    def show(energy:list,r:list,settings:dict):
        assert len(energy)==len(r)
        if settings["TeV_mode"]:
            for i in range(len(energy)):
                energy[i]=energy[i]/1000
        plt.title(settings["title"])
        plt.xlabel(settings["xlabel"])
        plt.ylabel(settings["ylabel"])
        if settings["color"]:
            plt.plot(energy,r,label=settings["label"],color=settings["color"])
        else:
            plt.plot(energy,r,label=settings["label"])
        if settings["logX_mode"]:
            plt.xscale("log")
        if settings["logY_mode"]:
            plt.yscale("log")
        plt.legend()
        if settings["save"]["switch"]:
            plt.savefig(settings["save"]["head_name"]+"_"+settings["title"]+".png" if settings["save"]["head_name"]!=None else settings["title"]+".png",dpi=settings["save"]["dpi"])
        plt.show()

    if train_type=="particle":
        energy=d["result"]["gamma"]["energy"]
        r=d["result"]["gamma"]["r"]
        if len(energy)!=0:
            show(energy,r,settings["particle"]["gamma"])
        energy=d["result"]["proton"]["energy"]
        r=d["result"]["proton"]["r"]
        if len(energy)!=0:
            show(energy,r,settings["particle"]["proton"])
        energy=d["result"]["q"]["energy"]
        r=d["result"]["q"]["r"]
        if len(energy)!=0:
            show(energy,r,settings["particle"]["q"])
    
    elif train_type=="energy":
        energy=d["result"]["gamma"]["energy"]
        r=d["result"]["gamma"]["loss"]
        if len(energy)!=0:
            show(energy,r,settings["energy"]["gamma"])
        energy=d["result"]["proton"]["energy"]
        r=d["result"]["proton"]["loss"]
        if len(energy)!=0:
            show(energy,r,settings["energy"]["proton"])
        
    elif train_type=="position":
        energy=d["result"]["gamma"]["energy"]
        r=d["result"]["gamma"]["loss"]
        if len(energy)!=0:
            show(energy,r,settings["position"]["gamma"]["loss"])
        r=d["result"]["gamma"]["loss_0"]
        if len(energy)!=0:
            show(energy,r,settings["position"]["gamma"]["loss_0"])
        r=d["result"]["gamma"]["loss_1"]
        if len(energy)!=0:
            show(energy,r,settings["position"]["gamma"]["loss_1"])
        
        energy=d["result"]["proton"]["energy"]
        r=d["result"]["proton"]["loss"]
        if len(energy)!=0:
            show(energy,r,settings["position"]["proton"]["loss"])
        r=d["result"]["proton"]["loss_0"]
        if len(energy)!=0:
            show(energy,r,settings["position"]["proton"]["loss_0"])
        r=d["result"]["proton"]["loss_1"]
        if len(energy)!=0:
            show(energy,r,settings["position"]["proton"]["loss_1"])
    
    elif train_type=="angle":
        energy=d["result"]["gamma"]["energy"]
        r=d["result"]["gamma"]["loss"]
        if len(energy)!=0:
            show(energy,r,settings["angle"]["gamma"])
        energy=d["result"]["proton"]["energy"]
        r=d["result"]["proton"]["loss"]
        if len(energy)!=0:
            show(energy,r,settings["angle"]["proton"])

    else:
        raise Exception("invalid train type")
