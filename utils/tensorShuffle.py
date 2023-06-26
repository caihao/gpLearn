import torch
import random

def tensor_shuffle(train_type:str,data,label:torch.tensor=None):
    if train_type=="particle":
        order=[i for i in range(len(data))]
        random.shuffle(order)
        data_list=data.numpy().tolist()
        data_dtype=data.dtype
        data_list_new=[]
        if label!=None:
            assert len(data)==len(label)
            label_list=label.numpy().tolist()
            label_dtype=label.dtype
            label_list_new=[]
            for i in order:
                data_list_new.append(data_list[i])
                label_list_new.append(label_list[i])
        else:
            for i in order:
                data_list_new.append(data_list[i])
        return torch.tensor(data_list_new,dtype=data_dtype),torch.tensor(label_list_new,dtype=label_dtype)

    elif train_type=="energy":
        order=[i for i in range(len(data[0]))]
        assert len(data[0])==len(data[1])
        random.shuffle(order)
        data_list=[data[0].numpy().tolist(),data[1].numpy().tolist()]
        data_dtype=[data[0].dtype,data[1].dtype]
        data_list_new=[[],[]]
        if label!=None:
            assert len(label)==len(data[0])
            label_list=label.numpy().tolist()
            label_dtype=label.dtype
            label_list_new=[]
            for i in order:
                data_list_new[0].append(data_list[0][i])
                data_list_new[1].append(data_list[1][i])
                label_list_new.append(label_list[i])
        else:
            for i in order:
                data_list_new[0].append(data_list[0][i])
                data_list_new[1].append(data_list[1][i])
        return [torch.tensor(data_list_new[0],dtype=data_dtype[0]),torch.tensor(data_list_new[1],dtype=data_dtype[1])],torch.tensor(label_list_new,dtype=label_dtype)
    
    elif train_type=="position":
        order=[i for i in range(len(data))]
        random.shuffle(order)
        data_list=data.numpy().tolist()
        data_dtype=data.dtype
        data_list_new=[]
        if label!=None:
            assert len(data)==len(label)
            label_list=label.numpy().tolist()
            label_dtype=label.dtype
            label_list_new=[]
            for i in order:
                data_list_new.append(data_list[i])
                label_list_new.append(label_list[i])
        else:
            for i in order:
                data_list_new.append(data_list[i])
        return torch.tensor(data_list_new,dtype=data_dtype),torch.tensor(label_list_new,dtype=label_dtype)
    
    elif train_type=="angle":
        order=[i for i in range(len(data))]
        random.shuffle(order)
        data_list=data.numpy().tolist()
        data_dtype=data.dtype
        data_list_new=[]
        if label!=None:
            assert len(data)==len(label)
            label_list=label.numpy().tolist()
            label_dtype=label.dtype
            label_list_new=[]
            for i in order:
                data_list_new.append(data_list[i])
                label_list_new.append(label_list[i])
        else:
            for i in order:
                data_list_new.append(data_list[i])
        return torch.tensor(data_list_new,dtype=data_dtype),torch.tensor(label_list_new,dtype=label_dtype)

    else:
        raise Exception("invalid train type")

