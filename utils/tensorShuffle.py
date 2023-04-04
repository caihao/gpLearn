import torch
import random

def tensor_shuffle(data:torch.tensor,label:torch.tensor=None):
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
        

