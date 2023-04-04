import torch
import math

from bin.test import test
from utils.tensorShuffle import tensor_shuffle
from utils.log import Log

def softmax(X):
    X_exp=torch.exp(X)
    partition=X_exp.sum(1,keepdim=True)
    return X_exp/partition

def train(dataTensor,labelTensor,batchSize:int,epoch:int,model,device,optimizer,lossFunction,useSoftmax:bool=True,test_list:dict=None,current_result:dict=None,log:Log=None,need_data_info:bool=False):
    _model=model
    if useSoftmax:
        self_acc=current_result["acc"]
        self_q=current_result["q"]
    else:
        self_l=current_result["l"]
    if need_data_info:
        data_info_list=[]
    else:
        data_info_list=None
    for t in range(epoch):
        data_train,label_train=tensor_shuffle(dataTensor,labelTensor)
        if need_data_info:
            data_info_item={
                "train":[],
                "test":[]
            }
        if log!=None:
            log.write('training on epoch '+str(t+1)+'/'+str(epoch))
        print('training on epoch '+str(t+1)+'/'+str(epoch))
        current_index=0
        valid_number=0
        while current_index<len(data_train)-batchSize:
            if current_index%5000==0:
                if log!=None:
                    log.write(str(current_index)+'/'+str(len(data_train)))
                print(str(current_index)+'/'+str(len(data_train)))
            current_data=data_train[current_index:current_index+batchSize].to(device)
            current_label=label_train[current_index:current_index+batchSize].to(device)
            if useSoftmax:
                # print(current_data.shape)
                y_hat=softmax(_model(current_data))
                # print(y_hat.shape)
                m=torch.argmax(y_hat,dim=1)
                # print(m.shape)
                s=(m==current_label).int()
                valid_number=valid_number+s.sum().item()
            else:
                y_hat=_model(current_data)
            if useSoftmax:
                loss=lossFunction(y_hat,current_label)
            else:
                loss=lossFunction(y_hat.reshape(1),current_label)
            if not useSoftmax:
                valid_number=valid_number+loss.sum().item()
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            # 进行损失以及正确率记录
            if need_data_info:
                if useSoftmax:
                    with torch.no_grad():
                        data_info_item["train"].append({"batchSize":batchSize,"loss":loss.item(),"acc":s.sum().item()})
                else:
                    with torch.no_grad():
                        data_info_item["train"].append({"batchSize":batchSize,"loss":loss.sum().item()})

            current_index=current_index+batchSize
        
        if useSoftmax:
            if log!=None:
                log.write("training accuracy: "+str(valid_number/current_index))
            print("training accuracy: "+str(valid_number/current_index))
        else:
            if log!=None:
                log.write("training loss: "+str(valid_number/current_index))
            print("training loss: "+str(valid_number/current_index))
    
        if useSoftmax:
            gamma_eff,gamma_total,proton_eff,proton_total=test(test_list["test_data_list"],test_list["test_label_list"],test_list["test_type_list"],test_list["test_energy_list"],_model,device,True,log)
            acc_g=gamma_eff/gamma_total
            acc_p=proton_eff/proton_total
            acc=(gamma_eff+proton_eff)/(gamma_total+proton_total)
            try:
                q=acc_g/math.sqrt(1-acc_p)
            except:
                q=acc_g/math.sqrt(1-acc_p+0.00001)
            
            if need_data_info:
                data_info_item["test"].append({"acc_g":acc_g,"acc_p":acc_p,"acc":acc,"q":q})

            res="acc_g:%f, acc_p:%f, acc:%f, q:%f" % (acc_g,acc_p,acc,q)
            log.write("epoch "+str(t+1)+" finish")
            log.write(res)
            print("epoch "+str(t+1)+" finish")
            print(res)

            if acc>=self_acc and q>=self_q:
                torch.save({'model':_model,'acc':acc,'q':q},current_result["model_file"])
                self_acc=acc
                self_q=q
                model=_model
                log.write("model update")
                print("model update")
            else:
                _model=torch.load(current_result["model_file"])["model"]
                log.write("model continue...")
                print("model continue...")
        
        else:
            gamma_eff,gamma_total,proton_eff,proton_total=test(test_list["test_data_list"],test_list["test_label_list"],test_list["test_type_list"],test_list["test_energy_list"],_model,device,False,log)
            l_g=gamma_eff/(gamma_total+0.0001)
            l_p=proton_eff/(proton_total+0.0001)
            l=(gamma_eff+proton_eff)/(gamma_total+proton_total)

            if need_data_info:
                data_info_item["test"].append({"l_g":l_g,"l_p":l_p,"l":l})

            res="l_g:%f, l_p:%f, l:%f" % (l_g,l_p,l)
            log.write("epoch "+str(t+1)+" finish")
            log.write(res)
            print("epoch "+str(t+1)+" finish")
            print(res)

            if l<=self_l or self_l==0:
                torch.save({'model':_model,'l':l},current_result["model_file"])
                self_l=l
                model=_model
                log.write("model update")
                print("model update")
            else:
                _model=torch.load(current_result["model_file"])["model"]
                log.write("model continue...")
                print("model continue...")
        
        if need_data_info:
            data_info_list.append(data_info_item)

    if useSoftmax:
        return _model,{"acc":self_acc,"q":self_q},data_info_list
    else:
        return _model,{"l":self_l},data_info_list
