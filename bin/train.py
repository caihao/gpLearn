import torch
import math

from bin.test import test
from utils.tensorShuffle import tensor_shuffle
from utils.log import Log

def softmax(X):
    X_exp=torch.exp(X)
    partition=X_exp.sum(1,keepdim=True)
    return X_exp/partition

def train(dataTensor,labelTensor,batchSize:int,epoch:int,model,device,optimizer,lossFunction,train_type:str,test_list:dict=None,current_result:dict=None,log:Log=None,need_data_info:bool=False):
    _model=model
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
    if need_data_info:
        data_info_list=[]
    else:
        data_info_list=None
    for t in range(epoch):
        data_train,label_train=tensor_shuffle(train_type,dataTensor,labelTensor)
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
        if train_type=="position":
            valid_number_0=0
            valid_number_1=0
        if train_type=="particle":
            data_length=len(data_train)
        elif train_type=="energy":
            data_length=len(data_train[0])
        elif train_type=="position":
            data_length=len(data_train)
            # data_length=len(data_train[0])
        elif train_type=="angle":
            data_length=len(data_train)
        while current_index<data_length-batchSize:
            if current_index%5000==0:
                if log!=None:
                    log.write(str(current_index)+'/'+str(data_length))
                print(str(current_index)+'/'+str(data_length))
            
            current_label=label_train[current_index:current_index+batchSize].to(device)
            if train_type=="particle":
                current_data=data_train[current_index:current_index+batchSize].to(device)
                y_hat=softmax(_model(current_data))
                m=torch.argmax(y_hat,dim=1)
                s=(m==current_label).int()
                valid_number=valid_number+s.sum().item()
            elif train_type=="energy":
                current_data=[data_train[0][current_index:current_index+batchSize].to(device),data_train[1][current_index:current_index+batchSize].to(device)]
                y_hat=_model(current_data[0],current_data[1])
                # y_hat=_model(current_data[0])
            elif train_type=="position":
                current_data=data_train[current_index:current_index+batchSize].to(device)
                y_hat=_model(current_data)
                # current_data=[data_train[0][current_index:current_index+batchSize].to(device),data_train[1][current_index:current_index+batchSize].to(device)]
                # y_hat=_model(current_data[0],current_data[1])
            elif train_type=="angle":
                current_data=data_train[current_index:current_index+batchSize].to(device)
                y_hat=_model(current_data)
            
            if train_type=="particle":
                loss=lossFunction(y_hat,current_label)
            elif train_type=="energy":
                loss=lossFunction(y_hat.reshape(1),current_label)
                valid_number=valid_number+loss.sum().item()
            elif train_type=="position":
                loss=lossFunction(y_hat,current_label)
                # print(y_hat,current_label,loss)
                valid_number=valid_number+loss.sum().item()
                y_hat_0,y_hat_1=torch.split(y_hat,1,dim=1)
                current_label_0,current_label_1=torch.split(current_label,1,dim=1)
                with torch.no_grad():
                    loss_0=lossFunction(y_hat_0,current_label_0)
                    loss_1=lossFunction(y_hat_1,current_label_1)
                    valid_number_0=valid_number_0+loss_0.sum().item()
                    valid_number_1=valid_number_1+loss_1.sum().item()
                # print(loss.sum().item(),loss_0.sum().item(),loss_1.sum().item())
                # print(y_hat,current_label)
            elif train_type=="angle":
                # print(y_hat.shape,current_label.shape)
                loss=lossFunction(y_hat,current_label)
                valid_number=valid_number+loss.sum().item()
                # print(valid_number,loss.sum().item())
                # if loss.sum().item()>100:
                #     print(current_label)
                # print(y_hat,current_label)
                
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            # 进行损失以及正确率记录
            if need_data_info:
                if train_type=="particle":
                    with torch.no_grad():
                        data_info_item["train"].append({"batchSize":batchSize,"loss":loss.item(),"acc":s.sum().item()})
                elif train_type=="energy":
                    with torch.no_grad():
                        data_info_item["train"].append({"batchSize":batchSize,"loss":loss.sum().item()})
                elif train_type=="position":
                    with torch.no_grad():
                        data_info_item["train"].append({"batchSize":batchSize,"loss":loss.sum().item(),"loss_0":loss_0.sum().item(),"loss_1":loss_1.sum().item()})
                elif train_type=="angle":
                    with torch.no_grad():
                        data_info_item["train"].append({"batchSize":batchSize,"loss":loss.sum().item()})

            current_index=current_index+batchSize
        
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
            gamma_eff,gamma_total,proton_eff,proton_total=test(test_list["test_data_list"],test_list["test_label_list"],test_list["test_type_list"],test_list["test_energy_list"],_model,device,train_type,log)
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

            if q>=self_q:
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
        
        elif train_type=="energy":
            gamma_eff,gamma_total,proton_eff,proton_total=test(test_list["test_data_list"],test_list["test_label_list"],test_list["test_type_list"],test_list["test_energy_list"],_model,device,train_type,log)
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
        
        elif train_type=="position":
            gamma_eff,gamma_loss_0,gamma_loss_1,gamma_total,proton_eff,proton_loss_0,proton_loss_1,proton_total=test(test_list["test_data_list"],test_list["test_label_list"],test_list["test_type_list"],test_list["test_energy_list"],_model,device,train_type,log)
            l_g=gamma_eff/(gamma_total+0.0001)
            l_g_0=gamma_loss_0/(gamma_total+0.0001)
            l_g_1=gamma_loss_1/(gamma_total+0.0001)
            l_p=proton_eff/(proton_total+0.0001)
            l_p_0=proton_loss_0/(proton_total+0.0001)
            l_p_1=proton_loss_1/(proton_total+0.0001)
            l=(gamma_eff+proton_eff)/(gamma_total+proton_total+0.0001)
            l_0=(gamma_loss_0+proton_loss_0)/(gamma_total+proton_total+0.0001)
            l_1=(gamma_loss_1+proton_loss_1)/(gamma_total+proton_total+0.0001)

            if need_data_info:
                data_info_item["test"].append({
                    "l_g":l_g,"l_g_0":l_g_0,"l_g_1":l_g_1,
                    "l_p":l_p,"l_p_0":l_p_0,"l_p_1":l_p_1,
                    "l":l,"l_0":l_0,"l_1":l_1
                })

            res_log="l_g:%f, l_g_0:%f, l_g_1:%f, l_p:%f, l_p_0:%f, l_p_1:%f, l:%f, l_0:%f, l_1:%f" % (l_g,l_g_0,l_g_1,l_p,l_p_0,l_p_1,l,l_0,l_1)
            res_print="l_g:%f, l_g_0:%f, l_g_1:%f,\nl_p:%f, l_p_0:%f, l_p_1:%f,\nl:%f, l_0:%f, l_1:%f" % (l_g,l_g_0,l_g_1,l_p,l_p_0,l_p_1,l,l_0,l_1)
            log.write("epoch "+str(t+1)+" finish")
            log.write(res_log)
            print("epoch "+str(t+1)+" finish")
            print(res_print)

            if l<=self_l or self_l==0:
                torch.save({'model':_model,'l':l,'l_0':l_0,'l_1':l_1},current_result["model_file"])
                self_l=l
                self_l_0=l_0
                self_l_1=l_1
                model=_model
                log.write("model update")
                print("model update")
            else:
                _model=torch.load(current_result["model_file"])["model"]
                log.write("model continue...")
                print("model continue...")

        elif train_type=="angle":
            gamma_eff,gamma_total,proton_eff,proton_total=test(test_list["test_data_list"],test_list["test_label_list"],test_list["test_type_list"],test_list["test_energy_list"],_model,device,train_type,log)
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

    if train_type=="particle":
        return _model,{"acc":self_acc,"q":self_q},data_info_list
    elif train_type=="energy":
        return _model,{"l":self_l},data_info_list
    elif train_type=="position":
        return _model,{"l":self_l,"l_0":self_l_0,"l_1":self_l_1},data_info_list
    elif train_type=="angle":
        return _model,{"l":self_l},data_info_list
    else:
        raise Exception("invalid train type")
