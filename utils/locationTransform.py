import torch

#一维坐标转换为二维坐标
def location_transform(location):
    x=location//400
    y=location-400*x
    return (x,y)

# 坐标整体处理
def coordinate_transform(coordinate_list:list,total_range:int,reverse_range:int,certering:bool=False,weighted_average:bool=False,reverse_out_of_range:bool=False):
    new_coordinate_list=[]
    if not certering:
        min=int(total_range/2-reverse_range/2)
        max=min+reverse_range
        if reverse_out_of_range:
            for item in coordinate_list:
                new_coordinate_list.append((item[0]-min,item[1]-min,item[2]))
        else:
            for item in coordinate_list:
                if item[0]>=min and item[0]<max and item[1]>=min and item[1]<max:
                    new_coordinate_list.append((item[0]-min,item[1]-min,item[2]))
        return new_coordinate_list
    
    # 是否加权平均
    if weighted_average:
        x=0
        y=0
        weight=0
        for item in coordinate_list:
            x=x+item[0]*item[2]
            y=y+item[1]*item[2]
            weight=weight+item[2]
        x=x/weight
        y=y/weight
    else:
        x=0
        y=0
        weight=0
        for item in coordinate_list:
            x=x+item[0]
            y=y+item[1]
            weight=weight+1
        x=x/weight
        y=y/weight

    x_min=int(x-reverse_range/2)
    x_max=x_min+reverse_range
    y_min=int(y-reverse_range/2)
    y_max=y_min+reverse_range
    if reverse_out_of_range:
        for item in coordinate_list:
            new_coordinate_list.append((item[0]-x_min,item[1]-y_min,item[2]))
    else:
        for item in coordinate_list:
            if item[0]>=x_min and item[0]<x_max and item[1]>=y_min and item[1]<y_max:
                new_coordinate_list.append([item[0]-x_min,item[1]-y_min,item[2]])
    
    return new_coordinate_list,x,y


def coordinate_transform_angle(coordinate_dict_origin:dict,reverse_range:int,weighted_average:bool=False,use_out1_weight:bool=False):
    coordinate_dict={"1":[],"2":[],"3":[],"4":[]}
    for i in range(4):
        for item in coordinate_dict_origin[str(i+1)]:
            coordinate_dict[str(i+1)].append([int(item[0]/50),int(item[1]/50),item[2],item[3]])
            # coordinate_dict[str(i+1)].append([int(item[0]/10),int(item[1]/10),item[2],item[3]])
    
    new_coordinate_torch=torch.zeros((4,64,64,3),dtype=torch.float32)
    for ii in range(4):
        d=coordinate_dict[str(ii+1)]
        if weighted_average:
            if use_out1_weight:
                x=0
                y=0
                weight=0
                for i in range(4):
                    for item in d:
                        x=x+item[0]*item[2]
                        y=y+item[1]*item[2]
                        weight=weight+item[2]
                x=x/weight
                y=y/weight
            else:
                x=0
                y=0
                weight=0
                for i in range(4):
                    for item in d:
                        x=x+item[0]*item[3]
                        y=y+item[1]*item[3]
                        weight=weight+item[3]
                x=x/weight
                y=y/weight
        else:
            x=0
            y=0
            weight=0
            for i in range(4):
                for item in d:
                    x=x+item[0]
                    y=y+item[1]
                    weight=weight+1
            x=x/weight
            y=y/weight

        x_min=int(x-reverse_range/2)
        x_max=x_min+reverse_range
        y_min=int(y-reverse_range/2)
        y_max=y_min+reverse_range

        if ii==0:
            x_bias=50
            y_bias=-50
        elif ii==1:
            x_bias=-50
            y_bias=-50
        elif ii==2:
            x_bias=50
            y_bias=50
        elif ii==3:
            x_bias=-50
            y_bias=50
        for m in range(reverse_range):
            for n in range(reverse_range):
                new_coordinate_torch[ii][m][n][0]=(x_min+m)/100+x_bias
                new_coordinate_torch[ii][m][n][1]=(y_min+n)/100+y_bias

        for item in d:
            if x_min<=item[0]<x_max and y_min<=item[1]<y_max:
                # new_coordinate_torch[ii][item[0]-x_min][item[1]-y_min][2]=new_coordinate_torch[ii][item[0]-x_min][item[1]-y_min][2]+item[2]
                # new_coordinate_torch[ii][item[0]-x_min][item[1]-y_min][3]=new_coordinate_torch[ii][item[0]-x_min][item[1]-y_min][3]+item[3]
                new_coordinate_torch[ii][item[0]-x_min][item[1]-y_min][2]=new_coordinate_torch[ii][item[0]-x_min][item[1]-y_min][2]+item[3]

    return new_coordinate_torch