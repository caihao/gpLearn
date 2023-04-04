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
    
    if False:
        total=0
        for item in new_coordinate_list:
            total=total+item[2]
        for i in range(len(new_coordinate_list)):
            new_coordinate_list[i][2]=new_coordinate_list[i][2]/total
    return new_coordinate_list
    
