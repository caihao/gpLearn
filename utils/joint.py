import numpy as np

def weighted_line_fit(points):
    # 提取x坐标、y坐标和权值
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    weights = [p[2] for p in points]

    # 计算加权拟合直线的斜率和截距
    k, b = np.polyfit(x, y, 1, w=weights)

    return k, b

def weighted_line_fit_need_center(points):
    # 提取x坐标、y坐标和权值
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    weights = [p[2] for p in points]

    # 计算加权拟合直线的斜率和截距
    try:
        k, b = np.polyfit(x, y, 1, w=weights)
    except:
        return True,0,0,0,0

    x_value=0
    y_value=0
    w_value=0
    for item in points:
        x_value=x_value+item[0]*item[2]
        y_value=y_value+item[1]*item[2]
        w_value=w_value+item[2]

    return False, k, b, x_value/w_value, y_value/w_value

def coor_trans(points:list,position:int,max_ran:int=400,min_ran:int=64,lower:bool=False,t:bool=False,spin:int=0):
    if lower==True:
        lower_points=[[max_ran-item[0]+1,item[1],item[2]] for item in points]
    else:
        lower_points=points
    if t==True:
        t_points=[[item[1],item[0],item[2]] for item in lower_points]
    else:
        t_points=lower_points
    spin_points=t_points
    for _ in range(spin):
        spin_points=[[item[1],max_ran-item[0]+1,item[2]] for item in spin_points]
    
    x=[item[0] for item in spin_points]
    y=[item[1] for item in spin_points]
    w=[item[2] for item in spin_points]

    # x_value=0
    # y_value=0
    w_value=0
    for i in range(len(x)):
        # x_value=x_value+x[i]*w[i]
        # y_value=y_value+y[i]*w[i]
        w_value=w_value+w[i]
    # x_center=x_value/w_value
    # y_center=y_value/w_value
    # x_zeros=int(x_center-min_ran/2)
    # y_zeros=int(y_center-min_ran/2)
    # min_points=[[item[0]-x_zeros,item[1]-y_zeros,item[2]] for item in spin_points]
    # min_k,min_b=weighted_line_fit(min_points)

    if position==0:
        fin_points=[[(item[0]-200)*0.05,(item[1]-200)*0.05+100,item[2]] for item in spin_points]
    elif position==1:
        fin_points=[[(item[0]-200)*0.05+100,(item[1]-200)*0.05+100,item[2]] for item in spin_points]
    elif position==2:
        fin_points=[[(item[0]-200)*0.05,(item[1]-200)*0.05,item[2]] for item in spin_points]
    elif position==3:
        fin_points=[[(item[0]-200)*0.05+100,(item[1]-200)*0.05,item[2]] for item in spin_points]
    else:
        raise Exception("invalid position")

    return fin_points,w_value

def calculate_intersection(k1, b1, k2, b2):
    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1
    return x, y

def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def joint_single(json_data:dict,index:int,mode:int):
    if mode==0:
        fin_points_0,w_0=coor_trans(json_data["c4"][index]['1'],0,lower=False,t=False,spin=1)
        fin_points_1,w_1=coor_trans(json_data["c4"][index]['2'],1,lower=False,t=False,spin=0)
        fin_points_2,w_2=coor_trans(json_data["c4"][index]['3'],2,lower=False,t=True,spin=0)
        fin_points_3,w_3=coor_trans(json_data["c4"][index]['4'],3,lower=False,t=True,spin=1)
    elif mode==1:
        fin_points_0,w_0=coor_trans(json_data["c4"][index]['1'],0,lower=False,t=True,spin=1)
        fin_points_1,w_1=coor_trans(json_data["c4"][index]['2'],1,lower=False,t=True,spin=0)
        fin_points_2,w_2=coor_trans(json_data["c4"][index]['3'],2,lower=False,t=False,spin=0)
        fin_points_3,w_3=coor_trans(json_data["c4"][index]['4'],3,lower=False,t=False,spin=1)
    elif mode==2:
        fin_points_0,w_0=coor_trans(json_data["c4"][index]['1'],0,lower=False,t=False,spin=1)
        fin_points_1,w_1=coor_trans(json_data["c4"][index]['2'],1,lower=False,t=True,spin=0)
        fin_points_2,w_2=coor_trans(json_data["c4"][index]['3'],2,lower=False,t=False,spin=0)
        fin_points_3,w_3=coor_trans(json_data["c4"][index]['4'],3,lower=False,t=True,spin=1)
    elif mode==3:
        fin_points_0,w_0=coor_trans(json_data["c4"][index]['1'],0,lower=False,t=True,spin=1)
        fin_points_1,w_1=coor_trans(json_data["c4"][index]['2'],1,lower=False,t=False,spin=0)
        fin_points_2,w_2=coor_trans(json_data["c4"][index]['3'],2,lower=False,t=True,spin=0)
        fin_points_3,w_3=coor_trans(json_data["c4"][index]['4'],3,lower=False,t=False,spin=1)
    else:
        raise Exception("invalid mode")
    
    invalid_0,k_0,b_0,x_0,y_0=weighted_line_fit_need_center(fin_points_0)
    invalid_1,k_1,b_1,x_1,y_1=weighted_line_fit_need_center(fin_points_1)
    invalid_2,k_2,b_2,x_2,y_2=weighted_line_fit_need_center(fin_points_2)
    invalid_3,k_3,b_3,x_3,y_3=weighted_line_fit_need_center(fin_points_3)
    if invalid_0 or invalid_1 or invalid_2 or invalid_3:
        return False,0,0,0,[],[]

    m_0,n_0,p_0=*calculate_intersection(k_0,b_0,k_1,b_1),w_0+w_1
    m_1,n_1,p_1=*calculate_intersection(k_0,b_0,k_2,b_2),w_0+w_2
    m_2,n_2,p_2=*calculate_intersection(k_0,b_0,k_3,b_3),w_0+w_3
    m_3,n_3,p_3=*calculate_intersection(k_1,b_1,k_2,b_2),w_1+w_2
    m_4,n_4,p_4=*calculate_intersection(k_1,b_1,k_3,b_3),w_1+w_3
    m_5,n_5,p_5=*calculate_intersection(k_2,b_2,k_3,b_3),w_2+w_3

    sec_points=[[m_0,n_0,p_0],[m_1,n_1,p_1],[m_2,n_2,p_2],[m_3,n_3,p_3],[m_4,n_4,p_4],[m_5,n_5,p_5]]

    sec_x_value=0
    sec_y_value=0
    sec_w_value=0
    for item in sec_points:
        if 0<=item[0]<=100 and 0<item[1]<=100:
            sec_x_value=sec_x_value+item[0]*item[2]
            sec_y_value=sec_y_value+item[1]*item[2]
            sec_w_value=sec_w_value+item[2]
    try:
        sec_x=sec_x_value/sec_w_value
        sec_y=sec_y_value/sec_w_value
    except ZeroDivisionError:
        return False,0,0,0,[],[]

    sec_l=0
    l_0=distance(sec_x,sec_y,x_0,y_0)
    l_1=distance(sec_x,sec_y,x_1,y_1)
    l_2=distance(sec_x,sec_y,x_2,y_2)
    l_3=distance(sec_x,sec_y,x_3,y_3)
    sec_l=sec_l+l_0*w_0
    sec_l=sec_l+l_1*w_1
    sec_l=sec_l+l_2*w_2
    sec_l=sec_l+l_3*w_3
    
    return True,sec_x,sec_y,sec_l,[l_0,l_1,l_2,l_3],[[x_0,y_0,w_0],[x_1,y_1,w_1],[x_2,y_2,w_2],[x_3,y_3,w_3]]

def joint(json_data:dict,index:int):
    legal=False
    sec_x=None
    sec_y=None
    sec_l=None
    l_list=None
    point_list=None
    for i in range(4):
        legal_new,sec_x_new,sec_y_new,sec_l_new,l_list_new,point_list_new=joint_single(json_data,index,i)
        if legal_new==True:
            if sec_x==None:
                legal=True
                sec_x,sec_y,sec_l,l_list,point_list=sec_x_new,sec_y_new,sec_l_new,l_list_new,point_list_new
            else:
                if sec_l_new<sec_l:
                    legal=True
                    sec_x,sec_y,sec_l,l_list,point_list=sec_x_new,sec_y_new,sec_l_new,l_list_new,point_list_new

    return legal,sec_x,sec_y,sec_l,l_list,point_list


# import json
# with open("data/origin/gamma_nolimit_200.json",'r') as f:
#     json_data=json.load(f)
#     f.close()
# print(joint(json_data,11848))