from bin.dataTranslater import load_from_root_to_text,load_from_text_to_json
from bin.dataInfo import get_data_info
from utils.path import get_project_file_path

filePathHead=get_project_file_path("Picture_data")
# for i in [200]:
#     print(i)
    
#     tempPath,currentNumber=load_from_root_to_text(["gamma_20deg_700m_"+str(i)+"GeV"],filePathHead,endNum=100501)
#     load_from_text_to_json("gamma_nolimit",i,tempPath)
#     get_data_info("gamma_nolimit",i)

    # tempPath,currentNumber=load_from_root_to_text(["CR_20deg_700m_"+str(i)+"GeV"],filePathHead,endNum=100201)
    # load_from_text_to_json("proton_nolimit",i,tempPath)
    # get_data_info("proton_nolimit",i)




import json
import numpy as np

from utils.joint import joint
from utils.path import get_project_file_path

for eng in [200]:
    print("proton",eng)
    with open(get_project_file_path("data/origin/gamma_"+str(eng)+".json"),'r') as f:
        json_data=json.load(f)
        f.close()
    for i in range(len(json_data["c4"])):
        # print("##########",i)
        legal,sec_x,sec_y,sec_l,l_list,point_list=joint(json_data,i)
        if legal==True:
            json_data["c4"][i]["info"]={
                "legal":True,
                "center":{"x":sec_x,"y":sec_y,"l_w":sec_l},
                "l":{"1":l_list[0],"2":l_list[1],"3":l_list[2],"4":l_list[3]},
                "single":{
                    "1":{"x":point_list[0][0],"y":point_list[0][1],"w":point_list[0][2]},
                    "2":{"x":point_list[1][0],"y":point_list[1][1],"w":point_list[1][2]},
                    "3":{"x":point_list[2][0],"y":point_list[2][1],"w":point_list[2][2]},
                    "4":{"x":point_list[3][0],"y":point_list[3][1],"w":point_list[3][2]},
                }
            }
        else:
            json_data["c4"][i]["info"]={
                "legal":False
            }
    with open(get_project_file_path("data/origin/gamma_"+str(eng)+".json"),'w') as f:
        json.dump(json_data,f)
        f.close()
