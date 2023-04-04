import json

from utils.path import get_project_file_path

def get_data_info(particle:str,energy:int):
    jsonFileName=get_project_file_path("data/origin/"+particle+"_"+str(energy)+".json")
    with open(jsonFileName,'r') as f:
        jsonData=json.load(f)
        f.close()
    length_1=len(jsonData["c1"])
    length_2=len(jsonData["c2"])
    length_3=len(jsonData["c3"])
    length_4=len(jsonData["c4"])
    print(particle+" with energy "+str(energy)+" has total number:")
    print("pic_1:"+str(length_1)+", pic_2:"+str(length_2)+", pic_3:"+str(length_3)+", pic_4:"+str(length_4))

def get_data_number(particle:str,energy:int,allow_pic_number:int,allow_min_pix_number:int):
    jsonFileName=get_project_file_path("data/origin/"+particle+"_"+str(energy)+".json")
    with open(jsonFileName,'r') as f:
        jsonData=json.load(f)
        f.close()
    specfic_jsonData=jsonData["c"+str(allow_pic_number)]
    number=0
    for pic_item in specfic_jsonData:
        pic_number=len(pic_item["1"])+len(pic_item["2"])+len(pic_item["3"])+len(pic_item["4"])
        if pic_number>=allow_min_pix_number:
            number=number+1
    print(particle+" with energy:"+str(energy)+", pic_number:"+str(allow_pic_number)+", min_pix_number:"+str(allow_min_pix_number)+" has total number "+str(number))

