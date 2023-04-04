from bin.dataTranslater import load_from_root_to_text,load_from_text_to_json
from bin.dataInfo import get_data_info
from utils.path import get_project_file_path

filePathHead=get_project_file_path("Picture_data")
for i in [200]:
    print(i)
    
    tempPath,currentNumber=load_from_root_to_text(["gamma_20deg_700m_"+str(i)+"GeV"],filePathHead,endNum=100201)
    load_from_text_to_json("gamma_nolimit",i,tempPath)
    get_data_info("gamma_nolimit",i)

    tempPath,currentNumber=load_from_root_to_text(["CR_20deg_700m_"+str(i)+"GeV"],filePathHead,endNum=100201)
    load_from_text_to_json("proton_nolimit",i,tempPath)
    get_data_info("proton_nolimit",i)
