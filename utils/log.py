import time
import datetime
import os

from utils.path import get_project_file_path

class Log(object):
    def __init__(self,relative_current_file_name:str,current_program_version:str="long-term-support"):
        self.timeStamp=str(int(time.time()))
        self.log_file_name=get_project_file_path("data/log/"+self.timeStamp+".txt")
        with open(self.log_file_name,'w') as f:
            f=open(self.log_file_name,'w')
            f.write("# GP project standard log file\n\n")
            f.write("# process start on "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'\n')
            f.write("# program version: "+current_program_version+"\n")
            f.write("# run file name: "+relative_current_file_name+'\n')
            f.write("\n# file content:\n")
            f.write("###file content start###\n")
            current_file_name=get_project_file_path(relative_current_file_name)
            if not os.path.exists(current_file_name):
                raise Exception("file not exist")
            with open(current_file_name,'r') as fi:
                f.write(fi.read())
                fi.close()
            f.write("###file content end###\n\n")
            f.close()

    def write(self,t:str):
        with open(self.log_file_name,'a') as f:
            f.write("["+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+"] "+t+'\n')
            f.close()

    def info(self,key,value):
        with open(self.log_file_name,'a') as f:
            f.write("[INFO] "+str(key)+" "+str(value)+'\n')
            f.close()

    def method(self,t:str):
        with open(self.log_file_name,'a') as f:
            f.write('\n'+"[METHOD] "+t+'\n')
            f.close()

    def error(self,t:str):
        with open(self.log_file_name,'a') as f:
            f.write("[ERROR] "+t+'\n')
            f.close()
        self.close()

    def close(self):
        with open(self.log_file_name,'a') as f:
            f.write("["+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+"] process run finish\n")
            f.close()
