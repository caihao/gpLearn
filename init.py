import os

from utils.path import get_project_file_path

if not os.path.isdir(get_project_file_path("data")):
    os.makedirs(get_project_file_path("data"))

if not os.path.isdir(get_project_file_path("data/info")):
    os.makedirs(get_project_file_path("data/info"))
if not os.path.isdir(get_project_file_path("data/log")):
    os.makedirs(get_project_file_path("data/log"))
if not os.path.isdir(get_project_file_path("data/model")):
    os.makedirs(get_project_file_path("data/model"))
if not os.path.isdir(get_project_file_path("data/origin")):
    os.makedirs(get_project_file_path("data/origin"))
if not os.path.isdir(get_project_file_path("data/temp")):
    os.makedirs(get_project_file_path("data/temp"))

print("dirs init successfully")
