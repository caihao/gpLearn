import os

def get_project_path():
    """得到项目路径"""
    project_path = os.path.join(
        os.path.dirname(__file__),
        "..",
    )
    return project_path

def get_project_file_path(relative_file_path:str):
    return os.path.join(get_project_path(),relative_file_path)
