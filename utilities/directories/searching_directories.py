# Native Library | os
import os

def find_directory(base_directory, additional_directory_info):
    try:
        os.path.join(base_directory, additional_directory_info)
    except Exception as ERROR:
        print(f"> Error in finding directory:\n{ERROR}")
        return None
    
def does_directory_exist(path_to_directory: os.path):
    
    try:

        does_the_path_exist = os.path.exists(path_to_directory)
        return does_the_path_exist
    
    except Exception as ERROR:
        print(f"> Error in finding directory existence:\n{ERROR}")
        return False