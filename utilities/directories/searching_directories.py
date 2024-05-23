# Native Library | os
import os

def find_directory(base_directory: str, additional_directory_info: str):
    """
    # Description:

    Look for a directory by stitching together a base_directory
    (usually the pwd) and the "hypothesized" path where the file
    you're looking for lives.

    # Arguments:

    base_directory: str
    additional_directory_info: str
    """
    try:
        directory_path = os.path.join(base_directory, additional_directory_info)
        return directory_path
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