# Native Library | os
import os

# Native Library | errno
import errno

# statics > strings > directory names
from statics.strings.static_strings import _DIRECTORY_DATA
from statics.strings.static_strings import _DIRECTORY_EXTRACTIONS_MODELS_
from statics.strings.static_strings import _DIRECTORY_EXTRACTIONS_MODELS_KINEMATIC_SETS

def find_directory(base_directory: str, additional_directory_info: str) -> os.path:
    """
    # Description:

    Look for a directory by stitching together a base_directory
    (usually the pwd) and the "hypothesized" path where the file
    you're looking for lives.

    # Arguments:

    base_directory: str
    additional_directory_info: str
    
    # Returns

    directory_path: bool | False
    """
    try:
        directory_path = os.path.join(base_directory, additional_directory_info)
        return directory_path
    except Exception as ERROR:
        print(f"> Error in finding directory:\n{ERROR}")
        
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), f"{base_directory}/{additional_directory_info}")
    
def does_directory_exist(path_to_directory: os.path) -> bool:
    """
    # Description:

    Here, we just look for a given os.path within a given
    directory context.

    # Arguments:

    path_to_directory: os.path

    # Returns:

    does_the_path_exist: bool | False
    """

    try:

        does_the_path_exist = os.path.exists(path_to_directory)
        return does_the_path_exist
    
    except Exception as ERROR:
        print(f"> Error in finding directory existence:\n{ERROR}")
        return False
    
def create_directory(filepath_to_directory):
    """
    """

    if does_directory_exist(filepath_to_directory):

        return False
    
    else:

        try:

            did_os_make_the_directory = os.mkdir(filepath_to_directory)
            return did_os_make_the_directory
        
        except Exception as ERROR:
            print(f"> Error in creating the directory:\n{ERROR}")
            return False

def create_kinematic_set_directories(kinematic_set_number):

    # (1): Get the current working directory where `main.py` is running in:
    current_working_directory = os.getcwd()

    # (2): Construct the relevant filepath: `data/models/kinematic_sets/`
    kinematic_sets_filepath = f"{current_working_directory}//{_DIRECTORY_DATA}//{_DIRECTORY_EXTRACTIONS_MODELS_}//{_DIRECTORY_EXTRACTIONS_MODELS_KINEMATIC_SETS}"

    # (3): We create the filepath `root/data/models/kinematic_sets/set_number`
    did_we_create_kinematic_set_directory = create_directory(f"{kinematic_sets_filepath}//kinematic_set_{kinematic_set_number}")

    return did_we_create_kinematic_set_directory