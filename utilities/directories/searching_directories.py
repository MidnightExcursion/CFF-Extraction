# Native Library | os
import os

# Native Library | errno
import errno

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