# Native Library | os
import os

def find_directory(base_directory, additional_directory_info):
    try:
        os.path.join(base_directory, additional_directory_info)
    except Exception as ERROR:
        return None