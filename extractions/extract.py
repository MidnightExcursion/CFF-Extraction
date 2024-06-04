# Native Library | os
import os

# extractions > running_models > run_replica > run_DNN_replica
from extractions.running_models.run_replica_method import run_replica_method

# utilities > data_handling > pandas_reading > read_csv_file_with_pandas
from utilities.data_handling.pandas_reading import read_csv_file_with_pandas

# utilities > directories > find_directory
from utilities.directories.handling_directories import find_directory, create_kinematic_set_directories

# statics > strings > column names
from statics.strings.static_strings import _COLUMN_NAME_KINEMATIC_SET

# statics > strings > directory names
from statics.strings.static_strings import _DIRECTORY_DATA

def extraction(
        kinematics_dataframe_path: str,
        kinematic_set_number: int,
        number_of_replicas: int,
        verbose: bool = False):
    
    # (1: Get the current working directory where `main.py` is running in:
    current_working_directory = os.getcwd()

    # (2): Construct the filepath to the data -- should be in `data/dataframe_path.csv`:
    possible_data_path = f"{_DIRECTORY_DATA}\\{kinematics_dataframe_path}"

    if verbose:
        print(f"> Possible path to /data is: {possible_data_path}")

    # (3): Now, check if the kinematics is actually there:
    kinematics_dataframe_file_path = find_directory(current_working_directory, possible_data_path)

    # (4): If the file was there, turn it into a DF with Pandas:
    kinematics_dataframe = read_csv_file_with_pandas(kinematics_dataframe_file_path)

    if verbose:
        print(f"> Did we convert the kinematics file to a Pandas DF? {kinematics_dataframe is not None}")

    # (5): Partition the DF on a fixed kinematic set:
    fixed_kinematic_set_dataframe = kinematics_dataframe[kinematics_dataframe[_COLUMN_NAME_KINEMATIC_SET] == kinematic_set_number]

    # (6): We creat the kinematic set directory:
    create_kinematic_set_directories(kinematic_set_number)

    # # (1.6): Run the Replica Method. This performs the loop over N replicas:
    # trained_neural_network = run_replica_method(
    #     kinematics_dataframe,
    #     fixed_kinematic_set_dataframe,
    #     number_of_replicas,
    #     verbose)