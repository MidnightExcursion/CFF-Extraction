# Native Library | os
import os

# External Library | NumPy
import numpy as np

# extractions > running_models > run_replica > run_DNN_replica
from extractions.running_models.run_replica_method import run_replica_method

from extractions.running_models.analytics import predict_cross_section, predict_cffs

# utilities > data_handling > pandas_reading > read_csv_file_with_pandas
from utilities.data_handling.pandas_reading import read_csv_file_with_pandas

# utilities > directories > find_directory
from utilities.directories.handling_directories import find_directory, create_kinematic_set_directories, find_replica_directories

# utilities > plotting > construct_cff_histogram
from utilities.plotting.plot_customizer import construct_cff_histogram

# statics > strings > column names
from statics.strings.static_strings import _COLUMN_NAME_KINEMATIC_SET

# statics > strings > directory names
from statics.strings.static_strings import _DIRECTORY_DATA

def extraction(
        kinematics_dataframe_path: str,
        kinematic_set_number: int,
        number_of_replicas: int,
        verbose: bool = False):
    
    # (1): Get the current working directory where `main.py` is running in:
    current_working_directory = os.getcwd()

    # (2): Construct the filepath to the data -- should be in `data/dataframe_path.csv`:
    possible_data_path = f"{_DIRECTORY_DATA}\\{kinematics_dataframe_path}"

    if verbose:
        print(f"> Possible path to \\data is: {possible_data_path}")

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

    # (7): Run the Replica Method. This performs the loop over N replicas:
    run_replica_method(
        kinematics_dataframe,
        fixed_kinematic_set_dataframe,
        kinematic_set_number,
        number_of_replicas,
        verbose)
    
    # (8): Obtain all the replicas:
    directory_of_replicas = find_replica_directories(kinematic_set_number)
    print(directory_of_replicas)
    print(os.listdir(directory_of_replicas))
    list_of_all_replicas = os.listdir(find_replica_directories(kinematic_set_number))
    print(list_of_all_replicas)

    list_of_all_replica_networks = [os.listdir(replica_contents) for replica_contents in list_of_all_replicas]
    print(list_of_all_replica_networks)

    predicted_cross_sections_per_replica = np.array([])

    predicted_cross_section = predict_cross_section(kinematics_dataframe, list_of_all_replica_networks)
    predicted_cffs = predict_cffs(kinematics_dataframe, list_of_all_replica_networks)

    construct_cff_histogram(predicted_cffs)

    