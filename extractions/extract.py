# Native Library | os
import os

# External Library | NumPy
import numpy as np

# External Library | TensorFlow
import tensorflow as tf

# extractions > running_models > run_replica > run_DNN_replica
from extractions.running_models.run_replica_method import run_replica_method

from extractions.running_models.analytics import predict_cross_section, predict_cffs, construct_cff_histogram, construct_cross_section_plot

# utilities > data_handling > pandas_reading > read_csv_file_with_pandas
from utilities.data_handling.pandas_reading import read_csv_file_with_pandas

# utilities > directories > find_directory
from utilities.directories.handling_directories import find_directory
from utilities.directories.handling_directories import create_kinematic_set_directories
from utilities.directories.handling_directories import find_replica_directories
from utilities.directories.handling_directories import find_replica_model_directories
from utilities.directories.handling_directories import find_all_model_paths

# statics > strings > column names
from statics.strings.static_strings import _COLUMN_NAME_KINEMATIC_SET, _COLUMN_NAME_AZIMUTHAL_PHI

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

    print('fuk')

    # list_of_paths_to_replicas = find_all_model_paths(kinematic_set_number)
    list_of_directories_of_replicas = os.listdir(find_replica_directories(kinematic_set_number))
    print('fuk')
    list_of_all_replica_models = [os.listdir(find_replica_model_directories(kinematic_set_number, replica_index + 1))[0] for replica_index, _ in enumerate(list_of_directories_of_replicas)]
    print('fuk')
    paths_to_all_replica_models = [find_replica_model_directories(kinematic_set_number, replica_index + 1) for replica_index, _ in enumerate(list_of_directories_of_replicas)]

    print('fuk')
    cross_section_layers = [tf.keras.models.load_model(f"{filepath}/replica_{replica_index + 1}.keras", safe_mode = False) for replica_index, filepath in enumerate(paths_to_all_replica_models)]
    print(cross_section_layers)

    print('fuk')
    # predicted_cross_section = [predict_cross_section(kinematics_dataframe, tensorflow_model) for tensorflow_model in cross_section_layers]
    # predicted_cffs = [predict_cffs(kinematics_dataframe, tensorflow_model) for tensorflow_model in cross_section_layers]

    predicted_cross_section = []
    predicted_cffs = []

    print('fuk')
    for tensorflow_model in cross_section_layers:
        predicted_cffs.append(predict_cffs(kinematics_dataframe, tensorflow_model))
        predicted_cross_section.append(predict_cffs(kinematics_dataframe, tensorflow_model))

    cff_histogram_1 = construct_cff_histogram(
        predicted_cffs[0],
        plot_title = f"Kinematic Set {kinematic_set_number} | CFF1, Histogram (from Local)\nMean: {np.mean(predicted_cffs)}, Std Dev: {np.std(predicted_cffs)}')",
        x_label = 'F',
        y_label = 'Frequency')
    
    cff_histogram_1.savefig('cff1.png')
 
    cff_histogram_2 = construct_cff_histogram(
        predicted_cffs[1],
        plot_title = f"Kinematic Set {kinematic_set_number} | CFF2, Histogram (from Local)\nMean: {np.mean(predicted_cffs)}, Std Dev: {np.std(predicted_cffs)}')",
        x_label = 'F',
        y_label = 'Frequency')
    
    cff_histogram_2.savefig('cff2.png')

    cff_histogram_3 = construct_cff_histogram(
        predicted_cffs[2],
        plot_title = f"Kinematic Set {kinematic_set_number} | CFF3, Histogram (from Local)\nMean: {np.mean(predicted_cffs)}, Std Dev: {np.std(predicted_cffs)}')",
        x_label = 'F',
        y_label = 'Frequency')
    
    cff_histogram_3.savefig('cff3.png')

    cff_histogram_4 = construct_cff_histogram(
        predicted_cffs[3],
        plot_title = f"Kinematic Set {kinematic_set_number} | CFF3, Histogram (from Local)\nMean: {np.mean(predicted_cffs)}, Std Dev: {np.std(predicted_cffs)}')",
        x_label = 'F',
        y_label = 'Frequency')
    
    cff_histogram_4.savefig('cff4.png')

    cross_section_plot = construct_cross_section_plot(
        predicted_cross_section,
        fixed_kinematic_set_dataframe[_COLUMN_NAME_AZIMUTHAL_PHI],
        plot_title = f"Kinematic Set {kinematic_set_number} | CFF3, Histogram (from Local)\nMean: {np.mean(predicted_cffs)}, Std Dev: {np.std(predicted_cffs)}')",
        x_label = '\mathcal\{F\}',
        y_label = 'Frequency')
    
    cross_section_plot.savefig('crosssection.png')