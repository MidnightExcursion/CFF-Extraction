
# Native Library | os
import os

# Native Library | datetime
import datetime

# External Library | Pandas
import numpy as np

# External Library | Pandas
from pandas import DataFrame

# Native Library | TensorFlow
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from utilities.data_handling.split_ann_data import split_data
from utilities.generation.generate_pseudodata import generate_replica_data

from utilities.directories.handling_directories import create_replica_directories
from utilities.directories.handling_directories import create_replica_model_directories
from utilities.directories.handling_directories import create_replica_plots_directories

from extractions.running_models.analytics import perform_replica_analytics, construct_plot_data
from extractions.running_models.run_replica import run_DNN_replica
from extractions.running_models.obtain_replica_results import obtain_replica_results

from statics.strings.static_strings import _DIRECTORY_EXTRACTIONS, _DIRECTORY_EXTRACTIONS_MODELS_, _DIRECTORY_EXTRACTIONS_MODELS_KINEMATIC_SETS

from statics.strings.static_strings import _COLUMN_NAME_X_BJORKEN
from statics.strings.static_strings import _COLUMN_NAME_Q_SQUARED
from statics.strings.static_strings import _COLUMN_NAME_T_MOMENTUM_CHANGE
from statics.strings.static_strings import _COLUMN_NAME_AZIMUTHAL_PHI
from statics.strings.static_strings import _COLUMN_NAME_LEPTON_MOMENTUM 
from statics.strings.static_strings import _COLUMN_NAME_CROSS_SECTION
from statics.strings.static_strings import _COLUMN_NAME_CROSS_SECTION_ERROR

from statics.strings.static_strings import _HYPERPARAMETER_NUMBER_OF_EPOCHS
from statics.strings.static_strings import _HYPERPARAMETER_BATCH_SIZE
from statics.strings.static_strings import _HYPERPARAMETER_LR_PATIENCE
from statics.strings.static_strings import _HYPERPARAMETER_LR_FACTOR
from statics.strings.static_strings import _HYPERPARAMETER_EARLYSTOP_PATIENCE_INTEGER
from statics.strings.static_strings import _DNN_VERBOSE_SETTING

# DataFrame Columns
from statics.strings.static_strings import _COLUMN_NAME_KINEMATIC_SET

def run_replica_method(
        entire_kinematic_dataframe: DataFrame,
        kinematic_set_dataframe: DataFrame,
        kinematic_set_integer: int,
        number_of_replicas: int = 1,
        verbose: bool = False):
    """
    # Description:
    
    Run N = `number_of_replicas` over a specified, fixed kinematic range.

    # Parameters:

        kinematic_set_dataframe:

        number_of_replicas: 

        verbose:
    """

    # (1): Begin iterating over the replicas:
    for replica_index in range(number_of_replicas):

        # (1.1): Obtain the replica number by adding 1 to the index:
        replica_number = replica_index + 1

        # (1.2): Propose a replica name:
        current_replica_name = f"replica_{replica_number}"

        # (1.3): Immediately construct the filetype for the replica:
        model_file_name = f"{current_replica_name}.h5"

        # (1.4): Create the directory for the replica:
        did_we_create_replica_directory = create_replica_directories(kinematic_set_integer, replica_number)
        did_we_create_replica_model_directory = create_replica_model_directories(kinematic_set_integer, replica_number)
        did_we_create_replica_plots_directory = create_replica_plots_directories(kinematic_set_integer, replica_number)

        # (2): Begin timing the replica time:
        start_time_in_milliseconds = datetime.datetime.now().replace(microsecond = 0)
        
        if verbose:
            print(f"> Replica #{replica_index + 1} now running...")

        # (3): Generate the data for this DNN replica:
        generated_replica_data = generate_replica_data(kinematic_set_dataframe)

        # (4): Identify the "x values" for our model:
        x_data = generated_replica_data[[_COLUMN_NAME_Q_SQUARED, _COLUMN_NAME_X_BJORKEN, _COLUMN_NAME_T_MOMENTUM_CHANGE, _COLUMN_NAME_AZIMUTHAL_PHI, _COLUMN_NAME_LEPTON_MOMENTUM]]

        # (5): Identify the "y values" for our model:
        y_data = generated_replica_data[_COLUMN_NAME_CROSS_SECTION]
        y_data_error = generated_replica_data[_COLUMN_NAME_CROSS_SECTION_ERROR]
        
        # (6): Split the Data into Training/Testing:
        training_x_data, training_y_data, training_y_error_data, testing_x_data, testing_y_data, testing_y_error_data = split_data(x_data, y_data, y_data_error)

        # (7): Actually run the Replica:
        neural_network, neural_network_history = run_DNN_replica(training_x_data, training_y_data, testing_x_data, testing_y_data)

        directory_for_replica_model = 

        # (9): Hopefully (8) worked, then just save the data there:
        neural_network.save(
            directory_for_replica_model,
            save_format = 'h5'
        )

        print(f"> Saved replica!" )

        if verbose:
            print(f"> Replica #{replica_index + 1} finished running...")
    
        end_time_in_milliseconds = datetime.datetime.now().replace(microsecond = 0)

        print(f"> Replica job finished in {end_time_in_milliseconds - start_time_in_milliseconds}ms.")

        perform_replica_analytics(kinematic_set_dataframe, neural_network)

        print(neural_network_history.history['loss'])

        construct_plot_data(
            x_data = np.linspace(0, len(neural_network_history.history['loss']), _HYPERPARAMETER_NUMBER_OF_EPOCHS),
            y_data = neural_network_history.history['loss'],
            plot_title = "Loss",
            x_label = "Epoch",
            y_label = "Loss")

        obtain_replica_results(entire_kinematic_dataframe, neural_network)