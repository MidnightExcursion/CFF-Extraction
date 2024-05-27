
# Native Library | os
import os

# Native Library | datetime
import datetime

# External Library | Pandas
from pandas import DataFrame

# Native Library | TensorFlow
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from utilities.data_handling.split_ann_data import split_data
from utilities.generation.generate_pseudodata import generate_replica_data

from utilities.directories.searching_directories import find_directory

from extractions.running_models.analytics import perform_replica_analytics

from extractions.running_models.run_replica import run_DNN_replica

from statics.strings.static_strings import _DIRECTORY_EXTRACTIONS, _DIRECTORY_EXTRACTIONS_MODELS_, _DIRECTORY_EXTRACTIONS__MODELS_KINEMATIC_SETS

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
        kinematic_set_dataframe: DataFrame,
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

        # (6): Obtain the replica number by adding 1 to the index:
        replica_number = replica_index + 1

        # (7): Propose a replica name:
        model_file_name = f"replica_{replica_number}.h5"

        # (8): Propose a file path to save the replica data and then just find it:
        possible_data_path = f"{_DIRECTORY_EXTRACTIONS}\\{_DIRECTORY_EXTRACTIONS_MODELS_}//{_DIRECTORY_EXTRACTIONS__MODELS_KINEMATIC_SETS}"
        directory_for_model_data = find_directory(os.getcwd(), possible_data_path)

        # (9): Hopefully (8) worked, then just save the data there:
        neural_network.save(
            f"{directory_for_model_data}/{model_file_name}",
            save_format = 'h5'
        )

        if verbose:
            print(f"> Replica #{replica_number} finished running...")
    
        end_time_in_milliseconds = datetime.datetime.now().replace(microsecond = 0)

        print(f"> Replica job finished in {end_time_in_milliseconds - start_time_in_milliseconds}ms.")

        perform_replica_analytics(neural_network_history)