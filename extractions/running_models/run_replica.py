from pandas import DataFrame

from extractions.models.dnn_model import deep_neural_network

from utilities.data_handling.split_ann_data import split_data
from utilities.generation.generate_pseudodata import generate_replica_data

from statics.strings.static_strings import _COLUMN_NAME_X_BJORKEN
from statics.strings.static_strings import _COLUMN_NAME_Q_SQUARED
from statics.strings.static_strings import _COLUMN_NAME_T_MOMENTUM_CHANGE
from statics.strings.static_strings import _COLUMN_NAME_AZIMUTHAL_PHI
from statics.strings.static_strings import _COLUMN_NAME_LEPTON_MOMENTUM 
from statics.strings.static_strings import _COLUMN_NAME_CROSS_SECTION
from statics.strings.static_strings import _COLUMN_NAME_CROSS_SECTION_ERROR

def run_DNN_replica(kinematic_set_dataframe: DataFrame):

    # (1): Generate the data for this DNN replica:
    generated_replica_data = generate_replica_data(kinematic_set_dataframe)

    # (2): Identify the "x values" for our model:
    x_data = generated_replica_data[[_COLUMN_NAME_Q_SQUARED, _COLUMN_NAME_X_BJORKEN, _COLUMN_NAME_T_MOMENTUM_CHANGE, _COLUMN_NAME_AZIMUTHAL_PHI, _COLUMN_NAME_LEPTON_MOMENTUM]]

    # (3): Identify the "y values" for our model:
    y_data = generated_replica_data[_COLUMN_NAME_CROSS_SECTION]
    y_data_error = generated_replica_data[_COLUMN_NAME_CROSS_SECTION_ERROR]
    
    # (4): Split the Data into Training/Testing:
    train_x_data, training_y_data, training_y_error_data, testing_x_data, testing_y_data, testing_y_error_data = split_data(x_data, y_data, y_data_error)

    # (5): Initialize the DNN:
    neural_network = deep_neural_network()

    # (6): Perform the DNN Fitting Procedure:

    # (7): Save the Model:

    # (8): Perform the Calculation:

    # (9): Make some Figures: