# External Library | numpy
from numpy import array

# Native Library | TensorFlow
from tensorflow.keras.backend import function

from statics.strings.static_strings import _COLUMN_NAME_X_BJORKEN
from statics.strings.static_strings import _COLUMN_NAME_Q_SQUARED
from statics.strings.static_strings import _COLUMN_NAME_T_MOMENTUM_CHANGE
from statics.strings.static_strings import _COLUMN_NAME_AZIMUTHAL_PHI
from statics.strings.static_strings import _COLUMN_NAME_LEPTON_MOMENTUM 

def fit_neural_network(trained_neural_network):

    # (1): Obtain the kinematics of the entire dataframe:
    all_dataframe_kinematics = df[[_COLUMN_NAME_Q_SQUARED, _COLUMN_NAME_X_BJORKEN, _COLUMN_NAME_T_MOMENTUM_CHANGE, _COLUMN_NAME_AZIMUTHAL_PHI, _COLUMN_NAME_LEPTON_MOMENTUM]]
    
    # (2): Plug-and-chug the DF numbers into the trained DNN to get the CFFs!
    predicted_cffs = array(
        function(
            trained_neural_network.get_layer(name = 'input_layer').input,
            trained_neural_network.get_layer(name = 'cff_output_layer').output)(all_dataframe_kinematics))

    # (3); Plug-and-chug the numbers to get the CROSS SECTION!
    predicted_cross_sections = array(
        function(
            trained_neural_network.get_layer(name = 'input_layer').input,
            trained_neural_network.get_layer(name = 'TotalFLayer').output)(all_dataframe_kinematics))