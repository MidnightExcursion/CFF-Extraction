# External Library | numpy
from numpy import array

# External Library | Matplotlib
import matplotlib.pyplot as plt

# Native Library | TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Model

from utilities.plotting.plot_customizer import PlotCustomizer

from statics.strings.static_strings import _COLUMN_NAME_X_BJORKEN
from statics.strings.static_strings import _COLUMN_NAME_Q_SQUARED
from statics.strings.static_strings import _COLUMN_NAME_T_MOMENTUM_CHANGE
from statics.strings.static_strings import _COLUMN_NAME_AZIMUTHAL_PHI
from statics.strings.static_strings import _COLUMN_NAME_LEPTON_MOMENTUM 

def perform_replica_analytics(entire_dataframe, trained_neural_network):
    """
    # Description:
    -- 
    We now analyze the result of the replicas.

    # Arguments:

    entire_dataframe: pd.DataFrame

    trained_neural_network: tf.Model

    # Returns:


    """
    
    # (1): Obtain the kinematics of the entire dataframe:
    all_dataframe_kinematics = entire_dataframe[[_COLUMN_NAME_Q_SQUARED, _COLUMN_NAME_X_BJORKEN, _COLUMN_NAME_T_MOMENTUM_CHANGE, _COLUMN_NAME_AZIMUTHAL_PHI, _COLUMN_NAME_LEPTON_MOMENTUM]]
    
    # (2): Define TensorFlow functions for predicting CFFs:
    @tf.function
    def predict_cffs(input_data):
        output_layer = trained_neural_network.get_layer(name='cff_output_layer').output
        intermediate_model = Model(inputs = trained_neural_network.input, outputs = output_layer)
        return intermediate_model(input_data)
    
    # (3): Define TensorFlow functions for predicting the total cross-section:
    @tf.function
    def predict_cross_sections(input_data):
        return trained_neural_network(input_data)
    
    # (4): Plug-and-chug the DF numbers into the trained DNN to get the CFFs!
    predicted_cffs = array(predict_cffs(all_dataframe_kinematics))

    # (5) Plug-and-chug the numbers to get the CROSS SECTION!
    predicted_cross_sections = array(predict_cross_sections(all_dataframe_kinematics))

def construct_plot_data(x_data, y_data, plot_title = "", x_label = "", y_label = ""):
    """
    
    """
    
    # (1): Set up the Figure instance
    figure_instance = plt.figure(figsize = (18, 6))

    # (2): Add an Axes Object:
    axis_instance = figure_instance.add_subplot(1, 1, 1)
    
    # (3): Customize the Axes Object:
    plot_customization = PlotCustomizer(
        axis_instance,
        title = plot_title,
        xlabel = x_label,
        ylabel = y_label)
    
    # (4): Add data to the Axes Object:
    plot_customization.add_line_plot(
        x_data, 
        y_data,  
        color = 'black')
    
    return figure_instance