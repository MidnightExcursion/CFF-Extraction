# External Library | numpy
from numpy import array

# External Library | TensorFlow
from tensorflow.keras import backend

# External Library | Matplotlib
import matplotlib.pyplot as plt

# External Library | NumPy
from numpy import arange

# Native Library | TensorFlow
from utilities.plotting.plot_customizer import PlotCustomizer

from statics.strings.static_strings import _COLUMN_NAME_X_BJORKEN
from statics.strings.static_strings import _COLUMN_NAME_Q_SQUARED
from statics.strings.static_strings import _COLUMN_NAME_T_MOMENTUM_CHANGE
from statics.strings.static_strings import _COLUMN_NAME_AZIMUTHAL_PHI
from statics.strings.static_strings import _COLUMN_NAME_LEPTON_MOMENTUM 

from statics.strings.static_strings import _HYPERPARAMETER_NUMBER_OF_EPOCHS

def perform_replica_analytics(entire_dataframe, trained_neural_network):

    # (1): Obtain the kinematics of the entire dataframe:
    all_dataframe_kinematics = entire_dataframe[[_COLUMN_NAME_Q_SQUARED, _COLUMN_NAME_X_BJORKEN, _COLUMN_NAME_T_MOMENTUM_CHANGE, _COLUMN_NAME_AZIMUTHAL_PHI, _COLUMN_NAME_LEPTON_MOMENTUM]]
    
    # (2): Plug-and-chug the DF numbers into the trained DNN to get the CFFs!
    predicted_cffs = array(
        backend.function(
            trained_neural_network.get_layer(name = 'input_layer').input,
            trained_neural_network.get_layer(name = 'cff_output_layer').output)(all_dataframe_kinematics))
    
    print(f"> Predicting CFFs to be: {predicted_cffs}")

    # (3); Plug-and-chug the numbers to get the CROSS SECTION!
    predicted_cross_sections = array(
        backend.function(
            trained_neural_network.get_layer(name = 'input_layer').input,
            trained_neural_network.get_layer(name = 'TotalFLayer').output)(all_dataframe_kinematics))
    
    print(f"> Predicted cross section to be: {predicted_cross_sections}")


def construct_network_loss_plot(network_loss_data):

    # (1): Set up the Figure instance
    figure_instance = plt.figure(figsize = (18, 6))

    # (2): Add an Axes Object:
    axis_instance = figure_instance.add_subplot(1, 1, 1)
    
    # (3): Customize the Axes Object:
    plot_customization = PlotCustomizer(
        axis_instance,
        title = r"Loss vs. Epoch",
        xlabel = r"Epoch",
        ylabel = r"Loss")
    
    # (4): Add data to the Axes Object:
    plot_customization.add_line_plot(
        arange(_HYPERPARAMETER_NUMBER_OF_EPOCHS), 
        network_loss_data,  
        label = "Network Loss",
        color = 'blue')
    
    figure_instance.savefig('COCKZ.jpg')

def construct_network_validation_loss_plot(network_validation_loss_data):

    # (1): Set up the Figure instance
    figure_instance = plt.figure(figsize = (18, 6))

    # (2): Add an Axes Object:
    axis_instance = figure_instance.add_subplot(1, 1, 1)
    
    # (3): Customize the Axes Object:
    plot_customization = PlotCustomizer(
        axis_instance,
        title = r"Validation Loss vs. Epoch",
        xlabel = r"Epoch",
        ylabel = r"Validation")
    
    # (4): Add data to the Axes Object:
    plot_customization.add_line_plot(
        arange(_HYPERPARAMETER_NUMBER_OF_EPOCHS), 
        network_validation_loss_data,  
        label = "Validation Loss",
        color = 'blue')
    
    figure_instance.savefig('LOLER.jpg')
