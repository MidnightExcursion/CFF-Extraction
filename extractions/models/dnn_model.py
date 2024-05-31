import tensorflow as tf

# from tensorflow_addons.activations import tanhshrink
# tf.keras.utils.get_custom_objects().update({'tanhshrink': tanhshrink})

from extractions.models.cross_section_layer import TotalFLayer

from statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_LEARNING_RATE
from statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_1
from statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_2
from statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_3
from statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_4
from statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_5

def deep_neural_network():

    # (1): Initialize the Network with Uniform Random Sampling: [-0.1, -0.1]:
    initializer = tf.keras.initializers.RandomUniform(
        minval = -0.1,
        maxval = 0.1,
        seed = None)

    # (2) Make the TF Input Layer:
    inputs = tf.keras.Input(shape=(5,), name = 'input_layer')
    
    # (3): Define the five inputs to the network:
    # QQ, x_b, t, phi, k = tf.split(inputs[:, :5], num_or_size_splits = 5, axis = 1)

    # # (4): Combine the kinematics as a single list:
    # kinematics = tf.keras.layers.concatenate([QQ, x_b, t])

    kinematics = inputs[:, :3]  # Extract the kinematics part of the inputs
    # cffs = inputs[:, 5:]

    # (5): Define the Model Architecture:
    x1 = tf.keras.layers.Dense(_HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_1, activation = "relu", kernel_initializer = initializer)(kinematics)
    x2 = tf.keras.layers.Dense(_HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_2, activation = "tanh", kernel_initializer = initializer)(x1)
    x3 = tf.keras.layers.Dense(_HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_3, activation = "relu", kernel_initializer = initializer)(x2)
    x4 = tf.keras.layers.Dense(_HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_4, activation = "relu", kernel_initializer = initializer)(x3)
    outputs = tf.keras.layers.Dense(_HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_5, activation = "linear", kernel_initializer = initializer, name = 'cff_output_layer')(x4)
    
    # (6): We need both the Kinematics and the CFFs to evaluate the cross section:
    total_FInputs = tf.keras.layers.concatenate([inputs, outputs], axis = 1)
    
    # (7): Compute, algorithmically, the cross section:
    TotalF = TotalFLayer(name = 'TotalFLayer')(total_FInputs)

    # (8): Define the model as as Keras Model:
    tfModel = tf.keras.Model(inputs = inputs, outputs = TotalF, name = "tfmodel")

    # (9): Compile the model:
    tfModel.compile(
        optimizer = tf.keras.optimizers.Adam(_HYPERPARAMETER_LEARNING_RATE),
        loss = tf.keras.losses.MeanSquaredError()
    )

    # (10): Return the model:
    return tfModel