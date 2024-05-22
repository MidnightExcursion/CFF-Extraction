import tensorflow as tf

def DNNmodel():

    _HYPERPARAMETER_LEARNING_RATE = 0.001
    _HYPERPARAMETER_NUMBER_OF_EPOCHS = 1000
    _HYPERPARAMETER_NUMBER_OF_REPLICAS = 300
    _HYPERPARAMETER_EARLYSTOP_PATIENCE_INTEGER = 1000
    _HYPERPARAMETER_LR_PATIENCE = 400
    _HYPERPARAMETER_LR_FACTOR = 0.9

    _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_1 = 480
    _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_2 = 320
    _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_3 = 32
    _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_4 = 32
    _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_5 = 4

    # (1): Initialize the Network with Uniform Random Sampling: [-0.1, -0.1]:
    initializer = tf.keras.initializers.RandomUniform(
        minval = -0.1,
        maxval = 0.1,
        seed = None)

    # (2) Make the TF Input Layer:
    inputs = tf.keras.Input(shape=(5), name='input_layer')
    
    # (3): Define the five inputs to the network:
    QQ, x_b, t, phi, k = tf.split(inputs, num_or_size_splits=5, axis=1)

    # (4): Combine the kinematics as a single list:
    kinematics = tf.keras.layers.concatenate([QQ, x_b, t])

    # (5): Define the Model Architecture:
    x1 = tf.keras.layers.Dense(_HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_1, activation = "relu", kernel_initializer = initializer)(kinematics)
    x2 = tf.keras.layers.Dense(_HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_2, activation = "tanhshrink", kernel_initializer = initializer)(x1)
    x3 = tf.keras.layers.Dense(_HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_3, activation = "relu", kernel_initializer = initializer)(x2)
    x4 = tf.keras.layers.Dense(_HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_4, activation = "relu", kernel_initializer = initializer)(x3)
    outputs = tf.keras.layers.Dense(_HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_5, activation = "linear", kernel_initializer = initializer, name = 'cff_output_layer')(x4)
    
    # (6): We need both the Kinematics and the CFFs to evaluate the cross section:
    total_FInputs = tf.keras.layers.concatenate([inputs, outputs], axis = 1)
    
    # (7): Compute, algorithmically, the cross section:
    TotalF = TotalFLayer(name='TotalFLayer')(total_FInputs)

    # (8): Define the model as as Keras Model:
    tfModel = tf.keras.Model(inputs=inputs, outputs = TotalF, name = "tfmodel")

    # (9): Compile the model:
    tfModel.compile(
        optimizer = tf.keras.optimizers.Adam(_HYPERPARAMETER_LEARNING_RATE),
        loss = tf.keras.losses.MeanSquaredError()
    )
    return tfModel