from pandas import DataFrame

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from extractions.models.dnn_model import deep_neural_network

from statics.strings.static_strings import _HYPERPARAMETER_NUMBER_OF_EPOCHS
from statics.strings.static_strings import _HYPERPARAMETER_BATCH_SIZE
from statics.strings.static_strings import _HYPERPARAMETER_LR_PATIENCE
from statics.strings.static_strings import _HYPERPARAMETER_LR_FACTOR
from statics.strings.static_strings import _HYPERPARAMETER_EARLYSTOP_PATIENCE_INTEGER
from statics.strings.static_strings import _DNN_VERBOSE_SETTING

def run_DNN_replica(
        training_x_data: DataFrame,
        training_y_data: DataFrame,
        testing_x_data: DataFrame,
        testing_y_data: DataFrame):

    # (1): Initialize the DNN -- Look at this function to see how the DNN is customized"
    neural_network = deep_neural_network()
    
    # (2): Fit the Model:
    neural_network_training_history = neural_network.fit(
        training_x_data,
        training_y_data,
        validation_data = (testing_x_data, testing_y_data),
        epochs = _HYPERPARAMETER_NUMBER_OF_EPOCHS,
        callbacks = [
            ReduceLROnPlateau(monitor = 'loss', factor = _HYPERPARAMETER_LR_FACTOR, patience = _HYPERPARAMETER_LR_PATIENCE, mode = 'auto'),
            EarlyStopping(monitor = 'loss',patience = _HYPERPARAMETER_EARLYSTOP_PATIENCE_INTEGER)
        ],
        batch_size = _HYPERPARAMETER_BATCH_SIZE,
        verbose = _DNN_VERBOSE_SETTING
    )

    # (3): Return the history/diagnostic data from the training:
    return neural_network, neural_network_training_history