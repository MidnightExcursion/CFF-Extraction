from pandas import DataFrame

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from extractions.models.dnn_model import deep_neural_network

from statics.strings.static_strings import _COLUMN_NAME_X_BJORKEN
from statics.strings.static_strings import _COLUMN_NAME_Q_SQUARED
from statics.strings.static_strings import _COLUMN_NAME_T_MOMENTUM_CHANGE
from statics.strings.static_strings import _COLUMN_NAME_AZIMUTHAL_PHI
from statics.strings.static_strings import _COLUMN_NAME_LEPTON_MOMENTUM 

from statics.strings.static_strings import _HYPERPARAMETER_NUMBER_OF_EPOCHS
from statics.strings.static_strings import _HYPERPARAMETER_BATCH_SIZE
from statics.strings.static_strings import _HYPERPARAMETER_LR_PATIENCE
from statics.strings.static_strings import _HYPERPARAMETER_LR_FACTOR
from statics.strings.static_strings import _HYPERPARAMETER_EARLYSTOP_PATIENCE_INTEGER
from statics.strings.static_strings import _DNN_VERBOSE_SETTING

def run_DNN_replica(training_x_data, training_y_data, testing_x_data, testing_y_data):

    # (5): Initialize the DNN:
    neural_network = deep_neural_network()
    
    # (7): Fit the Model:
    neural_network_training_history = neural_network.fit(
        training_x_data,
        training_y_data,
        validation_data = (testing_x_data, testing_y_data),
        epochs = _HYPERPARAMETER_NUMBER_OF_EPOCHS,
        callbacks = [
            ReduceLROnPlateau(monitor = 'loss', factor = _HYPERPARAMETER_LR_FACTOR, patience = _HYPERPARAMETER_LR_PATIENCE, mode='auto'),
            EarlyStopping(monitor = 'loss',patience = _HYPERPARAMETER_EARLYSTOP_PATIENCE_INTEGER)
        ],
        batch_size = _HYPERPARAMETER_BATCH_SIZE,
        verbose = _DNN_VERBOSE_SETTING
    )

    # (8): Return:
    return neural_network_training_history