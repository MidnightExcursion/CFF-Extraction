from pandas import DataFrame

import tensorflow as tf

from extractions.models.dnn_model import full_cross_section_dnn_model

from statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_NUMBER_OF_EPOCHS
from statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_BATCH_SIZE
from statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_LR_PATIENCE
from statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_LR_FACTOR
from statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_EARLYSTOP_PATIENCE_INTEGER
from statics.model_architecture.model_hyperparameters import _DNN_VERBOSE_SETTING

def run_DNN_replica(
        training_x_data: DataFrame,
        training_y_data: DataFrame,
        testing_x_data: DataFrame,
        testing_y_data: DataFrame):
    """
    # Description:
    Run a single DNN replica.
    """

    # (1): iNITIALIZE THE FULL MODEL:
    full_cross_section_model = full_cross_section_dnn_model()
    
    # (2): Fit the Model:
    neural_network_training_history = full_cross_section_model.fit(
        training_x_data,
        training_y_data,
        validation_data = (testing_x_data, testing_y_data),
        epochs = _HYPERPARAMETER_NUMBER_OF_EPOCHS,
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor = _HYPERPARAMETER_LR_FACTOR, patience = _HYPERPARAMETER_LR_PATIENCE, mode = 'auto'),
            tf.keras.callbacks.EarlyStopping(monitor = 'loss',patience = _HYPERPARAMETER_EARLYSTOP_PATIENCE_INTEGER)
        ],
        batch_size = _HYPERPARAMETER_BATCH_SIZE,
        verbose = _DNN_VERBOSE_SETTING
    )

    # (3): Return the history/diagnostic data from the training:
    return full_cross_section_model, neural_network_training_history
