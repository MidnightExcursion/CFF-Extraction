from numpy import ones
from numpy.random import choice

def split_data(x_data, y_data, y_error_data, fraction_of_data_to_split: float = 0.1):
    """
    
    """

    _CHOOSE_WITH_REPLACEMENT_SETTING = False

    # (1): Obtain the number of rows in the y_data:
    length_of_y_data = len(y_data)

    # (2): Obtain a list of indices that refer to each row in the y_data:
    list_of_possible_indices_of_y_data = list(range(length_of_y_data))

    # (3): Calculate (with coersion) the number of indices to randomly select 
    total_number_of_elements_to_select = int(length_of_y_data * fraction_of_data_to_split)
    
    # (4): Generate random indices for the Test Data:
    test_indices  = choice(list_of_possible_indices_of_y_data, size = total_number_of_elements_to_select, replace = _CHOOSE_WITH_REPLACEMENT_SETTING)

    # (5): Create a mask of indices
    mask = ones(len(y_data), dtype = bool)
    mask[test_indices] = False

    # (6): Obtain the Testing and Training Data for the x_data: (uses Pandas' i(ndex)loc)
    testing_x_data = x_data.iloc[test_indices]
    train_x_data = x_data[mask]

    # (6): Obtain the Testing and Training Data for the y_data: (uses Pandas' i(ndex)loc)
    testing_y_data = y_data.iloc[test_indices]
    training_y_data = y_data[mask]

    # (6): Obtain the Testing and Training Data for the y_data error: (uses Pandas' i(ndex)loc)
    testing_y_error_data = y_error_data.iloc[test_indices]
    training_y_error_data = y_error_data[mask]

    # (7): Return the generated training/testing data:
    return train_x_data, training_y_data, training_y_error_data, testing_x_data, testing_y_data, testing_y_error_data

