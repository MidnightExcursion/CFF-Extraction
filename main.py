# Native Library | argparse
import argparse

# Native Library | os
import os

from statics.strings.static_strings import _DIRECTORY_DATA

from statics.strings.static_strings import _ARGPARSE_DESCRIPTION

from statics.strings.static_strings import _ARGPARSE_ARGUMENT_INPUT_DATAFILE
from statics.strings.static_strings import _ARGPARSE_ARGUMENT_KINEMATIC_SET_NUMBER
from statics.strings.static_strings import _ARGPARSE_ARGUMENT_NUMBER_REPLICAS
from statics.strings.static_strings import _ARGPARSE_ARGUMENT_VERBOSE

from statics.strings.static_strings import _ARGPARSE_ARGUMENT_DESCRIPTION_INPUT_DATAFILE
from statics.strings.static_strings import _ARGPARSE_ARGUMENT_DESCRIPTION_KINEMATIC_SET_NUMBER
from statics.strings.static_strings import _ARGPARSE_ARGUMENT_DESCRIPTION_NUMBER_REPLICAS
from statics.strings.static_strings import _ARGPARSE_ARGUMENT_DESCRIPTION_VERBOSE

# DataFrame Columns
from statics.strings.static_strings import _COLUMN_NAME_KINEMATIC_SET

def main(
        kinematics_dataframe_path: str,
        kinematic_set_number: int,
        number_of_replicas: int,
        verbose: bool = False):

    # extractions > running_models > run_replica > run_DNN_replica
    from extractions.running_models.run_replica_method import run_replica_method

    # utilities > data_handling > pandas_reading > read_csv_file_with_pandas
    from utilities.data_handling.pandas_reading import read_csv_file_with_pandas

    # utilities > directories > find_directory
    from utilities.directories.searching_directories import find_directory
    
    try:

        # (1): Construct the filepath to the data:
        possible_data_path = f"{_DIRECTORY_DATA}\\{kinematics_dataframe_path}"

        if verbose:
            print(f"> Possible data path is: {possible_data_path}")

        # (2): Now, check if the kinematics is actually there:
        kinematics_dataframe_file_path = find_directory(os.getcwd(), possible_data_path)

        if verbose:
            print(f"> Did we find the filepath for the kinematics? {kinematics_dataframe_file_path}")

        # (3): If the file was there, turn it into a DF with Pandas:
        kinematics_dataframe = read_csv_file_with_pandas(kinematics_dataframe_file_path)

        if verbose:
            print(f"> Did we convert the kinematics file to a Pandas DF? {kinematics_dataframe is not None}")

        # (4): Partition the DF on a fixed kinematic set:
        fixed_kinematic_set_dataframe = kinematics_dataframe[kinematics_dataframe[_COLUMN_NAME_KINEMATIC_SET] == kinematic_set_number]

        if verbose:
            print(f"> Did we manage to fix to a kinematic range? {fixed_kinematic_set_dataframe is not None}")

        # (5): Run the thing:
        trained_neural_network = run_replica_method(fixed_kinematic_set_dataframe, number_of_replicas, verbose)

        # (6): Perform Analytics:
        network_analytics = fit_neural_network(kinematics_dataframe, trained_neural_network)

    except Exception as ERROR:
        print(f"> Error when running main.py:\n{ERROR}")
        return


if __name__ == "__main__":

    # (1): Create an instance of the ArgumentParser
    parser = argparse.ArgumentParser(description = _ARGPARSE_DESCRIPTION)

    # (2): Enforce the path to the datafile:
    parser.add_argument(
        '-d',
        _ARGPARSE_ARGUMENT_INPUT_DATAFILE,
        type = str,
        required = True,
        help = _ARGPARSE_ARGUMENT_DESCRIPTION_INPUT_DATAFILE)
    
    # (3): Enforce the path to the datafile:
    parser.add_argument(
        '-kin',
        _ARGPARSE_ARGUMENT_KINEMATIC_SET_NUMBER,
        type = int,
        required = True,
        help = _ARGPARSE_ARGUMENT_DESCRIPTION_KINEMATIC_SET_NUMBER)

    # (4): Enforce the number of replicas:
    parser.add_argument(
        '-nr',
        _ARGPARSE_ARGUMENT_NUMBER_REPLICAS,
        type = int,
        required = True,
        help = _ARGPARSE_ARGUMENT_DESCRIPTION_NUMBER_REPLICAS)

    # (5): Ask, but don't enforce debugging verbosity:
    parser.add_argument(
        '-v',
        _ARGPARSE_ARGUMENT_VERBOSE,
        required = False,
        action = 'store_false',
        help = _ARGPARSE_ARGUMENT_DESCRIPTION_VERBOSE)
    
    arguments = parser.parse_args()

    main(
        kinematics_dataframe_path = arguments.input_datafile,
        kinematic_set_number = arguments.kinematic_set,
        number_of_replicas = arguments.number_of_replicas,
        verbose = arguments.verbose)