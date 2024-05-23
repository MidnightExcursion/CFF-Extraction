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

    # Native Library | datetime
    import datetime

    # extractions > running_models > run_replica > run_DNN_replica
    from extractions.running_models.run_replica import run_DNN_replica

    # utilities > data_handling > pandas_reading > read_csv_file_with_pandas
    from utilities.data_handling.pandas_reading import read_csv_file_with_pandas

    # utilities > directories > find_directory
    from utilities.directories.find_directory import find_directory
    
    # (1): Construct the filepath to the data:
    possible_data_path = f"{_DIRECTORY_DATA}\\{kinematics_dataframe_path}"

    # (2): Now, check if the kinematics is actually there:
    kinematics_dataframe_file_path = find_directory(os.getcwd(possible_data_path))

    # (3): If the file was there, turn it into a DF with Pandas:
    kinematics_dataframe = read_csv_file_with_pandas(kinematics_dataframe_file_path)

    # (4): Partition the DF on a fixed kinematic set:
    fixed_kinematic_set_dataframe = kinematics_dataframe[kinematics_dataframe[_COLUMN_NAME_KINEMATIC_SET] == kinematic_set_number]

    # (5): Begin timing the replica functions:
    start_time_in_milliseconds = datetime.datetime.now().replace(microsecond = 0)

    for replica in range(number_of_replicas):
        
        if verbose:
            print(f"> Replica #{replica + 1} now running...")

        DNN_history = run_DNN_replica(fixed_kinematic_set_dataframe)

        if verbose:
            print(f"> Replica #{replica} finished running...")

    end_time_in_milliseconds = datetime.datetime.now().replace(microsecond = 0)

    print(f"> Replica job finished in {end_time_in_milliseconds - start_time_in_milliseconds}ms.")


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