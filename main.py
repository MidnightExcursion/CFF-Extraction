# Native Library | argparse
import argparse

# Native Library | os
import os

from extractions.extract import extraction

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
    
    try:

        extraction(kinematics_dataframe_path,
        kinematic_set_number,
        number_of_replicas,
        verbose)

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