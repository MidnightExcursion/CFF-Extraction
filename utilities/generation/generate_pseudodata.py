from pandas import DataFrame

from statics.strings.static_strings import _COLUMN_NAME_CROSS_SECTION, _COLUMN_NAME_CROSS_SECTION_ERROR

from utilities.mathematics.statistics import sample_from_numpy_normal_distribution

def generate_replica_data(dataframe: DataFrame):
    """
    """

    try:

        # (1): Reaplce the Cross Section column with a number sampled from a Normal Distribution:
        dataframe.loc[_COLUMN_NAME_CROSS_SECTION] = [sample_from_numpy_normal_distribution(*cross_section_and_error_tuple) for cross_section_and_error_tuple in tuple(zip(dataframe[_COLUMN_NAME_CROSS_SECTION], dataframe[_COLUMN_NAME_CROSS_SECTION_ERROR]))]

        # (2): Let's just set these things to be the same for clarity:
        replica_dataframe = dataframe

        # (3): Return the manipulated dataframe:
        return replica_dataframe
    
    except Exception as ERROR:
        
        print(f"> Error when generating replica data -- No manipulation performed:\n{ERROR}")
        return dataframe