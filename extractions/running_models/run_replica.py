from pandas import DataFrame

from utilities.generation.generate_pseudodata import generate_replica_data

def run_DNN_replica(kinematic_set, replica_number: int, base_dataframe: DataFrame):

    # (1): Generate the data for this DNN replica:
    generated_replica_data = generate_replica_data(base_dataframe)
    
    # (2): 