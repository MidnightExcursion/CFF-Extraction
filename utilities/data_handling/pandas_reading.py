from pandas import DataFrame, read_csv

def read_csv_file_with_pandas(name_of_csv_file: str) -> DataFrame:
    """
    Description
    --------------
    We are using Pandas to read the .csv file. If this works, then the
    function will return the 

    Parameters
    --------------
    name_of_csv_file: str
    
    Returns
    --------------
    pandas_read_csv: Pandas DF

    Function Flow
    --------------
    (1): Try to read the CSV with Pandas. If it works,
        it works. YAY! If it don't, then return None.
    
    Notes
    --------------
    """
    
    try:

        pandas_read_csv = read_csv(name_of_csv_file)
        return pandas_read_csv
    
    except Exception as ERROR:
        
        print(f"> Error reading the .csv file with Pandas:\n> {ERROR}")
        return None