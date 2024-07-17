import os
import pandas as pd

from src.utils.unzip import unzip_no_pass

def create_dataframe(path_to_zip, 
                     out_path="./data/spam_filter", 
                     unzip=False,
                     save_ext="csv"
                    ):
    '''
    This creates a dataframe from spam filtering dataset.
    
    Args:
        path_to_zip:
            (str) path to zip file.
        out_path:
            (str) path to store processed data.
        unzip:
            (bool) whether to unzip files before processing. Default False.
        save_ext:
            (str) file extension of saved data. Choices: csv, parquet.
            Default csv.
    
    Returns:
        None
    '''
    assert save_ext in ["csv", "parquet"], "Extensions allowed are csv and parquet."
    
    path_to_zip = os.path.normpath(path_to_zip)
    out_path = os.path.normpath(out_path)
    
    # unzip file first
    if unzip:
        unzip_no_pass(path=path_to_zip, out_path=out_path)
    
    filename = os.path.basename(path_to_zip).split(".")[0]
    
    #start of processing unzipped files
    df = pd.read_csv(os.path.join(out_path, filename, "labels"), sep="/", header=None)
    
    # clean dataframe
    # initial columns are 0, 1, 2, 3
    df[0] = df[0].str.replace(" ..", "") # labels has extra artifacts
    df = df.drop(columns=1) # drop unnecessary column
    df = df.rename(columns = {0 : "Class", 2 : "Folder", 3: "File"}) # rename
    
    if save_ext == "csv":
        df.to_csv(os.path.join(out_path, f"{filename}.{save_ext}"), index=False)
    elif save_ext == "parquet":
        df.to_parquet(os.path.join(out_path, f"{filename}.{save_ext}"))