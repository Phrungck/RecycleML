import os
import pandas as pd

from src.utils.unzip import unzip_no_pass

# create dataframe from spam filtering dataset
def create_dataframe(path_to_zip, 
                     out_path="./data/spam_filter", 
                     unzip=False,
                     save_ext="csv"
                    ):
    
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
    
    # replace spam and ham values
    df["Class"] = df["Class"].replace({"ham" : 0, "spam" : 1})
    
    # combine with text / email data
    for i in range(len(df)):
        folder_id = "{0:0=3d}".format(df.at[i,'Folder'])
        file_id = "{0:0=3d}".format(df.at[i,'File'])
        
        path = os.path.normpath(os.path.join(out_path, filename, "data", folder_id, file_id))
        
        # there will be an error if utf-8 is used as encoding
        df.at[i, "Email"] = open(path, encoding="latin1").read()
        
    df["Email"] = df["Email"].str.lower()

    if save_ext == "csv":
        df.to_csv(os.path.join(out_path, f"{filename}.{save_ext}"), index=False)
    elif save_ext == "parquet":
        df.to_parquet(os.path.join(out_path, f"{filename}.{save_ext}"))