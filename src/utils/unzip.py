import os
import zipfile

def unzip_no_pass(path, out_path):
    '''
    Extracts all contents of a zip file without a password.
    
    Args:
        path:
            (str) path to zip file to extract.
        out_path:
            (str) path to directory to store the extracted contents.
            
    Returns:
        None
    '''
    # make sure out path directory existsge
    os.makedirs(out_path, exist_ok=True)
    
    with zipfile.ZipFile(path, 'r') as file:
        file.extractall(out_path)