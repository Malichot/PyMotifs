import pandas as pd
import os

# Ref : https://stackoverflow.com/questions/69118811/how-to-read-all-txt-files-from-a-directory

def read_txts(path):
    """
    Read all txts inside a folder and put them into a dataframe.
    Folder must contain only txt files...
    """
    files_content = [] # create empty list to save content

    for filename in filter(lambda p: p.endswith("txt"), os.listdir(path)): # filtre les fichiers qui se terminent par txt
      # et liste les fichiers dans le path.
        filepath = os.path.join(path, filename)
        with open(filepath, mode='r') as f:
            files_content += [f.read()]

    print(f'There are {len(files_content)} texts in folder')
    all_files = os.listdir(path=path)
    df = pd.DataFrame()
    df['filename'] = all_files
    df['text'] = files_content
    return(df)

read_txts(path="/Users/adesacy/Desktop/Motifs_Python/corpus")