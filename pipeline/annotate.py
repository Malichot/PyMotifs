import pandas as pd
import os 
import csv
import spacy_udpipe
import re
# !pip install spacy-udpipe

def annotate(path_txt, save_path):
    """
    Function that takes .txt files, put them into a dataframe, tokenizes texts and annotate them. 
    Download the annotation model if you don't already have it.
    Uses Sapcy UDPipe annotator and do : lemmatization, pos-tagging, morphology.
    
    Parameters: 
    path_txt = path to your folder containing your texts .txt. Be careful that no hidden files are in the folder. 
        Ex : "~/Users/Desktop/PyMotifs/corpus"
    save_csv = path your folder to save the generated csv and name of the csv.
        Ex : "~/Users/Desktop/PyMotifs/output/corpus_annotated.csv"
    
    """
    # Loop over : https://stackoverflow.com/questions/69118811/how-to-read-all-txt-files-from-a-directory
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

    read_txts(path=path_txt)

    # Save into object : 
    df = read_txts(path=path_txt)

    # Change apostrophs : 

    def clean_a_bit(df):
        """
        Function to clean differnt apostrophs and withdraw possible na
        values from df
        """
        df['text'] = df['text'].replace("’", "'")
        df['text'] = df['text'].replace("'", "'")
        # Retrait des NA dans la colonne mots : 
        df['text'] = df['text'].dropna(how = 'any', axis = 0)# Drop the row 

        return(df)

    clean_a_bit(df)

    ## ------------------------------------------------------------------

    # Annotation : 

    spacy_udpipe.download("fr") # Download french model : 

    nlp = spacy_udpipe.load("fr")

    # Create a Tokenizer with the default settings for French
    # including punctuation rules and exceptions

    tokenizer = nlp.tokenizer

    # Création d'une nouvelle dataframe pour accueillir les données de l'étiquetage : 
    # On ne veut pas garder le texte intégral dans le nouveau tableau.

    annotated_datas = pd.DataFrame()

    # récupération de la colonne filename : 

    annotated_datas['filename'] = df['filename']

    # On tokenise les textes :

    annotated_datas['words'] = df['text'].apply(lambda x: nlp.tokenizer(str(x)))
    annotated_datas.head(10)

    # "Explostion" des données : un mot par ligne : 
    annotated_datas = annotated_datas.explode("words", ignore_index=True)
    annotated_datas.head(10)

    ## ------------------------------------------------------------------

    # Étiquetage et lemmatisation : 

    ## Thx to Ed Rushton :
    # Cf. https://stackoverflow.com/questions/44395656/applying-spacy-parser-to-pandas-dataframe-w-multiprocessing

    ## Spacy is highly optimised and does the multiprocessing for you. 
    ## As a result, I think your best bet is to take the data out of 
    ## the Dataframe and pass it to the Spacy pipeline as a list rather 
    ## than trying to use .apply directly.
    ## You then need to the collate the results of the parse, and put 
    ## this back into the Dataframe. 

    lemma = []
    pos = []
    morph = []
    dep = []

    for doc in nlp.pipe(annotated_datas['words'].astype('unicode').values, batch_size=50):
        if doc.has_annotation:
            #tokens.append([n.text for n in doc])
            lemma.append([n.lemma_ for n in doc])
            pos.append([n.pos_ for n in doc])
            morph.append([n.morph for n in doc])
            dep.append([n.dep_ for n in doc])
        else:
            # We want to make sure that the lists of parsed results have the
            # same number of entries of the original Dataframe, so add some blanks in case the parse fails
            # tokens.append(None)
            lemma.append(None)
            pos.append(None)
            morph.append(None)

    # corpus_test['tokens'] = tokens
    annotated_datas['lemma'] = lemma
    annotated_datas['pos'] = pos
    annotated_datas['morph'] = morph
    print(annotated_datas.head())
    
    
    # Explosion des données : 
    annotated_datas = annotated_datas.explode("words", ignore_index=True)
    #annotated_datas = annotated_datas.explode("pos", ignore_index=True)
    #annotated_datas = annotated_datas.explode("lemma", ignore_index=True)
    
    # -----------------------------------------------------------------------------
    
    # Saving into csv : 

    annotated_datas.to_csv(path_or_buf=save_path, encoding='utf-8')