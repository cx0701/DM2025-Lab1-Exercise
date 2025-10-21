import nltk
import pandas as pd
"""
Helper functions for data mining lab session 2018 Fall Semester
Author: Elvis Saravia
Email: ellfae@gmail.com
"""

def format_rows(docs):
    """ format the text field and strip special characters """
    D = []
    for d in docs.data:
        temp_d = " ".join(d.split("\n")).strip('\n\t')
        D.append([temp_d])
    return D

def format_labels(target, docs):
    """ format the labels """
    return docs.target_names[target]

def check_missing_values(row):
    """ functions that check and verifies if there are missing values in dataframe """
    counter = 0
    for element in row:
        if element == True:
            counter+=1
    return ("The amoung of missing records is: ", counter)
def check_unk_values(df):
    """
    回傳每個類別型欄位中 'UNK' 的數量（Series 一欄）
    """
    unk_counts = {}
    
    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
            unk_counts[col] = (df[col] == 'UNK').sum()
    
    return pd.Series(unk_counts, name='UNK Count')
def tokenize_text(text, remove_stopwords=False):
    """
    Tokenize text using the nltk library
    """
    tokens = []
    for d in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(d, language='english'):
            # filters here
            tokens.append(word)
    return tokens

def check_negative_values(df):
    """
    回傳每個欄位負值的數量（Series 一欄）
    """
    negative_counts = {}
    
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            negative_counts[col] = (df[col] < 0).sum()
    
    return pd.Series(negative_counts, name='Negative Count')

def drop_negative_rows(df, cols, inplace=False):
    """
    刪除指定欄位含負值的列
    回傳：
    DataFrame（如果 inplace=True，則回傳 None）
    """
    # 找出含負值的列索引
    negative_index = df[~(df[cols] >= 0).all(axis=1)].index
    
    if inplace:
        df.drop(index=negative_index, inplace=True)
    else:
        return df.drop(index=negative_index)
