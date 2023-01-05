import sys
import os
import re
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function reads two dataframes, merges them and return the merged dataframe.
    Input: messages_filepath: str
           categories_filepath: str
    -----------
    Output: df_merge: pd.DataFrame

    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df_merge = pd.merge(
        messages,
        categories,
        how='inner',
        on='id'
    )
    return df_merge

def clean_data(df):
    """
    This function takes an input dataframe and cleans it.
    Input: df: pd.DataFrame
    -----------
    Output: df_concat: pd.DataFrame

    """

    df_clean = df.copy()

    # split by the 36 different categories.
    df_cat = df_clean.categories.str.split(';', expand=True)

    cat_row_cols = df_cat.iloc[0].apply(lambda x: x[:-2])
    df_cat.rename(
        columns = cat_row_cols,
        inplace = True
    )
    for col in df_cat.columns:
        df_cat[col] = df_cat[col].astype(str).replace('.*-', '', regex=True)
        df_cat[col] = df_cat[col].astype(int)

    # remove the child_alone columns as all values equal 0.
    df_cat = df_cat.drop(columns = 'child_alone')
    # remove categories (now one-hot encoded).
    df_clean = df_clean.drop(columns = 'categories')
    df_concat = pd.concat([df_clean, df_cat], axis=1)
    
    # convert all the values=2 in the related column to 1. Suppose this is an error.
    df_concat.related.replace(2, 1, inplace=True)

    # remove duplicates

    df_concat.drop_duplicates(inplace=True)
    return df_concat


def save_data(df, database_filename):
    """
    Save the merged dataframe to the database.
    Input: df: pd.DataFrame
           database_filename: str
    -----------
    Output: None
    """
    root_path = os.getcwd()
    db_path = 'sqlite:///' + str(root_path) + '/' + database_filename
    engine = create_engine(db_path)
    # df.to_sql(db_path, engine, index=False, if_exists='replace')
    df.to_sql('disaster_table', engine, index=False, if_exists='replace')
    return


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()