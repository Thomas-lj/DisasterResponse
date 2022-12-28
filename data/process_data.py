import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
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

    df_clean = df_clean.drop(columns = 'categories')
    df_concat = pd.concat([df_clean, df_cat], axis=1)

    df_unique = df_concat.drop_duplicates()

    return df_unique


def save_data(df, database_filename):
    engine = create_engine('sqlite:///InsertDatabaseName.db')
    df.to_sql(database_filename, engine, index=False)
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