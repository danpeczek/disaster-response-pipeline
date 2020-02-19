import os
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath, index_col=False)
    categories = pd.read_csv(categories_filepath, index_col=False)
    if messages.shape[0] != categories.shape[0]:
        print("Critical Error: messages and categories have different number of rows")
        return None
    df = pd.concat([messages, categories], join='inner', axis=1)
    return df


def clean_data(df):
    # Remove super column 'id'
    df = df.drop(columns=['id'])

    # Clean 'categories' column
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.iloc[0]
    category_colnames = [r.split('-')[0] for r in row]
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda val: val.split('-')[1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], downcast='integer')

    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories], axis=1)

    # Related column contain three categories, changed 2 to 1 for binary classification.
    df[(df['related'] == 2)] = 1

    # Remove duplicates in message column
    df = df.drop_duplicates(['message'])
    return df


def save_data(df, database_filepath):
    database_string = "sqlite:///"+database_filepath
    db_name = database_filepath.split(os.path.sep)[-1]
    db_name = db_name.split('.')[0]
    engine = create_engine(database_string)
    df.to_sql(db_name, engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')

if __name__ == '__main__':
    main()