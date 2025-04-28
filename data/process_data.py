import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets and merge into a single DataFrame.

    Args:
        messages_filepath (str): Path to the messages CSV file.
        categories_filepath (str): Path to the categories CSV file.

    Returns:
        pandas.DataFrame: Merged DataFrame containing messages and categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    Clean the merged DataFrame by splitting categories into separate columns,
    converting values to binary, and removing duplicates.

    Args:
        df (pandas.DataFrame): Merged DataFrame with a 'categories' column.

    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    # Split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    # Extract column names from first row of categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames

    # Convert category values to 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)
        categories[column] = categories[column].apply(lambda x: 1 if x > 1 else x)

    # Drop the original categories column from df and concatenate new category columns
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save the cleaned DataFrame into a SQLite database.

    Args:
        df (pandas.DataFrame): Cleaned DataFrame.
        database_filename (str): Filepath for the SQLite database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


if __name__ == '__main__':
    # Hardcoded file paths
    messages_filepath = 'data/disaster_messages.csv'
    categories_filepath = 'data/disaster_categories.csv'
    database_filepath = 'data/DisasterResponse.db'

    print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
    df = load_data(messages_filepath, categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)

    print(f'Saving data...\n    DATABASE: {database_filepath}')
    save_data(df, database_filepath)

    print('Cleaned data saved to database!')
