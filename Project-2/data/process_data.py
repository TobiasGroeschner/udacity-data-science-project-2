import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:

    """
    Loads two datasets and merges them

    Parameters:
    messages_iflepath: messages.csv filepath
    categories_filepath: categories.csv filepath

    Returns:
    df: dataframe containing messages_filepath and categories_filepath merged

    """

    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, on="id")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    """
    Cleans the dataset

    Parameters:
    df: dataframe containing messages_filepath and categories_filepath merged

    Returns:
    df: Cleaned dataframe

    """

    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(pat=";", expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]
    # extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x.split("-")[0])
    # define categories.columns
    categories.columns = category_colnames

    # iterate to set each value to be the last character of the string
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop(columns="categories")

    # concat the
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    engine = create_engine("sqlite:///" + database_filename)

    # table name
    df.to_sql("DisasterResponse.db", engine, index=False)


def main():

    messages_filepath, categories_filepath, database_filepath = (
        "/home/tobias_groeschner/projects/DataScience/Project-2/data/disaster_messages.csv",
        "/home/tobias_groeschner/projects/DataScience/Project-2/data/disaster_categories.csv",
        "disaster_response.db",
    )

    print(
        "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
            messages_filepath, categories_filepath
        )
    )

    df = load_data(messages_filepath, categories_filepath)

    print("Cleaning data...")
    df = clean_data(df)

    print("Saving data...\n    DATABASE: {}".format(database_filepath))
    save_data(df, database_filepath)

    print("Cleaned data saved to database!")


if __name__ == "__main__":
    main()
