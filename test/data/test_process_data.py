import pandas as pd
from data import process_data
from sqlalchemy import create_engine
import os


def test_load_data():
    # GIVEN
    expected_df = pd.read_csv('test/test_data/test_loaded.csv', index_col=False)
    expected_df = expected_df.rename(columns={"id.1": "id"})
    # WHEN
    df = process_data.load_data('test/test_data/test_data1.csv', 'test/test_data/test_data2.csv')
    # THEN
    assert (df.values == expected_df.values).all()


def test_load_data_with_different_number_of_rows_return_none():
    # WHEN
    df = process_data.load_data('test/test_data/test_data1.csv', 'test/test_data/test_data2_malformed.csv')
    # THEN
    assert df is None, "Load data does not returned None"


def test_clean_data():
    # GIVEN
    expected_df = pd.read_csv('test/test_data/test_cleaned.csv', index_col=False)
    df = pd.read_csv('test/test_data/test_loaded.csv', index_col=False)
    # Need to be consistent with results of pd.concat on two dataframes, which we don't have here
    df = df.rename(columns={"id.1": "id"})

    # WHEN
    df = process_data.clean_data(df)

    # THEN
    assert (df.values == expected_df.values).all()


def test_save_data():
    #GIVEN
    expected_df = pd.read_csv('test/test_data/test_cleaned.csv', index_col=False)

    # WHEN
    df = pd.read_csv('test/test_data/test_cleaned.csv', index_col=False)
    database_path='test'+os.path.sep+'data'+os.path.sep+'test_db_generated.db'
    process_data.save_data(df, database_path)
    engine_df = create_engine("sqlite:///"+database_path)
    df = pd.read_sql('SELECT * FROM test_db_generated', con=engine_df)
    os.remove(database_path)

    # THEN
    assert (df.values == expected_df.values).all()
