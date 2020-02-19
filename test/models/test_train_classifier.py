import os
import pandas as pd
from data import process_data
from models import train_classifier

def test_load_data():
    # GIVEN
    expected_df = pd.read_csv('test/test_data/test_cleaned.csv', index_col=False)
    expected_X = expected_df['message']
    expected_Y = expected_df.iloc[:, 3:]
    expected_category_names = expected_Y.columns
    expected_Y = expected_Y.values

    database_path = 'test' + os.path.sep + 'data' + os.path.sep + 'test_db_generated.db'
    process_data.save_data(expected_df, database_path)

    # WHEN
    X, Y, category_names = train_classifier.load_data(database_path)
    os.remove(database_path)
    # THEN
    assert (X == expected_X).all()
    assert Y.shape == expected_Y.shape
    for i in range(0, Y.shape[1]):
        assert (Y[i] == expected_Y[i]).all()
    assert (category_names == expected_category_names).all()

def test_tokenize_data():
    # GIVEN
    expected_text = ['alien', 'are', 'go', 'to', 'kill', 'dr', 'watson', 'watson', 'scream', 'till', '100', 'birthday', 'no', 'it', 's', 'now', 'possibl']
    text = "Aliens are going to kill Dr. Watson. Watson!!! Screaming till 100 birthday (no it's now possible)."
    clean_tokens = train_classifier.tokenize(text)
    print(expected_text)
    print(clean_tokens)
    assert expected_text == clean_tokens
