import os
import sqlite3
import numpy as np
import pandas as pd

def create_connection(db_file):
    """A function to create a connection to a SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Exception as e:
        print(e)
        
    return conn

def load_merged_data(PATH):
    """A function to load the collated data into a dataframe."""
    print('\nReloading data.')
    conn = create_connection(PATH)
    query = ("SELECT * FROM 'Fires'")

    try:
        fires_df = pd.read_sql_query(query, conn)
        print('Compiled data has been restored.\n')
    except Exception as e:
        print('Data load error: ',e)
    
    return fires_df

def change_obj_dtype(df, objects, to_dtype):
    """A function to change the datatype of multiple columns in a dataframe."""
    if to_dtype == 'int':
        for obj in objects:
            df[obj] = pd.to_numeric(df[obj], errors='coerce', downcast='integer')
    elif to_dtype == 'float':
        for obj in objects:
            df[obj] = pd.to_numeric(df[obj], errors='coerce', downcast='float')        
    elif to_dtype == 'categories':
        for obj in objects:
            df[obj] = df[obj].astype('category')
    elif to_dtype == 'date':
        for obj in objects:
            df[obj] = pd.to_datetime(df[obj], infer_datetime_format=True)
    else:
        raise Exception("The target dtype is not supported by change_obj_dtype.")
    
    return df

def partition(ml_settings, df):
    """A function to partition the data into training, validation, and test sets."""
    print('Separating into training, validation, and test sets.')
    print('\tTraining Set Ratio: {0:.2f}'.format(
                    1 - ml_settings.val_set_ratio - ml_settings.test_set_ratio))
    print('\tValidation Set Ratio: {0:.2f}'.format(ml_settings.val_set_ratio))
    print('\tTest Set Ratio: {0:.2f}\n'.format(ml_settings.test_set_ratio))

    # Sort dataframe chronoligically.
    df = df.sort_values(by='DISCOVERY_DATE')

    # Split according to predefined settings.
    val_set_size = int(len(df) * ml_settings.val_set_ratio)
    test_set_size = int(len(df) * ml_settings.test_set_ratio)

    train_val_bound = int(len(df) - val_set_size - test_set_size)
    val_test_bound = int(len(df) - test_set_size)

    train_set = df.iloc[:train_val_bound]
    val_set = df.iloc[train_val_bound:val_test_bound]
    test_set = df.iloc[val_test_bound:]

    # Shuffle the individual groups.
    if not ml_settings.hp_search:
        train_set = train_set.sample(frac=1, random_state=42).reset_index(drop=True)
        val_set = val_set.sample(frac=1, random_state=42).reset_index(drop=True)
        test_set = test_set.sample(frac=1, random_state=42).reset_index(drop=True)

    return train_set, val_set, test_set

def separate_labels(dataset):
    """A function to separate dataset into feature matrix and label vector."""
    X = dataset.drop('STAT_CAUSE_DESCR', axis=1)
    y = dataset['STAT_CAUSE_DESCR']
    return X, y

def separate_time_data(ml_settings):
    """A function to prepare the compiled data for learning."""

    # Load the data.
    PATH = os.path.join(ml_settings.DATA_COL_DIR, ml_settings.COMBINED_DATA)
    fires_df = load_merged_data(PATH)

    # Format dates into timestamps and objects to appropriate integers
    # or categories
    obj_to_date = ['DISCOVERY_DATE', 'CONT_DATE'] 
    obj_to_int = ['DISCOVERY_TIME', 'CONT_TIME']
    obj_to_cat = ['STATE', 'OWNER_CODE', 'FIRE_SIZE_CLASS', 'STAT_CAUSE_DESCR']

    try:
        fires_df = change_obj_dtype(fires_df, obj_to_date, 'date')
        fires_df = change_obj_dtype(fires_df, obj_to_int, 'int')
        fires_df = change_obj_dtype(fires_df, obj_to_cat, 'categories')
    except Exception as e:
        print(e)

    # If prototyping, use small subset of data.
    if ml_settings.prototyping:
        fires_df = fires_df.iloc[:ml_settings.sample_size]
        print('Using Reduced Data Set for Prototyping: ')
        print('\tRetaining {0} data points.\n'.format(ml_settings.sample_size))

    # Split into training, validation, and test sets.
    train_set, val_set, test_set = partition(ml_settings, fires_df)

    # Separate each set into feature matrices and label vectors.
    X_train, y_train = separate_labels(train_set)
    X_val, y_val = separate_labels(val_set)
    X_test, y_test = separate_labels(test_set)

    return X_train, X_val, X_test, y_train, y_val, y_test