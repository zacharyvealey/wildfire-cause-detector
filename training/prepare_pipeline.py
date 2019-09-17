from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from .convert_to_cycle import ConvertToCyclical

def prep_pipeline(ml_settings):
    """
    A function to create a pipeline for preparing the post-processed
    data for training.
    """
    print('Creating machine learning pipeline.')
    print('\tImputing and standardizing values.')
    print('\tPerforming sine and cosine transformation for cyclical features.\n')

    cat_attribs = ml_settings.cat_attribs
    num_attribs = ml_settings.num_attribs
    cycle_cols = ml_settings.cycle_cols

    # Full data processing pipeline.
    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ("std_scaler", StandardScaler()),
        ])

    cycle_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('cycle', ConvertToCyclical(ml_settings)),
            ("std_scaler", StandardScaler()),
        ])

    preparation = ColumnTransformer([
            ("nums", num_pipeline, num_attribs),
            ("cycles", cycle_pipeline, cycle_cols),
            ("cats", OneHotEncoder(categories='auto', handle_unknown='ignore',sparse=False), cat_attribs),
        ])

    return preparation