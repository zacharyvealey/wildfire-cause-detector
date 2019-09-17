import pandas as pd
import tensorflow as tf

class Settings():
    """A class to store all settings for machine learning project."""

    def __init__(self):
        """Initialize machine learning settings."""
        # File settings and locations.
        self.DATA_DIR = 'data'
        self.DATA_COL_DIR = 'data_collated'

        self.FIRE_DATABASE = 'FPA_FOD_20170508.sqlite'
        self.CLIMATE_DATA = 'GlobalLandTemperaturesByCity.csv'
        self.STOCK_DATA = 'historical_stock_prices.csv'
        self.COMBINED_DATA = 'combined_data.db'

        self.MODEL_PATH = 'models/dnn_wildfires.ckpt'

        # Setting to use reduced data for prototyping purposes.
        self.prototyping = False
        self.sample_size = 80000

        # Start date of data
        self.start = pd.to_datetime('1992-01-01')

        # Stocks in stock data to keep for analysis.
        self.stocks = ['MSFT', 'AAPL', 'GE', 'JNJ', 'JPM', 'PG']

        # Settings for validation and test set partitioning.
        self.val_set_ratio = 0.15
        self.test_set_ratio = 0.15

        # Separation of features for pipeline preparation 
        self.cat_attribs = ['STATE', 'FIRE_SIZE_CLASS', 'OWNER_CODE', 'City']
        self.num_attribs = ['FIRE_YEAR', 'LATITUDE', 'LONGITUDE', 'FIRE_SIZE', 
                    'FIRE_LENGTH', 'DIST_TO_MAJOR_CITY', 'AverageTemperature',
                    'AverageTemperatureUncertainty', 'AAPL', 'GE', 'JNJ', 
                    'JPM', 'MSFT', 'PG']
        self.cycle_cols = ['DISC_MONTH', 'DISC_DAY_OF_WEEK', 'DISCOVERY_TIME', 
                    'DISCOVERY_DOY', 'CONT_MONTH', 'CONT_DAY_OF_WEEK',
                    'CONT_TIME']

        # Define the ranges of the cycles in cycle_cols and whether any offset for
        # zero-indexing is needed (i.e., 'DISC_MONTH' cycles over a 12 month period
        # and the months need an offset of one to start the indicies at 0 for Jan.).
        self.cycle_ranges = [12, 7, 2400, 365, 12, 7, 2400]
        self.cycle_offsets = [1, 0, 0, 1, 1, 0, 0]

        # Parameters for deep learning model determined from randomized 
        # hyperparameter search.
        self.n_hidden_layers = 4
        self.n_neurons = 200
        self.batch_size = 500
        self.batch_norm_momentum = 0.999
        self.dropout_rate = 0.4
        self.learning_rate = 0.01
        self.activation = tf.nn.elu

        # Hyperparameter settings .
        self.hp_search = False






