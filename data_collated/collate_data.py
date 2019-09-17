import os
import sys
import sqlite3
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def check_data_exits(ml_settings):
    """A function to check if data file exists."""
    dir = ml_settings.DATA_DIR
    if not os.path.exists( os.path.join(dir, ml_settings.FIRE_DATABASE) ):
        raise Exception("Missing database for US wildfires.")
    if not os.path.exists( os.path.join(dir, ml_settings.CLIMATE_DATA) ):
        raise Exception("Missing historical climate data.")
    if not os.path.exists( os.path.join(dir, ml_settings.STOCK_DATA) ):
        raise Exception("Missing historical stock data.")

def create_connection(db_file):
    """A function to create a connection to a SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Exception as e:
        print(e)
        
    return conn
    
def load_data_into_df(ml_settings):
    """
    A function to load the wildfire, climate, and stocks data into 
    individual dataframes.
    """
    # Specify paths to data.
    dir = ml_settings.DATA_DIR
    FIRE_DATABASE = os.path.join(dir, ml_settings.FIRE_DATABASE)
    CLIMATE_DATA = os.path.join(dir, ml_settings.CLIMATE_DATA)
    STOCK_DATA = os.path.join(dir, ml_settings.STOCK_DATA)
    
    # Create connection for US wildfire SQLite database.
    conn = create_connection(FIRE_DATABASE)
    query = ("SELECT FIRE_YEAR, STAT_CAUSE_DESCR, LATITUDE, LONGITUDE, STATE, "
                    "DISCOVERY_DATE, DISCOVERY_DOY, DISCOVERY_TIME, CONT_DATE, "
                    "CONT_TIME, FIRE_SIZE, FIRE_SIZE_CLASS, OWNER_CODE "
                    "FROM 'Fires'")

    # Load data.
    print('Attempting to Load Raw Data for Collation:')
    try:
        fires_df = pd.read_sql_query(query, conn)
        print('\tSuccess: US Wildfire data loaded into dataframe.')
    except Exception as e:
        print('Data load error: ',e)
        sys.exit(1)

    try:
        climate_df = pd.read_csv(CLIMATE_DATA)
        print('\tSuccess: Historical climate data loaded into dataframe.')
    except Exception as e:
        print('Data load error: ',e)
        sys.exit(1)
        
    try:
        stock_df = pd.read_csv(STOCK_DATA)
        print('\tSuccess: Historical stock data loaded into dataframe.\n')
    except Exception as e:
        print('Data load error: ',e)
        sys.exit(1)

    return fires_df, climate_df, stock_df

def format_gregorian(df, date_cols):
    """A function to change format of dates from Julian to Gregorian"""
    for col in date_cols:
        df[col] = pd.to_datetime(
                        df[col] - pd.Timestamp(0).to_julian_date(), unit='D')
        
    return df

def set_lat_and_long(cat):
    """
    A function to reformat the latitude and longitude to numeric values 
    depending on the cardinal direction.
    """
    direction = cat[-1]
    coord = np.nan
    
    if direction == 'N' or direction == 'E':
        coord = float(cat[:-1])
    elif direction == 'S'or direction == 'W':
        coord = -1 * float(cat[:-1])
        
    return coord

def add_nearest_city(fire_df, city_df):
    """A function to add the nearest city that contains climate data."""
    nA = np.array(list(zip(fire_df['LATITUDE'], fire_df['LONGITUDE'])))
    nB = np.array(list(zip(city_df['Latitude'], city_df['Longitude'])))
    btree = cKDTree(nB)
    
    dist, idx = btree.query(nA, k=1)
    df = pd.concat(
        [fire_df, city_df.loc[idx, city_df.columns == 'City'].reset_index(),
         pd.Series(dist, name='DIST_TO_MAJOR_CITY')], axis=1)
    return df

def add_months_and_dows(df, date_cols):
    """
    A function to add dataframe columns for the month and day of the week
    for each datetime column listed in date_cols.
    """
    for col in date_cols:
        prefix = ""
        if len(col) >= 4:
            prefix = col[:4] + '_'
            
        df[prefix + 'MONTH'] = pd.DatetimeIndex(df[col]).month
        df[prefix + 'DAY_OF_WEEK'] = df[col].dt.weekday
    
    return df

def set_label(cat):
    """
    A function to set label to 0 for fires stared by natural causes,
    or 1 for human-induced wildfires.
    """
    natural = ['Lightning']
    human = ['Fireworks','Smoking','Children','Campfire',
             'Equipment Use','Debris Burning','Structure',
             'Powerline','Railroad','Arson','Missing/Undefined',
             'Miscellaneous'
            ]
    
    if cat in natural:
        cause = 0
    else:
        cause = 1
    return cause

def process_climate(climate_df, ml_settings):
    """A function to process the climate data for combination."""
    print('\tProcessing climate data.')

    # Keep only the data in the US and that occurs from 1992 on.
    print('\t\tSelecting climate data for US only.')
    climate_df = climate_df[climate_df['Country'] == 'United States']
    climate_df = climate_df.dropna(axis=0, how='any')

    print('\t\tFormatting date fields.')
    climate_df['dt'] = pd.to_datetime(climate_df['dt'], infer_datetime_format=True)
    climate_df = climate_df.loc[climate_df['dt'] >= ml_settings.start]

    # Reformat the latitude and longitude into numerical values.
    print('\t\tReformatting latitude and longitude into numerical values.')
    climate_df['Latitude'] = climate_df['Latitude'].apply(lambda x: set_lat_and_long(x))
    climate_df['Longitude'] = climate_df['Longitude'].apply(lambda x: set_lat_and_long(x))

    return climate_df

def process_stocks(stock_df, ml_settings):
    """A function to process the stock data for combination."""
    print('\tProcessing stock data.')

    # Format dates and only keep entries from 1992 on.
    print('\t\tFormatting date fields.')
    stock_df['date'] = pd.to_datetime(stock_df['date'], infer_datetime_format=True)
    stock_df = stock_df.loc[stock_df['date'] >= ml_settings.start]

    # Keep only a representaive subset of the stocks.
    print('\t\tKeeping only subset of stocks.')
    stock_df = stock_df[stock_df['ticker'].isin(ml_settings.stocks)]

    # Pivot dataframe to include adjusted closing prices for 
    # the listed stocks as columns.
    stock_df = stock_df.pivot(index='date', columns='ticker', values='adj_close').reset_index()
    stock_df.rename(columns = {'date':'DISCOVERY_DATE'}, inplace = True)

    return stock_df

def process_fires(fires_df, climate_df):
    """
    A function to process the US wildfire data.
    """
    print('\tProcessing US wildfire data.')

    # Change date fields into Gregorian format.
    print('\t\tTransforming date fields into Gregorian format.')
    date_cols = ['DISCOVERY_DATE', 'CONT_DATE']    
    fires_df = format_gregorian(fires_df, date_cols)

    # Add columns that contain the length of the burning as well as days since
    # last fire near the same nearest major city.
    print('\t\tAdding fire duration.')
    fires_df['FIRE_LENGTH'] = fires_df['CONT_DATE'].dt.dayofyear - fires_df['DISCOVERY_DOY']

    # Add the nearest major city to fire.
    print('\t\tAdding location of nearest (major) city to fire.')
    city_loc_df = climate_df[['City','Latitude','Longitude']].drop_duplicates(
                                                    subset='City').reset_index()
    fires_df = add_nearest_city(fires_df, city_loc_df)

    # Add the start of the nearest month to dataframe for merging 
    # with climate data.
    print('\t\tAdding first day of the month when fire occured (to merge).')
    fires_df['dt'] = fires_df['DISCOVERY_DATE'] + pd.offsets.Day() - pd.offsets.MonthBegin()

    # Add columns corresponding to the month and day of the week for Discovery
    # and Containment dates.
    print('\t\tAdding month and day of the week fire occured.')
    fires_df = add_months_and_dows(fires_df, date_cols)

    # Group labels according to natural or human caused fire.
    print('\t\tCreating label for natural or human caused fire.')
    fires_df['STAT_CAUSE_DESCR'] = fires_df['STAT_CAUSE_DESCR'].astype('category')
    fires_df['STAT_CAUSE_DESCR'] = fires_df['STAT_CAUSE_DESCR'].apply(lambda x: set_label(x))

    return fires_df

def merge_data(fires_df, climate_df, stock_df):
    """
    A function to merge the US wildfire, climate, and stock data.
    """
    print('\tMerging dataframes.')
    # Merge wildfire data and climate data.
    print('\t\tMerging climate data.')
    fires_df = pd.merge(fires_df, 
            climate_df.drop(['Country','Latitude','Longitude'], axis=1),
            on=['dt', 'City'])
    fires_df = fires_df.drop(['index','dt'], axis=1)

    # Merge combined data with stock data.
    print('\t\tMerging stock data.')
    fires_df = pd.merge(fires_df, stock_df, on='DISCOVERY_DATE')

    return fires_df

def collate_data(ml_settings):
    """
    A function to combine the three data sets into a single SQLite database.
    """
    # Check if the combined data file has already been created.
    PATH = os.path.join(ml_settings.DATA_COL_DIR, ml_settings.COMBINED_DATA)
    if os.path.isfile(PATH):
        print("Data files have already been collated, proceeding with analysis.")
        return
    else:
        print('Data files need to first be processed and then combined (~1-2min).\n')

    try:
        check_data_exits(ml_settings)
    except Exception as e:
        print(e)

    # Load dataframes for wildfire, climate, and stock data.
    fires_df, climate_df, stock_df = load_data_into_df(ml_settings)

    # Process and merge the data.
    print('Processing individual data sets, then combining:')
    climate_df = process_climate(climate_df, ml_settings)
    stock_df = process_stocks(stock_df, ml_settings)
    fires_df = process_fires(fires_df, climate_df)

    merged_df = merge_data(fires_df, climate_df, stock_df)

    # Save combined data.
    print('\nSaving combined data.')
    try:
        conn = sqlite3.connect(PATH)
        merged_df.to_sql("Fires", conn, if_exists="replace")
    except Exception as e:
        print(e)