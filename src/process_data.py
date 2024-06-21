import pandas as pd
import numpy as np
from scipy.spatial import distance

def load_and_clean_data(filepath):
    """
    Load and clean the occupancy estimation data.

    Parameters:
    - filepath (str): Path to the CSV file containing the data.

    Returns:
    - df_cleaned (DataFrame): Cleaned DataFrame with processed features.
    """
    # Load the raw data
    df_raw = pd.read_csv(filepath)
    
    # Combine date and time columns into a datetime column
    if 'Date' in df_raw.columns and 'Time' in df_raw.columns:
        df_raw['datetime'] = pd.to_datetime(df_raw['Date'] + ' ' + df_raw['Time'])
        df_raw.drop(columns=['Date', 'Time'], inplace=True)

    # Reorder columns to have 'datetime' as the first column
    cols = list(df_raw.columns)
    if 'datetime' in cols:
        cols.remove('datetime')
    df_cleaned = df_raw[['datetime'] + cols].copy()  # Make a copy to avoid modifying the original df_raw

    # Calculate mean values for temperature, light, sound, CO2, CO2 Slope, and PIR
    temp_mean = df_cleaned.loc[:, 'S1_Temp':'S4_Temp'].mean(axis=1)
    light_mean = df_cleaned.loc[:, 'S1_Light':'S4_Light'].mean(axis=1)
    sound_mean = df_cleaned.loc[:, 'S1_Sound':'S4_Sound'].mean(axis=1)
    co2_mean = df_cleaned['S5_CO2'].mean()
    co2_slope_mean = df_cleaned['S5_CO2_Slope'].mean()
    pir_mean = df_cleaned.loc[:, 'S6_PIR':'S7_PIR'].mean(axis=1)

    # Append mean values as new columns using .loc to modify the original DataFrame
    df_cleaned.loc[:, 'Mean_Temperature'] = temp_mean
    df_cleaned.loc[:, 'Mean_light'] = light_mean
    df_cleaned.loc[:, 'Mean_sound'] = sound_mean
    df_cleaned.loc[:, 'Mean_CO2'] = co2_mean
    df_cleaned.loc[:, 'Mean_CO2_Slope'] = co2_slope_mean
    df_cleaned.loc[:, 'Mean_PIR'] = pir_mean

    return df_cleaned

def calculate_mean_by_weekend(df_cleaned):
    """
    Calculate mean values grouped by 'is_weekend'.

    Parameters:
    - df_cleaned (DataFrame): Cleaned DataFrame with processed features.

    Returns:
    - df_grouped (DataFrame): DataFrame with mean values grouped by 'is_weekend'.
    """
    # Create a new column 'is_weekend' based on day of the week
    df_cleaned.loc[:, 'is_weekend'] = np.where(df_cleaned['datetime'].dt.dayofweek < 5, 'Weekday', 'Weekend')
    df_grouped = df_cleaned.groupby('is_weekend').mean()
    return df_grouped


def remove_outliers(df_raw):
    sensors = ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light', 'S2_Light',
           'S3_Light', 'S4_Light', 'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound',
           'S5_CO2', 'S5_CO2_Slope', 'S6_PIR', 'S7_PIR']

    # Select only the sensor columns
    df_sensors = df_raw[sensors]


    # Calculate the mean and covariance of the sensor data
    mean = df_sensors.mean()
    cov = df_sensors.cov()

    # Calculate the Mahalanobis distance of each data point
    distances = df_sensors.apply(lambda row: distance.mahalanobis(row, mean, np.linalg.inv(cov)), axis=1)

    # Define a threshold for the Mahalanobis distance
    threshold = np.percentile(distances, 2)  # Adjust this value as needed

    # Create a mask for the outliers
    mask = distances < threshold

    # Remove the outliers from the sensor columns
    df_cleaned = df_raw.copy()
    df_cleaned.loc[~mask, sensors] = np.nan

    return df_cleaned
