import pandas as pd
import numpy as np

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
        df_raw['datetime_str'] = df_raw['Date'] + ' ' + df_raw['Time']
        df_raw['datetime'] = pd.to_datetime(df_raw['datetime_str'])
        df_raw.drop(columns=['Date', 'Time', 'datetime_str'], inplace=True)

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

    # Create a new column 'is_weekend' based on day of the week
    df_cleaned.loc[:, 'is_weekend'] = np.where(df_cleaned['datetime'].dt.dayofweek < 5, 'Weekday', 'Weekend')

    return df_cleaned

def calculate_mean_by_weekend(df_cleaned):
    """
    Calculate mean values grouped by 'is_weekend'.

    Parameters:
    - df_cleaned (DataFrame): Cleaned DataFrame with processed features.

    Returns:
    - df_grouped (DataFrame): DataFrame with mean values grouped by 'is_weekend'.
    """
    df_grouped = df_cleaned.groupby('is_weekend').mean()
    return df_grouped
