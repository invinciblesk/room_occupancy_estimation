
import os
import argparse
from src.process_data import load_and_clean_data, calculate_mean_by_weekend, remove_outliers
from src.visualize_data import (
    with_outliers,
    without_outliers,
    plot_feature_distribution, 
    plot_correlation_matrix, 
    plot_time_series, 
    plot_boxplots, 
    plot_autocorrelation, 
    plot_pairplot, 
    plot_mean_sensor_readings_vs_occupancy, 
    plot_correlation_matrix_sensors
)

def main(data_file, output_dir):
    # Load and clean data
    df_raw = load_and_clean_data(data_file)

    # Define the sensors
    sensors = ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light', 'S2_Light',
               'S3_Light', 'S4_Light', 'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound',
               'S5_CO2', 'S5_CO2_Slope', 'S6_PIR', 'S7_PIR']

    # Define the percentile for outlier detection
    percentile = 2  # Adjust this value as needed

    # Remove outliers
    df_cleaned = remove_outliers(df_raw)

    # Calculate mean values grouped by 'is_weekend'
    df_grouped = calculate_mean_by_weekend(df_cleaned)
    
    # Generate plots
    with_outliers(df_raw, output_dir)
    without_outliers(df_raw, sensors, percentile, output_dir)
    plot_feature_distribution(df_cleaned, output_dir)
    plot_correlation_matrix(df_cleaned, output_dir)
    plot_time_series(df_cleaned, output_dir)
    plot_boxplots(df_cleaned, output_dir)
    plot_autocorrelation(df_cleaned, output_dir)
    plot_pairplot(df_cleaned, output_dir)
    plot_mean_sensor_readings_vs_occupancy(df_cleaned, output_dir)
    plot_correlation_matrix_sensors(df_cleaned, output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and visualize room occupancy data.')
    parser.add_argument('--data', required=True, help='Path to the data file.')
    parser.add_argument('--output', required=True, help='Directory to save the output plots.')
    
    args = parser.parse_args()
    
    main(args.data, args.output)
