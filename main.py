
import os
import argparse
from src.process_data import load_and_clean_data, calculate_mean_by_weekend
from src.visualize_data import (
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
    df_cleaned = load_and_clean_data(data_file)
    
    # Generate plots
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
