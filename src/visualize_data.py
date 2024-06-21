import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pandas.plotting import autocorrelation_plot
from scipy.spatial import distance

# Your existing code ...

def save_plot(fig, filename, output_dir):
    """
    Save the given figure to the specified directory with the given filename.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(os.path.join(output_dir, filename))

def with_outliers(df_cleaned, output_dir):
    """
    Plot boxplots of sensor readings with outliers.
    """
    sensors = ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light', 'S2_Light',
           'S3_Light', 'S4_Light', 'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound',
           'S5_CO2', 'S5_CO2_Slope', 'S6_PIR', 'S7_PIR']

    fig, axs = plt.subplots(len(sensors) // 2, 2, figsize=(12, 12))

    plt.suptitle("With Outliers")
    plt.tight_layout()
    save_plot(fig, 'with_outliers.png', output_dir)


def without_outliers(df_raw, sensors, percentile, output_dir):
    """
    Remove outliers from sensor readings using the Mahalanobis distance and save the cleaned DataFrame to a CSV file.
    """
     # Calculate the mean and covariance of the sensor data
    sensors = ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light', 'S2_Light',
       'S3_Light', 'S4_Light', 'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound',
       'S5_CO2', 'S5_CO2_Slope', 'S6_PIR', 'S7_PIR']

# Select only the sensor columns
    df_sensors = df_raw[sensors]
    mean = df_sensors[sensors].mean()
    cov = df_sensors[sensors].cov()

    # Calculate the Mahalanobis distance of each data point
    distances = df_sensors[sensors].apply(lambda row: distance.mahalanobis(row, mean, np.linalg.inv(cov)), axis=1)

    # Define a threshold for the Mahalanobis distance
    threshold = np.percentile(distances, 2)

    # Create a mask for the outliers
    mask = distances < threshold

    # Remove the outliers from the sensor columns
    df_cleaned = df_raw.copy()
    df_cleaned.loc[~mask, sensors] = np.nan

    # Plot boxplot for visualization
    fig, axs = plt.subplots(len(sensors) // 2, 2, figsize=(12, 12))

    for i, sensor in enumerate(sensors):
        row = i // 2
        col = i % 2
        axs[row, col].boxplot(df_cleaned[sensor].dropna())
        axs[row, col].set_title(f"Boxplot of {sensor}")

    plt.suptitle("Without Outliers")
    plt.tight_layout()
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'without_outliers.png'))


def weekday_vs_weekend(df_grouped, output_dir):
    # Plot the grouped data
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Define the sensor readings and their titles
    sensor_readings = ['Mean_Temperature', 'Mean_light', 'Mean_sound', 'Mean_CO2', 'Mean_CO2_Slope', 'Mean_PIR']
    titles = ['Mean Temperature', 'Mean Light', 'Mean Sound', 'Mean CO2', 'Mean CO2 Slope', 'Mean PIR']

    # Iterate over the sensor readings and plot them
    for i, (sensor_reading, title) in enumerate(zip(sensor_readings, titles)):
        row = i // 3
        col = i % 3
        axs[row, col].bar(df_grouped.index, df_grouped[sensor_reading])
        axs[row, col].set_title(title)
        axs[row, col].set_xlabel('Weekend')
        axs[row, col].set_ylabel(title)

    # Adjust the layout
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'weekday_vs_weekend.png'))


def sensor_readings_weekend_vs_weekday(df_grouped, sensor, output_dir):
    """
    Plot the mean sensor readings for weekend and weekday.
    """
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Define the sensor readings and their titles
    sensor_readings = ['Mean_Temperature', 'Mean_light', 'Mean_sound', 'Mean_CO2', 'Mean_CO2_Slope', 'Mean_PIR']
    titles = ['Mean Temperature', 'Mean Light', 'Mean Sound', 'Mean CO2', 'Mean CO2 Slope', 'Mean PIR']

    # Iterate over the sensor readings and plot them
    for i, (sensor_reading, title) in enumerate(zip(sensor_readings, titles)):
        row = i // 3
        col = i % 3
        axs[row, col].bar(df_grouped.index, df_grouped[sensor_reading])
        axs[row, col].set_title(title)
        axs[row, col].set_xlabel('Weekend')
        axs[row, col].set_ylabel(title)

    # Adjust the layout
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'sensor_readings_weekend_vs_weekday.png'))

def plot_feature_distribution(df_cleaned, output_dir):
    """
    Plot the distribution of all features in the dataframe.
    """
    cols = df_cleaned.columns.difference(['datetime', 'is_weekend'])
    fig, axs = plt.subplots(5, 5, figsize=(20, 20))
    for i, col in enumerate(cols):
        ax = axs[i // 5, i % 5]
        sns.histplot(df_cleaned[col], kde=True, ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    save_plot(fig, 'feature_distribution.png', output_dir)

def plot_correlation_matrix(df_cleaned, output_dir):
    """
    Plot the correlation matrix of numeric features in the dataframe.
    """
    # Select numeric columns for correlation matrix
    numeric_cols = df_cleaned.select_dtypes(include=np.number).columns
    
    # Exclude 'datetime' and 'is_weekend' from numeric columns
    numeric_cols = numeric_cols.difference(['datetime', 'is_weekend'])
    
    corr = df_cleaned[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='viridis', ax=ax)
    ax.set_title('Correlation Matrix')
    save_plot(fig, 'correlation_matrix.png', output_dir)

def plot_time_series(df_cleaned, output_dir):
    """
    Plot the time series of all features in the dataframe.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    df_cleaned.set_index('datetime').plot(ax=ax)
    ax.set_title('Time Series Plot')
    plt.tight_layout()
    save_plot(fig, 'time_series.png', output_dir)

def plot_boxplots(df_cleaned, output_dir):
    """
    Plot boxplots of each feature against the room occupancy count.
    """
    features = df_cleaned.columns.difference(['datetime', 'is_weekend'])
    ncols = 2
    nrows = (len(features) + ncols - 1) // ncols  # Calculate number of rows based on features
    fig, axs = plt.subplots(nrows, ncols, figsize=(14, 8 * nrows))
    
    # Flatten axs if nrows == 1 to handle single row case
    if nrows == 1:
        axs = axs.reshape(1, -1)
    
    for i, feature in enumerate(features):
        row = i // ncols
        col = i % ncols
        sns.boxplot(x='Room_Occupancy_Count', y=feature, data=df_cleaned, palette='coolwarm', ax=axs[row, col])
        axs[row, col].set_title(f'Boxplot of {feature} vs Room_Occupancy_Count')
        axs[row, col].set_xticks(axs[row, col].get_xticks())  # Set explicit tick positions
        axs[row, col].set_xticklabels(axs[row, col].get_xticklabels(), rotation=45)  # Set tick labels with rotation
    
    plt.tight_layout()
    save_plot(fig, 'boxplots.png', output_dir)

def plot_autocorrelation(df_cleaned, output_dir):
    """
    Plot the autocorrelation of all mean variables and save it.
    """
    
    # Calculate autocorrelation of all mean variables
    mean_columns = [col for col in df_cleaned.columns if 'Mean' in col]

    plt.figure(figsize=(10,5))

    for col in mean_columns:
        autocorrelation_plot(df_cleaned[col].dropna().tolist(), label=col)

    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Plot of Mean Variables')
    plt.legend(loc='upper right')

    # Save the plot to output_dir
    fig = plt.gcf()  # Get the current figure
    save_plot(fig, 'autocorrelation.png', output_dir)

def plot_pairplot(df_cleaned, output_dir):
    """
    Plot pair plots of temperature readings from sensors S1, S2, S3, and S4.
    """
    fig = sns.pairplot(df_cleaned[['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp']])
    fig.fig.suptitle('Pair Plot of Temperature Readings from Sensors S1, S2, S3, and S4', y=1.02)
    save_plot(fig.fig, 'pairplot.png', output_dir)

def plot_mean_sensor_readings_vs_occupancy(df_cleaned, output_dir):
    """
    Plot mean sensor readings against room occupancy count, with polynomial fit.
    """
    features = ['Mean_Temperature', 'Mean_light', 'Mean_sound', 'Mean_CO2']

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    for i, feature in enumerate(features):
        row = i // 2
        col = i % 2

        # Extract data
        x = df_cleaned[feature].values
        y = df_cleaned['Room_Occupancy_Count'].values

        # Remove NaNs and ensure alignment
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]

        if len(x) < 3:
            # If fewer than 3 data points are available, skip fitting
            axs[row, col].scatter(x, y)
            axs[row, col].set_title(f'Room Occupancy vs {feature}')
            axs[row, col].set_xlabel(feature)
            axs[row, col].set_ylabel('Room Occupancy Count')
        else:
            # Fit a polynomial using curve_fit (which handles warnings better than np.polyfit)
            def polynomial_func(x, *coefficients):
                return np.polyval(coefficients, x)

            # Initial guess for coefficients (degree=2 polynomial)
            p0 = np.ones(3)

            try:
                coefficients, _ = curve_fit(polynomial_func, x, y, p0=p0)
                x_new = np.linspace(x.min(), x.max(), 500)
                y_fit = polynomial_func(x_new, *coefficients)

                axs[row, col].scatter(x, y)
                axs[row, col].plot(x_new, y_fit, 'r-', label='Polynomial Fit')
                axs[row, col].set_title(f'Room Occupancy vs {feature}')
                axs[row, col].set_xlabel(feature)
                axs[row, col].set_ylabel('Room Occupancy Count')
                axs[row, col].legend()

            except RuntimeError:
                # Curve fitting failed, fall back to simple scatter plot
                axs[row, col].scatter(x, y)
                axs[row, col].set_title(f'Room Occupancy vs {feature}')
                axs[row, col].set_xlabel(feature)
                axs[row, col].set_ylabel('Room Occupancy Count')

    plt.tight_layout()
    save_plot(fig, 'mean_sensor_readings_vs_occupancy.png', output_dir)
 

def plot_correlation_matrix_sensors(df_cleaned, output_dir):
    """
    Plot the correlation matrix of the sensor readings.
    """
    sensors_cols = ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp',
                    'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light',
                    'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound']
    
    corr = df_cleaned[sensors_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='viridis', ax=ax)
    ax.set_title('Correlation Matrix of Sensor Readings')
    save_plot(fig, 'correlation_matrix_sensors.png', output_dir)
