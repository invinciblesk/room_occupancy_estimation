import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go


# Load the data
df_raw = pd.read_csv('../data/Occupancy_Estimation.csv')

## Process the data

# Combine date and time columns
if 'Date' in df_raw.columns and 'Time' in df_raw.columns:
    df_raw['datetime_str'] = df_raw['Date'] + ' ' + df_raw['Time']

    # Convert the combined column to datetime
    df_raw['datetime'] = pd.to_datetime(df_raw['datetime_str'])

    # Drop the original date and time columns
    df_raw.drop(columns=['Date', 'Time', 'datetime_str'], inplace=True)

# Get a list of all the columns
cols = list(df_raw.columns)

# Remove 'datetime' from the list if it exists
if 'datetime' in cols:
    cols.remove('datetime')

# Reorder the DataFrame to have 'datetime' as the first column
df_cleaned = df_raw[['datetime'] + cols]

# Calculate the mean of the temperature columns
temp_mean = df_cleaned.loc[:, 'S1_Temp':'S4_Temp'].mean(axis=1)

# Calculate the mean of the light columns
light_mean = df_cleaned.loc[:, 'S1_Light':'S4_Light'].mean(axis=1)

# Calculate the mean of the sound columns
sound_mean = df_cleaned.loc[:, 'S1_Sound':'S4_Sound'].mean(axis=1)

# Calculate the mean of the CO2 columns
co2_mean = df_cleaned['S5_CO2'].mean()

# Calculate the mean of the CO2_Slope columns
co2_slope_mean = df_cleaned['S5_CO2_Slope'].mean()

# Calculate the mean of the PIR columns
pir_mean = df_cleaned.loc[:, 'S6_PIR':'S7_PIR'].mean(axis=1)

# Append the new columns at the end of the DataFrame using .loc
df_cleaned.loc[:, 'Mean_Temperature'] = temp_mean
df_cleaned.loc[:, 'Mean_light'] = light_mean
df_cleaned.loc[:, 'Mean_sound'] = sound_mean
df_cleaned.loc[:, 'Mean_CO2'] = co2_mean
df_cleaned.loc[:, 'Mean_CO2_Slope'] = co2_slope_mean
df_cleaned.loc[:, 'Mean_PIR'] = pir_mean

# Create a new column 'is_weekend' that is True if the day of the week is a weekend and False otherwise
df_cleaned.loc[:, 'is_weekend'] = np.where(df_cleaned['datetime'].dt.dayofweek < 5, 'Weekday', 'Weekend')

# Group by the 'is_weekend' column and calculate the mean of the other columns
df_grouped = df_cleaned.groupby('is_weekend').mean()




#Visualize the data

######################################################
# visualize the mean sensor readings
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Define the sensor readings and their titles
sensor_readings = ['Mean_Temperature', 'Mean_light', 'Mean_sound', 'Mean_CO2', 'Mean_CO2_Slope', 'Mean_PIR']
titles = ['Mean Temperature', 'Mean Light', 'Mean Sound', 'Mean CO2', 'Mean CO2 Slope', 'Mean PIR']

# Iterate over the sensor readings and plot them
for i, (sensor_reading, title) in enumerate(zip(sensor_readings, titles)):
    row = i // 3
    col = i % 3
    axs[row, col].plot(df_cleaned.index, df_cleaned[sensor_reading])
    axs[row, col].set_title(title)
    axs[row, col].set_xlabel('Date')
    axs[row, col].set_ylabel(title)
    plt.setp(axs[row, col].xaxis.get_majorticklabels(), rotation=45)

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()

 ###########################################################

#perform a rolling mean analysis on three sensor readings
# Define the rolling window size
window_size = 10

# Create a figure with 1 row and 3 columns
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Define the sensor readings, their colors, and their titles
sensor_readings = ['Mean_Temperature', 'Mean_light', 'Mean_sound']
colors = ['red', 'green', 'blue']
titles = ['Rolling Mean Temperature', 'Rolling Mean Light', 'Rolling Mean Sound']

# Iterate over the sensor readings and plot their rolling means
for i, (sensor_reading, color, title) in enumerate(zip(sensor_readings, colors, titles)):
    axs[i].plot(df_cleaned.index, df_cleaned[sensor_reading].rolling(window_size).mean(), color=color)
    axs[i].set_title(title)
    axs[i].set_xlabel('Date')
    axs[i].set_ylabel(title)
    plt.setp(axs[i].xaxis.get_majorticklabels(), rotation=45)

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()

 ###########################################################


# List of mean columns
mean_columns = ['Mean_Temperature', 'Mean_light', 'Mean_sound', 'Mean_CO2', 'Mean_CO2_Slope', 'Mean_PIR']

fig, axs = plt.subplots(2, 3, figsize=(20, 10))
for i, column in enumerate(mean_columns):
    if column in df_grouped.columns:
        row = i // 3
        col = i % 3
        df_grouped[column].plot(kind='bar', ax=axs[row, col])
        axs[row, col].set_title('Weekday vs Weekend ' + column + ' comparison')
        axs[row, col].set_ylabel(column)
        axs[row, col].set_xlabel('Day Type')
        axs[row, col].set_xticklabels(axs[row, col].get_xticklabels(), rotation=0)  # to keep the x-axis labels vertical

plt.tight_layout()
plt.show()

######################################################

# Set 'datetime' column as index
df_cleaned.set_index('datetime', inplace=True)

# Compare temperature readings from different sensors
df_cleaned[['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp']].plot()

# Add title and labels
plt.title('Temperature Readings from Different Sensors')
plt.xlabel('Time')
plt.ylabel('Temperature')

# Format x-axis labels
plt.gcf().autofmt_xdate()

plt.show()


######################################################


# Create a box plot for the temperature sensors
df_cleaned[['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp']].boxplot()

# Add title and labels
plt.title('Boxplot of Temperature Sensors')
plt.xlabel('Sensor')
plt.ylabel('Temperature')

plt.show()

######################################################

#Autocorrelation plot of the mean temperature
autocorrelation_plot(df_cleaned['Mean_Temperature'])
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Plot of Mean Temperature')
plt.show()

######################################################

#Pairplot of temperature readings from sensors
sns.pairplot(df_cleaned[['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp']])
plt.suptitle('Pair Plot of Temperature Readings from Sensors S1, S2, S3, and S4', y=1.02)  # increased y value
plt.show();

######################################################

# Sensor readings (mean) vs Room Occupancy Count (Scatter plot)
features = ['Mean_Temperature', 'Mean_light', 'Mean_sound', 'Mean_CO2']
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for i, feature in enumerate(features):
    row = i // 2
    col = i % 2
    axs[row, col].scatter(df_cleaned[feature], df_cleaned['Room_Occupancy_Count'])
    axs[row, col].set_title(f'Room Occupancy vs {feature}')
    axs[row, col].set_xlabel(feature)
    axs[row, col].set_ylabel('Room Occupancy Count')

plt.tight_layout()
plt.show()


######################################################

#Sensor readings (mean) vs Room Occupancy Count (bar plot)
# List of mean columns
mean_columns = ['Mean_Temperature', 'Mean_light', 'Mean_sound', 'Mean_CO2', 'Mean_CO2_Slope', 'Mean_PIR']

fig, axs = plt.subplots(2, 3, figsize=(20, 10))

for i, column in enumerate(mean_columns):
    if column in df_grouped.columns:
        row = i // 3
        col = i % 3
        df_grouped.plot(kind='bar', x='Room_Occupancy_Count', y=column, ax=axs[row, col])
        axs[row, col].set_title('Comparison of ' + column + ' vs Room Occupancy Count')
        axs[row, col].set_ylabel(column)
        axs[row, col].set_xlabel('Room Occupancy Count')
        axs[row, col].set_xticklabels(axs[row, col].get_xticklabels(), rotation=0)  # to keep the x-axis labels vertical

plt.tight_layout()
plt.show()


######################################################
#Correlation matrix of sensor readings
correlation_matrix = df_cleaned[['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light', 'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound']].corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix of Sensor Readings')
plt.show()

######################################################


# # Mean Sensor Readings vs Room Occupancy Count over time
# Create a figure
fig = go.Figure()

# Add traces for mean sensor readings
for column in ['Mean_Temperature', 'Mean_light', 'Mean_sound', 'Mean_CO2', 'Mean_CO2_Slope', 'Mean_PIR']:
    fig.add_trace(go.Scatter(x=df_cleaned.index, y=df_cleaned[column], name=column))

# Add trace for room occupancy count
fig.add_trace(go.Scatter(x=df_cleaned.index, y=df_cleaned['Room_Occupancy_Count'], name='Room_Occupancy_Count', yaxis='y2'))

# Update layout to include 2 y-axes and adjust legend position
fig.update_layout(
    yaxis=dict(title='Mean Sensor Readings'),
    yaxis2=dict(title='Room Occupancy Count', overlaying='y', side='right'),
    title='Mean Sensor Readings vs Room Occupancy Count',
    hovermode="x unified",
    legend=dict(x=8, y=7)  
)

fig.show()