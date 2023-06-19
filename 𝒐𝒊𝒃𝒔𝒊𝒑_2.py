import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('unemployment_data.csv')

# Filter the data for the desired columns
filtered_data = data[['Region', 'Date', 'Estimated Unemployment Rate (%)']]

# Convert the 'Date' column to datetime format
filtered_data['Date'] = pd.to_datetime(filtered_data['Date'], dayfirst=True)

# Group the data by 'Region' and calculate the average unemployment rate for each region
average_unemployment = filtered_data.groupby('Region')['Estimated Unemployment Rate (%)'].mean()

# Plotting the average unemployment rate for each region
plt.figure(figsize=(10, 6))
sns.barplot(x=average_unemployment.index, y=average_unemployment.values)
plt.title('Average Unemployment Rate by Region')
plt.xlabel('Region')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.show()

# Plotting a line plot of unemployment rate over time for a specific region
region = 'Andhra Pradesh'
region_data = filtered_data[filtered_data['Region'] == region]
plt.figure(figsize=(10, 6))
sns.lineplot(x=region_data['Date'], y=region_data['Estimated Unemployment Rate (%)'])
plt.title(f'Unemployment Rate Trend for {region}')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.show()

# Plotting a histogram of the unemployment rate distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Estimated Unemployment Rate (%)'], bins=10, kde=True)
plt.title('Unemployment Rate Distribution')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Count')
plt.show()

# Creating a box plot to visualize the distribution of unemployment rates across regions
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['Region'], y=data['Estimated Unemployment Rate (%)'])
plt.title('Unemployment Rate Distribution by Region')
plt.xlabel('Region')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.show()
