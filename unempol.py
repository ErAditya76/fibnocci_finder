import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample data since we can't download directly from Kaggle
num_days = 332
data = pd.DataFrame({
    'date': pd.date_range(start='2020-01-01', periods=num_days, freq='D'),
    'region': ['North', 'South', 'East', 'West'] * (num_days // 4) + ['North'] * (num_days % 4),
    'area_type': ['Urban', 'Rural'] * (num_days // 2) + ['Urban'] * (num_days % 2),
    'estimated_unemployment_rate': [7.5 + i/100 for i in range(num_days)]
})

# Data Cleaning
# Rename columns for easier access
data.columns = [col.strip().replace(' ', '_').lower() for col in data.columns]

# Fill missing values if any
data.fillna(method='ffill', inplace=True)

# Exploratory Data Analysis
plt.figure(figsize=(12, 6))
sns.lineplot(
    x='date', 
    y='estimated_unemployment_rate', 
    data=data, 
    label='Unemployment Rate'
)
plt.xticks(rotation=45)
plt.title('Trends in Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Estimated Unemployment Rate')
plt.legend()
plt.tight_layout()
plt.show()

# Regional Analysis
plt.figure(figsize=(14, 7))
sns.boxplot(
    x='region', 
    y='estimated_unemployment_rate', 
    data=data
)
plt.title('Unemployment Rate Across Regions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Heatmap of unemployment rates by region and area type
pivot_data = data.pivot_table(
    index='region', 
    columns='area_type', 
    values='estimated_unemployment_rate', 
    aggfunc='mean'
)
plt.figure(figsize=(12, 8))
sns.heatmap(
    pivot_data, 
    annot=True, 
    cmap='YlGnBu', 
    fmt='.2f'
)
plt.title('Average Unemployment Rates by Region and Area Type')
plt.tight_layout()
plt.show()
