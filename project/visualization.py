import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

def load_datasets():
    gdp_data_url = "https://raw.githubusercontent.com/errorsymon/Data/d710147cfb374060422bd86a1889d33e54fa3f2b/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_9865.csv"
    country_metadata_url = "https://raw.githubusercontent.com/errorsymon/Data/d710147cfb374060422bd86a1889d33e54fa3f2b/Metadata_Country_API_NY.GDP.MKTP.CD_DS2_en_csv_v2_9865.csv"
    usa_data_url = "https://raw.githubusercontent.com/errorsymon/Data/refs/heads/main/API_USA_DS2_en_csv_v2_3173.csv"
    brazil_data_url = "https://raw.githubusercontent.com/errorsymon/Data/refs/heads/main/brazil.csv"
    
    gdp_data = pd.read_csv(gdp_data_url, skiprows=4)
    country_metadata = pd.read_csv(country_metadata_url)
    usa_data = pd.read_csv(usa_data_url)
    brazil_data = pd.read_csv(brazil_data_url)
    return gdp_data, country_metadata, usa_data, brazil_data

def preprocess_gdp_data(data):
    years = [col for col in data.columns if col.isdigit()]
    relevant_columns = ['Country Name', 'Country Code'] + years
    data = data[relevant_columns].melt(id_vars=['Country Name', 'Country Code'], var_name='Year', value_name='GDP')
    data['Year'] = data['Year'].astype(int)
    data.dropna(subset=['GDP'], inplace=True)
    return data

def preprocess_country_metadata(data):
    relevant_columns = ['Country Code', 'Region', 'IncomeGroup']
    return data[relevant_columns]

def preprocess_indicator_data(data):
    data = data.iloc[1:].reset_index(drop=True)
    return data.melt(id_vars=["Indicator Code"], var_name="Year", value_name="Value").pivot(index="Year", columns="Indicator Code", values="Value").reset_index()

def merge_data(gdp_data, country_metadata):
    return pd.merge(gdp_data, country_metadata, on='Country Code', how='left')

def train_and_get_importance(data, target_col):
    data = data.fillna(0)
    y = data[target_col]
    X = data.drop(columns=[target_col, "Country Name", "Country Code", "Region", "IncomeGroup"], errors="ignore")
    label_encoder = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = label_encoder.fit_transform(X[col].astype(str))
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
    return importance, model

def encode_data(usa_data, brazil_data, usa_top_indicators, brazil_top_indicators):
    # Create an instance of LabelEncoder
    label_encoder = LabelEncoder()

    # Apply LabelEncoder for categorical columns in USA data
    for col in usa_top_indicators:
        if usa_data[col].dtype == 'object':  # Check if the column is categorical
            usa_data[col] = label_encoder.fit_transform(usa_data[col].astype(str))

    # Apply LabelEncoder for categorical columns in Brazil data
    for col in brazil_top_indicators:
        if brazil_data[col].dtype == 'object':  # Check if the column is categorical
            brazil_data[col] = label_encoder.fit_transform(brazil_data[col].astype(str))

    return usa_data, brazil_data

def calculate_and_plot_correlation(usa_data, brazil_data, usa_top_indicators, brazil_top_indicators):
    # Encode the data to ensure all columns are numeric
    usa_data_encoded, brazil_data_encoded = encode_data(usa_data, brazil_data, usa_top_indicators, brazil_top_indicators)

    # Merge the USA and Brazil data based on 'Year'
    combined_data = pd.merge(usa_data_encoded[['Year'] + usa_top_indicators], brazil_data_encoded[['Year'] + brazil_top_indicators], 
                             on='Year', suffixes=('_USA', '_Brazil'))
    
    # Calculate the correlation matrix
    correlation_matrix = combined_data.drop(columns=['Year']).corr()

    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Top Indicators for USA and Brazil')
    plt.tight_layout()
    plt.show()

def plot_gdp_changes(gdp_data, country_name, title):
    country_data = gdp_data[gdp_data['Country Name'] == country_name]
    country_data = country_data.sort_values('Year')
    
    # Calculate GDP change and percentage change
    country_data['GDP_Change'] = country_data['GDP'].diff()
    country_data['GDP_Change_Percent'] = (country_data['GDP_Change'] / country_data['GDP'].shift(1)) * 100
    
    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(country_data['Year'], country_data['GDP'], label=f'{country_name} GDP', color='blue', marker='o')
    plt.title(f'{title} GDP Over the Years')
    plt.xlabel('Year')
    plt.ylabel('GDP in USD')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot GDP Change as a percentage
    plt.figure(figsize=(10, 6))
    plt.plot(country_data['Year'], country_data['GDP_Change_Percent'], label=f'{country_name} GDP % Change', color='red', marker='o')
    plt.title(f'{title} GDP Percentage Change Over the Years')
    plt.xlabel('Year')
    plt.ylabel('GDP Percentage Change (%)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_feature_importance(importance_df, country_name, top_n=10):
    # Plotting top `n` features that contribute to GDP
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=top_features, hue='Feature', palette='viridis', legend=False)
    plt.title(f'Top {top_n} Factors Contributing to GDP for {country_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

def plot_combined_gdp_changes(usa_data, brazil_data):
    # Calculate GDP Change and GDP Change Percent for USA
    usa_data['GDP_Change'] = usa_data['GDP'].diff()
    usa_data['GDP_Change_Percent'] = (usa_data['GDP_Change'] / usa_data['GDP'].shift(1)) * 100

    # Calculate GDP Change and GDP Change Percent for Brazil
    brazil_data['GDP_Change'] = brazil_data['GDP'].diff()
    brazil_data['GDP_Change_Percent'] = (brazil_data['GDP_Change'] / brazil_data['GDP'].shift(1)) * 100

    # Create the combined plot
    plt.figure(figsize=(12, 6))

    
    plt.subplot(1, 2, 1)
    plt.plot(usa_data['Year'], usa_data['GDP'], label='USA GDP', color='blue', marker='o')
    plt.plot(brazil_data['Year'], brazil_data['GDP'], label='Brazil GDP', color='green', marker='o')
    plt.title('GDP Growth Comparison')
    plt.xlabel('Year')
    plt.ylabel('GDP in USD')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()

   
    plt.subplot(1, 2, 2)
    plt.plot(usa_data['Year'], usa_data['GDP_Change_Percent'], label='USA GDP % Change', color='red', marker='o')
    plt.plot(brazil_data['Year'], brazil_data['GDP_Change_Percent'], label='Brazil GDP % Change', color='orange', marker='o')
    plt.title('GDP Percentage Change Comparison')
    plt.xlabel('Year')
    plt.ylabel('GDP Percentage Change (%)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    gdp_data, country_metadata, usa_data, brazil_data = load_datasets()
    gdp_data = preprocess_gdp_data(gdp_data)
    country_metadata = preprocess_country_metadata(country_metadata)
    merged_data = merge_data(gdp_data, country_metadata)
    merged_data['Year'] = merged_data['Year'].astype(str)  # Ensure consistency
    
  
    usa_pivot = preprocess_indicator_data(usa_data)
    brazil_pivot = preprocess_indicator_data(brazil_data)
    usa_pivot['Year'] = usa_pivot['Year'].astype(str)
    brazil_pivot['Year'] = brazil_pivot['Year'].astype(str)
    
    
    usa_merged = pd.merge(usa_pivot, merged_data, on="Year", how="left").fillna(0)
    brazil_merged = pd.merge(brazil_pivot, merged_data, on="Year", how="left").fillna(0)
    
    # Train models and get the important factors 
    usa_importances, usa_model = train_and_get_importance(usa_merged, target_col="GDP")
    brazil_importances, brazil_model = train_and_get_importance(brazil_merged, target_col="GDP")

    # Show the top important data for usa and Brazil 
    plot_feature_importance(usa_importances, 'USA')
    plot_feature_importance(brazil_importances, 'Brazil')

    # Plot Gdp changes for both countries 
    plot_gdp_changes(gdp_data, 'United States', 'USA')
    plot_gdp_changes(gdp_data, 'Brazil', 'Brazil')

    # Plot the merged factors
    plot_combined_gdp_changes(usa_merged, brazil_merged)
    
    # Extract the top factors
    usa_top_indicators = usa_importances['Feature'].head(10).tolist()
    brazil_top_indicators = brazil_importances['Feature'].head(10).tolist()

    # Computing the correlation matrix for both countries
    calculate_and_plot_correlation(usa_merged, brazil_merged, usa_top_indicators, brazil_top_indicators)

if __name__ == "__main__":
    main()
