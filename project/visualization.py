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

def find_top_gdp_change_year(data, country):
    country_data = data[data['Country Name'] == country].sort_values(by='Year')
    country_data['GDP_Change'] = country_data['GDP'].diff()
    country_data['GDP_Change_Percent'] = (country_data['GDP_Change'] / country_data['GDP'].shift(1)) * 100
    max_change_year = country_data.loc[country_data['GDP_Change'].idxmax(), 'Year']
    max_change = country_data['GDP_Change'].max()
    max_change_percent = country_data['GDP_Change_Percent'].max()
    return max_change_year, max_change, max_change_percent

def load_importance_files():
    # Load the feature importance CSV files for USA and Brazil
    usa_importance_file = 'USA_Important_Features.csv'
    brazil_importance_file = 'Brazil_Important_Features.csv'
    
    usa_importances = pd.read_csv(usa_importance_file)
    brazil_importances = pd.read_csv(brazil_importance_file)
    
    return usa_importances, brazil_importances

def extract_top_indicators(importance_df, top_n=5):
    # Extract the top `n` features (indicators) based on importance value
    top_indicators = importance_df.head(top_n)['Feature'].tolist()
    return top_indicators

def save_to_sqlite(usa_top_indicators, brazil_top_indicators, merged_usa_data, merged_brazil_data):
    # Connect to SQLite database
    conn = sqlite3.connect('gdp_brazil_usa.db')

    # Create a table for USA top features
    usa_top_indicators_data = merged_usa_data[['Year'] + usa_top_indicators]
    usa_top_indicators_data.to_sql('usa_top_indicators', conn, if_exists='replace', index=False)

    # Create a table for Brazil top features
    brazil_top_indicators_data = merged_brazil_data[['Year'] + brazil_top_indicators]
    brazil_top_indicators_data.to_sql('brazil_top_indicators', conn, if_exists='replace', index=False)

    # Commit changes and close the connection
    conn.commit()
    conn.close()
    print("Top indicators for USA and Brazil have been saved to SQLite.")

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

def plot_feature_importance(importance_df, country_name, top_n=101):
    # Plotting top `n` features that contribute to GDP
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=top_features, hue='Feature', palette='viridis', legend=False)
    plt.title(f'Top {top_n} Factors Contributing to GDP for {country_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

def plot_gdp_growth_comparison(usa_data, brazil_data):
    plt.figure(figsize=(10, 6))
    plt.plot(usa_data['Year'], usa_data['GDP'], label='USA GDP', color='black', marker='o')
    plt.plot(brazil_data['Year'], brazil_data['GDP'], label='Brazil GDP', color='green', marker='o')
    plt.title('GDP Growth Comparison Between USA and Brazil')
    plt.xlabel('Year')
    plt.ylabel('GDP in USD')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(usa_data, brazil_data, usa_top_indicators, brazil_top_indicators):
    # Merge the top indicator columns for USA and Brazil
    combined_data = pd.merge(usa_data[['Year'] + usa_top_indicators], brazil_data[['Year'] + brazil_top_indicators], on='Year', suffixes=('_USA', '_Brazil'))
    
    # Check if there are any numeric columns left after dropping 'Year'
    numeric_data = combined_data.drop(columns=['Year']).select_dtypes(include=[np.number])
    
    # If there are no numeric columns, print an error and return
    if numeric_data.empty:
        print("Error: No numeric columns available for correlation.")
        return

    # Calculate the correlation matrix
    correlation_matrix = numeric_data.corr()
    
    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Top Indicators for USA and Brazil')
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

    # Plot GDP Growth
    plt.subplot(1, 2, 1)
    plt.plot(usa_data['Year'], usa_data['GDP'], label='USA GDP', color='blue', marker='o')
    plt.plot(brazil_data['Year'], brazil_data['GDP'], label='Brazil GDP', color='green', marker='o')
    plt.title('GDP Growth Comparison')
    plt.xlabel('Year')
    plt.ylabel('GDP in USD')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()

    # Plot GDP Change Percentage
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


def plot_correlation_matrix_for_common_indicators(usa_data, brazil_data, usa_top_indicators, brazil_top_indicators):
    # Identify the common top indicators between USA and Brazil
    common_indicators = list(set(usa_top_indicators).intersection(brazil_top_indicators))
    
    if not common_indicators:
        print("No common top indicators between USA and Brazil for correlation.")
        return
    
    # Merge the top indicator columns for USA and Brazil based on common indicators
    combined_data = pd.merge(usa_data[['Year'] + common_indicators], brazil_data[['Year'] + common_indicators], on='Year', suffixes=('_USA', '_Brazil'))
    
    # Check if there are any numeric columns left after dropping 'Year'
    numeric_data = combined_data.drop(columns=['Year']).select_dtypes(include=[np.number])
    
    # If there are no numeric columns, print an error and return
    if numeric_data.empty:
        print("Error: No numeric columns available for correlation.")
        return

    # Calculate the correlation matrix
    correlation_matrix = numeric_data.corr()
    
    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Common Top Indicators for USA and Brazil')
    plt.tight_layout()
    plt.show()

def main():
    gdp_data, country_metadata, usa_data, brazil_data = load_datasets()
    gdp_data = preprocess_gdp_data(gdp_data)
    country_metadata = preprocess_country_metadata(country_metadata)
    merged_data = merge_data(gdp_data, country_metadata)
    merged_data['Year'] = merged_data['Year'].astype(str)  # Ensure consistency
    
    # Preprocess USA and Brazil data
    usa_pivot = preprocess_indicator_data(usa_data)
    brazil_pivot = preprocess_indicator_data(brazil_data)
    usa_pivot['Year'] = usa_pivot['Year'].astype(str)
    brazil_pivot['Year'] = brazil_pivot['Year'].astype(str)
    
    # Merge datasets with USA and Brazil
    usa_merged = pd.merge(usa_pivot, merged_data, on="Year", how="left").fillna(0)
    brazil_merged = pd.merge(brazil_pivot, merged_data, on="Year", how="left").fillna(0)
    
    # Train models and get feature importances
    usa_importances, usa_model = train_and_get_importance(usa_merged, target_col="GDP")
    brazil_importances, brazil_model = train_and_get_importance(brazil_merged, target_col="GDP")
    
    # Visualize GDP changes and top factors
    plot_combined_gdp_changes(usa_merged, brazil_merged)
    
    plot_feature_importance(usa_importances, 'USA', top_n=10)
    plot_feature_importance(brazil_importances, 'Brazil', top_n=10)
    
    # Save feature importances
    output_dir = os.getcwd()
    usa_features_file = os.path.join(output_dir, 'USA_Important_Features.csv')
    brazil_features_file = os.path.join(output_dir, 'Brazil_Important_Features.csv')
    usa_importances.to_csv(usa_features_file, index=False)
    brazil_importances.to_csv(brazil_features_file, index=False)
    
    # Extract top 10 indicators for both USA and Brazil
    usa_top_indicators = extract_top_indicators(usa_importances, top_n=10)
    brazil_top_indicators = extract_top_indicators(brazil_importances, top_n=10)
    print(f"Top 5 indicators for USA: {usa_top_indicators}")
    print(f"Top 5 indicators for Brazil: {brazil_top_indicators}")
    
    # Save the top indicators to SQLite database
    save_to_sqlite(usa_top_indicators, brazil_top_indicators, usa_merged, brazil_merged)

    # Find the top GDP change year for USA and Brazil
    usa_top_year, usa_max_change, usa_max_change_percent = find_top_gdp_change_year(gdp_data, 'United States')
    brazil_top_year, brazil_max_change, brazil_max_change_percent = find_top_gdp_change_year(gdp_data, 'Brazil')
    print(f"USA: Year with highest GDP change: {usa_top_year} (Change: {usa_max_change}, Percent Change: {usa_max_change_percent:.2f}%)")
    print(f"Brazil: Year with highest GDP change: {brazil_top_year} (Change: {brazil_max_change}, Percent Change: {brazil_max_change_percent:.2f}%)")
    
    # Create and save data tables for USA and Brazil
    usa_top_indicators_data = usa_merged[['Year'] + usa_top_indicators].assign(Country='USA')
    brazil_top_indicators_data = brazil_merged[['Year'] + brazil_top_indicators].assign(Country='Brazil')

    usa_top_indicators_data = usa_top_indicators_data.drop_duplicates()  # Remove duplicate rows
    brazil_top_indicators_data = brazil_top_indicators_data.drop_duplicates()  # Remove duplicate rows

    # Combine and save final_data.csv in two sections
    with open('final_data.csv', 'w') as f:
        f.write("USA Data\n")
        usa_top_indicators_data.to_csv(f, index=False)
        f.write("\nBrazil Data\n")
        brazil_top_indicators_data.to_csv(f, index=False)
        f.write(f"\nUSA: Year with highest GDP change: {usa_top_year} (Change: {usa_max_change})\n")
        f.write(f"Brazil: Year with highest GDP change: {brazil_top_year} (Change: {brazil_max_change})\n")
    
    print("Final data has been saved to final_data.csv in separate tables for USA and Brazil.")
    
    # Plot correlation matrix of top factors for USA and Brazil
    plot_correlation_matrix_for_common_indicators(usa_merged, brazil_merged, usa_top_indicators, brazil_top_indicators)

if __name__ == "__main__":
    main()
