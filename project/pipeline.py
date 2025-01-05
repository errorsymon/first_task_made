import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os
import sqlite3

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
    
    # Save feature importances
    output_dir = os.getcwd()
    usa_features_file = os.path.join(output_dir, 'USA_Important_Features.csv')
    brazil_features_file = os.path.join(output_dir, 'Brazil_Important_Features.csv')
    usa_importances.to_csv(usa_features_file, index=False)
    brazil_importances.to_csv(brazil_features_file, index=False)
    
    # Extract top 5 indicators for both USA and Brazil
    usa_top_indicators = extract_top_indicators(usa_importances, top_n=5)
    brazil_top_indicators = extract_top_indicators(brazil_importances, top_n=5)
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

if __name__ == "__main__":
    main()
