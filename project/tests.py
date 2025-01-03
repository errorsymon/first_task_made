import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os
import sqlite3
import tempfile

# File paths for temporary storage in CI
output_dir = tempfile.gettempdir()

def load_datasets():
    try:
        gdp_data_url = "https://raw.githubusercontent.com/errorsymon/Data/d710147cfb374060422bd86a1889d33e54fa3f2b/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_9865.csv"
        country_metadata_url = "https://raw.githubusercontent.com/errorsymon/Data/d710147cfb374060422bd86a1889d33e54fa3f2b/Metadata_Country_API_NY.GDP.MKTP.CD_DS2_en_csv_v2_9865.csv"
        usa_data_url = "https://raw.githubusercontent.com/errorsymon/Data/refs/heads/main/API_USA_DS2_en_csv_v2_3173.csv"
        brazil_data_url = "https://raw.githubusercontent.com/errorsymon/Data/refs/heads/main/brazil.csv"

        gdp_data = pd.read_csv(gdp_data_url, skiprows=4)
        country_metadata = pd.read_csv(country_metadata_url)
        usa_data = pd.read_csv(usa_data_url)
        brazil_data = pd.read_csv(brazil_data_url)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        raise
    return gdp_data, country_metadata, usa_data, brazil_data

def preprocess_gdp_data(data):
    years = [col for col in data.columns if col.isdigit()]
    relevant_columns = ['Country Name', 'Country Code'] + years
    data = data[relevant_columns].melt(id_vars=['Country Name', 'Country Code'], var_name='Year', value_name='GDP')
    data['Year'] = data['Year'].astype(int)
    data.dropna(subset=['GDP'], inplace=True)
    return data

def main():
    print("Loading datasets...")
    gdp_data, country_metadata, usa_data, brazil_data = load_datasets()

    print("Preprocessing GDP data...")
    gdp_data = preprocess_gdp_data(gdp_data)
    print(f"GDP data processed: {gdp_data.shape}")

    # Example for testing purposes
    print("Saving final data to temp directory...")
    gdp_data.to_csv(os.path.join(output_dir, "processed_gdp_data.csv"), index=False)
    print(f"Processed data saved to {output_dir}")

if __name__ == "__main__":
    main()
