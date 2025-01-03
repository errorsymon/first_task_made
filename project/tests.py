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

def load_importance_files():
    usa_importance_file = 'USA_Important_Features.csv'
    brazil_importance_file = 'Brazil_Important_Features.csv'

    usa_importances = pd.read_csv(usa_importance_file)
    brazil_importances = pd.read_csv(brazil_importance_file)

    return usa_importances, brazil_importances

def extract_top_indicators(importance_df, top_n=5):
    top_indicators = importance_df.head(top_n)['Feature'].tolist()
    return top_indicators

def save_to_sqlite(usa_top_indicators, brazil_top_indicators, merged_usa_data, merged_brazil_data):
    conn = sqlite3.connect('gdp_brazil_usa.db')

    usa_top_indicators_data = merged_usa_data[['Year'] + usa_top_indicators]
    usa_top_indicators_data.to_sql('usa_top_indicators', conn, if_exists='replace', index=False)

    brazil_top_indicators_data = merged_brazil_data[['Year'] + brazil_top_indicators]
    brazil_top_indicators_data.to_sql('brazil_top_indicators', conn, if_exists='replace', index=False)

    conn.commit()
    conn.close()
    print("Top indicators for USA and Brazil have been saved to SQLite.")

def main():
    gdp_data, country_metadata, usa_data, brazil_data = load_datasets()
    gdp_data = preprocess_gdp_data(gdp_data)
    country_metadata = preprocess_country_metadata(country_metadata)
    merged_data = merge_data(gdp_data, country_metadata)
    merged_data['Year'] = merged_data['Year'].astype(str)

    usa_pivot = preprocess_indicator_data(usa_data)
    brazil_pivot = preprocess_indicator_data(brazil_data)
    usa_pivot['Year'] = usa_pivot['Year'].astype(str)
    brazil_pivot['Year'] = brazil_pivot['Year'].astype(str)

    usa_merged = pd.merge(usa_pivot, merged_data, on="Year", how="left").fillna(0)
    brazil_merged = pd.merge(brazil_pivot, merged_data, on="Year", how="left").fillna(0)

    usa_importances, usa_model = train_and_get_importance(usa_merged, target_col="GDP")
    brazil_importances, brazil_model = train_and_get_importance(brazil_merged, target_col="GDP")

    output_dir = os.getcwd()
    usa_features_file = os.path.join(output_dir, 'USA_Important_Features.csv')
    brazil_features_file = os.path.join(output_dir, 'Brazil_Important_Features.csv')
    usa_importances.to_csv(usa_features_file, index=False)
    brazil_importances.to_csv(brazil_features_file, index=False)

    usa_top_indicators = extract_top_indicators(usa_importances, top_n=5)
    brazil_top_indicators = extract_top_indicators(brazil_importances, top_n=5)
    print(f"Top 5 indicators for USA: {usa_top_indicators}")
    print(f"Top 5 indicators for Brazil: {brazil_top_indicators}")

    save_to_sqlite(usa_top_indicators, brazil_top_indicators, usa_merged, brazil_merged)

    usa_top_indicators_data = usa_merged[['Year'] + usa_top_indicators].assign(Country='USA')
    brazil_top_indicators_data = brazil_merged[['Year'] + brazil_top_indicators].assign(Country='Brazil')

    usa_top_indicators_data = usa_top_indicators_data.drop_duplicates()
    brazil_top_indicators_data = brazil_top_indicators_data.drop_duplicates()

    with open('final_data.csv', 'w') as f:
        f.write("USA Data\n")
        usa_top_indicators_data.to_csv(f, index=False)
        f.write("\nBrazil Data\n")
        brazil_top_indicators_data.to_csv(f, index=False)

    print("Final data has been saved to final_data.csv in separate tables for USA and Brazil.")

if __name__ == "__main__":
    main()
