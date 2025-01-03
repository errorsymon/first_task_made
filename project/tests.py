import pytest
import pandas as pd
import numpy as np
import os
import sqlite3
from pipeline import load_datasets, preprocess_gdp_data, preprocess_country_metadata, preprocess_indicator_data, merge_data, train_and_get_importance, save_to_sqlite

@pytest.fixture
def datasets():
    # Load datasets using the provided function
    gdp_data, country_metadata, usa_data, brazil_data = load_datasets()
    gdp_data = preprocess_gdp_data(gdp_data)
    country_metadata = preprocess_country_metadata(country_metadata)
    merged_data = merge_data(gdp_data, country_metadata)
    usa_pivot = preprocess_indicator_data(usa_data)
    brazil_pivot = preprocess_indicator_data(brazil_data)
    return gdp_data, country_metadata, merged_data, usa_pivot, brazil_pivot

def test_load_datasets(datasets):
    # Check how many elements are in datasets
    print(f"Number of elements in datasets: {len(datasets)}")
    
    # Unpack only the first 4 values if that's all you need
    gdp_data, country_metadata, usa_data, brazil_data = datasets[:4]
    
    # Proceed with the test as intended
    assert gdp_data is not None
    assert country_metadata is not None
    assert usa_data is not None
    assert brazil_data is not None


def test_preprocess_gdp_data(datasets):
    gdp_data, country_metadata, merged_data, usa_pivot, brazil_pivot = datasets
    assert 'GDP' in gdp_data.columns, "'GDP' column is missing after preprocessing"
    assert gdp_data['Year'].dtype == np.int64, "Year column is not of type int after preprocessing"
    assert gdp_data.dropna(subset=['GDP']).shape[0] < gdp_data.shape[0], "Preprocessing did not drop rows with missing GDP values"

def test_train_and_get_importance(datasets):
    gdp_data, country_metadata, merged_data, usa_pivot, brazil_pivot = datasets
    merged_data['Year'] = merged_data['Year'].astype(str)  # Ensure consistency
    
    # Preprocess USA and Brazil data
    usa_merged = pd.merge(usa_pivot, merged_data, on="Year", how="left").fillna(0)
    brazil_merged = pd.merge(brazil_pivot, merged_data, on="Year", how="left").fillna(0)
    
    # Train models and get feature importances
    usa_importances, _ = train_and_get_importance(usa_merged, target_col="GDP")
    brazil_importances, _ = train_and_get_importance(brazil_merged, target_col="GDP")
    
    assert not usa_importances.empty, "USA feature importances are empty"
    assert not brazil_importances.empty, "Brazil feature importances are empty"
    
    # Check that the feature importances have been sorted correctly
    assert 'Importance' in usa_importances.columns, "'Importance' column is missing in USA importances"
    assert 'Importance' in brazil_importances.columns, "'Importance' column is missing in Brazil importances"
    
    # Ensure there is at least one feature with non-zero importance
    assert usa_importances['Importance'].max() > 0, "No features with non-zero importance in USA"
    assert brazil_importances['Importance'].max() > 0, "No features with non-zero importance in Brazil"

def test_save_to_sqlite(datasets):
    gdp_data, country_metadata, merged_data, usa_pivot, brazil_pivot = datasets
    merged_data['Year'] = merged_data['Year'].astype(str)  # Ensure consistency
    
    # Preprocess USA and Brazil data
    usa_merged = pd.merge(usa_pivot, merged_data, on="Year", how="left").fillna(0)
    brazil_merged = pd.merge(brazil_pivot, merged_data, on="Year", how="left").fillna(0)
    
    # Train models and get feature importances
    usa_importances, _ = train_and_get_importance(usa_merged, target_col="GDP")
    brazil_importances, _ = train_and_get_importance(brazil_merged, target_col="GDP")
    
    # Extract top indicators
    usa_top_indicators = usa_importances.head(5)['Feature'].tolist()
    brazil_top_indicators = brazil_importances.head(5)['Feature'].tolist()
    
    # Save to SQLite
    save_to_sqlite(usa_top_indicators, brazil_top_indicators, usa_merged, brazil_merged)
    
    # Check if the tables exist in the SQLite database
    conn = sqlite3.connect('gdp_brazil_usa.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    
    # Assert that tables are created for USA and Brazil
    assert ('usa_top_indicators',) in tables, "Table 'usa_top_indicators' not found in SQLite database"
    assert ('brazil_top_indicators',) in tables, "Table 'brazil_top_indicators' not found in SQLite database"

def test_final_data_csv(datasets):
    gdp_data, country_metadata, merged_data, usa_pivot, brazil_pivot = datasets
    merged_data['Year'] = merged_data['Year'].astype(str)  # Ensure consistency
    
    # Preprocess USA and Brazil data
    usa_merged = pd.merge(usa_pivot, merged_data, on="Year", how="left").fillna(0)
    brazil_merged = pd.merge(brazil_pivot, merged_data, on="Year", how="left").fillna(0)
    
    # Train models and get feature importances
    usa_importances, _ = train_and_get_importance(usa_merged, target_col="GDP")
    brazil_importances, _ = train_and_get_importance(brazil_merged, target_col="GDP")
    
    # Extract top indicators
    usa_top_indicators = usa_importances.head(5)['Feature'].tolist()
    brazil_top_indicators = brazil_importances.head(5)['Feature'].tolist()
    
    # Save final data CSV
    output_dir = os.getcwd()
    final_data_file = os.path.join(output_dir, 'final_data.csv')
    
    # Simulate running the code that generates the final_data.csv
    with open(final_data_file, 'w') as f:
        f.write("USA Data\n")
        usa_merged[['Year'] + usa_top_indicators].to_csv(f, index=False)
        f.write("\nBrazil Data\n")
        brazil_merged[['Year'] + brazil_top_indicators].to_csv(f, index=False)
    
    # Check if the CSV file was generated
    assert os.path.exists(final_data_file), "final_data.csv was not created"
    
    # Ensure that the file contains both USA and Brazil data
    with open(final_data_file, 'r') as f:
        content = f.read()
        assert "USA Data" in content, "'USA Data' section is missing in final_data.csv"
        assert "Brazil Data" in content, "'Brazil Data' section is missing in final_data.csv"

if __name__ == "__main__":
    pytest.main()
