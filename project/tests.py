import pytest
import pandas as pd
import numpy as np
from pipeline import load_datasets, preprocess_gdp_data, preprocess_country_metadata, preprocess_indicator_data, merge_data, train_and_get_importance, extract_top_indicators, save_to_sqlite
from sklearn.ensemble import RandomForestRegressor

# Test for loading datasets
def test_load_datasets():
    gdp_data, country_metadata, usa_data, brazil_data = load_datasets()
    assert not gdp_data.empty, "GDP data is empty"
    assert not country_metadata.empty, "Country metadata is empty"
    assert not usa_data.empty, "USA data is empty"
    assert not brazil_data.empty, "Brazil data is empty"

# Test for GDP data preprocessing
def test_preprocess_gdp_data():
    gdp_data, _, _, _ = load_datasets()
    processed_gdp = preprocess_gdp_data(gdp_data)
    assert "GDP" in processed_gdp.columns, "GDP column missing in processed data"
    
    # Check if any NaN values exist in critical columns
    assert processed_gdp['GDP'].isna().sum() == 0, "NaN values found in GDP column"
    assert processed_gdp['Year'].isna().sum() == 0, "NaN values found in Year column"
    assert processed_gdp['Country Name'].isna().sum() == 0, "NaN values found in Country Name column"
    assert processed_gdp['Country Code'].isna().sum() == 0, "NaN values found in Country Code column"

# Test for country metadata preprocessing
def test_preprocess_country_metadata():
    _, country_metadata, _, _ = load_datasets()
    processed_metadata = preprocess_country_metadata(country_metadata)
    assert "Country Code" in processed_metadata.columns, "Country Code missing in processed metadata"
    assert "Region" in processed_metadata.columns, "Region column missing in processed metadata"
    assert "IncomeGroup" in processed_metadata.columns, "IncomeGroup column missing in processed metadata"

# Test for indicator data preprocessing (USA and Brazil)
def test_preprocess_indicator_data():
    _, _, usa_data, brazil_data = load_datasets()
    usa_pivot = preprocess_indicator_data(usa_data)
    brazil_pivot = preprocess_indicator_data(brazil_data)
    
    assert "Year" in usa_pivot.columns, "Year column missing in USA indicator data"
    assert "Year" in brazil_pivot.columns, "Year column missing in Brazil indicator data"
    
    # Check if the melted data is correctly pivoted
    assert usa_pivot.shape[0] > 0, "USA pivoted data is empty"
    assert brazil_pivot.shape[0] > 0, "Brazil pivoted data is empty"

# Test for merging GDP data with country metadata
def test_merge_data():
    gdp_data, country_metadata, _, _ = load_datasets()
    processed_gdp = preprocess_gdp_data(gdp_data)
    processed_metadata = preprocess_country_metadata(country_metadata)
    merged_data = merge_data(processed_gdp, processed_metadata)
    
    assert "Country Name" in merged_data.columns, "Country Name missing in merged data"
    assert "GDP" in merged_data.columns, "GDP column missing in merged data"
    
    # Check for missing values in critical columns
    assert merged_data[['Country Name', 'GDP']].isna().sum().sum() == 0, "NaN values found in critical merged data columns"
    
    # Allow NaN in 'Region' and 'IncomeGroup' columns but check for missing values in other columns
    assert merged_data.drop(columns=["Region", "IncomeGroup"]).isna().sum().sum() == 0, "NaN values found in merged data excluding 'Region' and 'IncomeGroup'"

# Test for RandomForest model and feature importance extraction
def test_train_and_get_importance():
    gdp_data, country_metadata, usa_data, brazil_data = load_datasets()
    gdp_data = preprocess_gdp_data(gdp_data)
    country_metadata = preprocess_country_metadata(country_metadata)
    merged_data = merge_data(gdp_data, country_metadata)
    
    # Preprocess USA and Brazil data
    usa_pivot = preprocess_indicator_data(usa_data)
    brazil_pivot = preprocess_indicator_data(brazil_data)
    
    # Ensure 'Year' is of type int64 in all dataframes before merging
    usa_pivot['Year'] = pd.to_numeric(usa_pivot['Year'], errors='coerce')
    brazil_pivot['Year'] = pd.to_numeric(brazil_pivot['Year'], errors='coerce')
    merged_data['Year'] = pd.to_numeric(merged_data['Year'], errors='coerce')
    
    # Merge datasets with USA and Brazil
    usa_merged = pd.merge(usa_pivot, merged_data, on="Year", how="left").fillna(0)
    brazil_merged = pd.merge(brazil_pivot, merged_data, on="Year", how="left").fillna(0)
    
    # Test the RandomForest model for USA data
    usa_importances, _ = train_and_get_importance(usa_merged, target_col="GDP")
    assert not usa_importances.empty, "USA feature importances are empty"
    assert "Feature" in usa_importances.columns, "Feature column missing in USA importance"
    assert "Importance" in usa_importances.columns, "Importance column missing in USA importance"
    
    # Test the RandomForest model for Brazil data
    brazil_importances, _ = train_and_get_importance(brazil_merged, target_col="GDP")
    assert not brazil_importances.empty, "Brazil feature importances are empty"
    assert "Feature" in brazil_importances.columns, "Feature column missing in Brazil importance"
    assert "Importance" in brazil_importances.columns, "Importance column missing in Brazil importance"

# Test for extracting top indicators
def test_extract_top_indicators():
    gdp_data, country_metadata, usa_data, brazil_data = load_datasets()
    gdp_data = preprocess_gdp_data(gdp_data)
    country_metadata = preprocess_country_metadata(country_metadata)
    merged_data = merge_data(gdp_data, country_metadata)
    
    # Preprocess USA and Brazil data
    usa_pivot = preprocess_indicator_data(usa_data)
    brazil_pivot = preprocess_indicator_data(brazil_data)
    
    # Merge datasets with USA and Brazil
    usa_merged = pd.merge(usa_pivot, merged_data, on="Year", how="left").fillna(0)
    brazil_merged = pd.merge(brazil_pivot, merged_data, on="Year", how="left").fillna(0)
    
    # Train models and get feature importances
    usa_importances, _ = train_and_get_importance(usa_merged, target_col="GDP")
    brazil_importances, _ = train_and_get_importance(brazil_merged, target_col="GDP")
    
    # Extract top 5 features
    usa_top_indicators = extract_top_indicators(usa_importances, top_n=5)
    brazil_top_indicators = extract_top_indicators(brazil_importances, top_n=5)
    
    assert len(usa_top_indicators) == 5, "Incorrect number of USA top indicators"
    assert len(brazil_top_indicators) == 5, "Incorrect number of Brazil top indicators"

# Test for saving data to SQLite (mock database connection)
def test_save_to_sqlite():
    gdp_data, country_metadata, usa_data, brazil_data = load_datasets()
    gdp_data = preprocess_gdp_data(gdp_data)
    country_metadata = preprocess_country_metadata(country_metadata)
    merged_data = merge_data(gdp_data, country_metadata)
    
    # Preprocess USA and Brazil data
    usa_pivot = preprocess_indicator_data(usa_data)
    brazil_pivot = preprocess_indicator_data(brazil_data)
    
    # Merge datasets with USA and Brazil
    usa_merged = pd.merge(usa_pivot, merged_data, on="Year", how="left").fillna(0)
    brazil_merged = pd.merge(brazil_pivot, merged_data, on="Year", how="left").fillna(0)
    
    # Extract top indicators
    usa_importances, _ = train_and_get_importance(usa_merged, target_col="GDP")
    brazil_importances, _ = train_and_get_importance(brazil_merged, target_col="GDP")
    usa_top_indicators = extract_top_indicators(usa_importances, top_n=5)
    brazil_top_indicators = extract_top_indicators(brazil_importances, top_n=5)
    
    # Check if SQLite saving works (no exceptions)
    try:
        save_to_sqlite(usa_top_indicators, brazil_top_indicators, usa_merged, brazil_merged)
    except Exception as e:
        pytest.fail(f"Saving to SQLite failed with error: {e}")

if __name__ == "__main__":
    pytest.main()
