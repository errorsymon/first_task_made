import pytest
import pandas as pd
from project.pipeline import (
    load_datasets, 
    preprocess_gdp_data, 
    preprocess_country_metadata, 
    preprocess_indicator_data, 
    merge_data, 
    train_and_get_importance, 
    extract_top_indicators, 
    save_to_sqlite
)

# Mock datasets for testing
@pytest.fixture
def mock_data():
    # Provide mock data that matches the structure of the actual datasets
    gdp_data = pd.DataFrame({
        'Country Name': ['USA', 'Brazil'],
        'Country Code': ['USA', 'BRA'],
        '2000': [1000000, 900000],
        '2001': [1100000, 950000],
        '2002': [1200000, 980000]
    })
    country_metadata = pd.DataFrame({
        'Country Code': ['USA', 'BRA'],
        'Region': ['North America', 'South America'],
        'IncomeGroup': ['High income', 'Upper middle income']
    })
    usa_data = pd.DataFrame({
        'Indicator Code': ['Indicator1', 'Indicator2'],
        '2000': [10, 15],
        '2001': [12, 16],
        '2002': [14, 17]
    })
    brazil_data = pd.DataFrame({
        'Indicator Code': ['Indicator1', 'Indicator2'],
        '2000': [20, 25],
        '2001': [22, 26],
        '2002': [24, 27]
    })
    return gdp_data, country_metadata, usa_data, brazil_data

def test_extract_top_indicators(mock_data):
    gdp_data, country_metadata, usa_data, brazil_data = mock_data

    # Preprocess data
    gdp_data = preprocess_gdp_data(gdp_data)
    country_metadata = preprocess_country_metadata(country_metadata)
    merged_data = merge_data(gdp_data, country_metadata)

    usa_pivot = preprocess_indicator_data(usa_data)
    brazil_pivot = preprocess_indicator_data(brazil_data)

    # Merge dataframes with USA and Brazil
    usa_merged = pd.merge(usa_pivot, merged_data, on="Year", how="left").fillna(0)
    brazil_merged = pd.merge(brazil_pivot, merged_data, on="Year", how="left").fillna(0)

    # Train models and get feature importances
    usa_importances, _ = train_and_get_importance(usa_merged, target_col="GDP")
    brazil_importances, _ = train_and_get_importance(brazil_merged, target_col="GDP")

    # Test extracting top indicators for USA
    top_usa_indicators = extract_top_indicators(usa_importances, top_n=5)
    top_brazil_indicators = extract_top_indicators(brazil_importances, top_n=5)

    # Assert that the top indicators are correctly extracted and are in the 'Feature' column
    assert isinstance(top_usa_indicators, list)
    assert isinstance(top_brazil_indicators, list)
    assert len(top_usa_indicators) == 5
    assert len(top_brazil_indicators) == 5
    assert all(isinstance(indicator, str) for indicator in top_usa_indicators)
    assert all(isinstance(indicator, str) for indicator in top_brazil_indicators)

def test_save_to_sqlite(mock_data, tmpdir):
    gdp_data, country_metadata, usa_data, brazil_data = mock_data

    # Preprocess data
    gdp_data = preprocess_gdp_data(gdp_data)
    country_metadata = preprocess_country_metadata(country_metadata)
    merged_data = merge_data(gdp_data, country_metadata)

    usa_pivot = preprocess_indicator_data(usa_data)
    brazil_pivot = preprocess_indicator_data(brazil_data)

    # Merge dataframes with USA and Brazil
    usa_merged = pd.merge(usa_pivot, merged_data, on="Year", how="left").fillna(0)
    brazil_merged = pd.merge(brazil_pivot, merged_data, on="Year", how="left").fillna(0)

    # Train models and get feature importances
    usa_importances, _ = train_and_get_importance(usa_merged, target_col="GDP")
    brazil_importances, _ = train_and_get_importance(brazil_merged, target_col="GDP")

    # Extract top indicators
    usa_top_indicators = extract_top_indicators(usa_importances, top_n=5)
    brazil_top_indicators = extract_top_indicators(brazil_importances, top_n=5)

    # Use a temporary directory for testing
    temp_dir = tmpdir.mkdir("test_data")

    # Save to SQLite database
    save_to_sqlite(usa_top_indicators, brazil_top_indicators, usa_merged, brazil_merged)

    # Check that the SQLite database was created in the temporary directory
    db_path = os.path.join(str(temp_dir), "gdp_brazil_usa.db")
    assert os.path.exists(db_path)

    # Check if the tables 'usa_top_indicators' and 'brazil_top_indicators' exist in the SQLite database
    conn = sqlite3.connect(db_path)
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql(query, conn)
    conn.close()
    
    assert 'usa_top_indicators' in tables['name'].values
    assert 'brazil_top_indicators' in tables['name'].values

