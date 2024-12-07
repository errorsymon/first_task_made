import pytest
import pandas as pd
from pipeline import (
    load_datasets,
    preprocess_gdp_data,
    preprocess_country_metadata,
    merge_data,
    calculate_gdp_growth_rate
)

# Mock data for testing
@pytest.fixture
def mock_gdp_data():
    return pd.DataFrame({
        "Country Name": ["United States", "Brazil", "India"],
        "Country Code": ["USA", "BRA", "IND"],
        "1989": [1000, 500, 200],
        "1990": [1100, 600, 300],
        "1991": [1200, None, 400],
    })

@pytest.fixture
def mock_country_metadata():
    return pd.DataFrame({
        "Country Code": ["USA", "BRA", "IND"],
        "Region": ["North America", "Latin America", "Asia"],
        "IncomeGroup": ["High income", "Upper middle income", "Lower middle income"]
    })

@pytest.fixture
def mock_filtered_gdp():
    return pd.DataFrame({
        "Country Name": ["Brazil", "Brazil", "United States", "United States"],
        "Country Code": ["BRA", "BRA", "USA", "USA"],
        "Year": [1989, 1990, 1989, 1990],
        "GDP": [500, 600, 1000, 1100]
    })

# Test loading datasets
def test_load_datasets():
    gdp_data, country_metadata, indicator_metadata, indicators_gdp_usa, indicators_gdp_bra = load_datasets()
    assert isinstance(gdp_data, pd.DataFrame)
    assert isinstance(country_metadata, pd.DataFrame)
    assert isinstance(indicator_metadata, pd.DataFrame)
    assert isinstance(indicators_gdp_usa, pd.DataFrame)
    assert isinstance(indicators_gdp_bra, pd.DataFrame)

# Test preprocessing GDP data
def test_preprocess_gdp_data(mock_gdp_data):
    gdp_data_cleaned = preprocess_gdp_data(mock_gdp_data)
    assert "Year" in gdp_data_cleaned.columns
    assert "GDP" in gdp_data_cleaned.columns
    assert gdp_data_cleaned['Country Code'].nunique() == 1  # Only USA data is expected after filtering

# Test preprocessing country metadata
def test_preprocess_country_metadata(mock_country_metadata):
    country_metadata_cleaned = preprocess_country_metadata(mock_country_metadata)
    assert set(country_metadata_cleaned.columns) == {"Country Code", "Region", "IncomeGroup"}
    assert len(country_metadata_cleaned) == 3

# Test merging data
def test_merge_data(mock_gdp_data, mock_country_metadata):
    gdp_data_cleaned = preprocess_gdp_data(mock_gdp_data)
    country_metadata_cleaned = preprocess_country_metadata(mock_country_metadata)
    merged_df = merge_data(gdp_data_cleaned, country_metadata_cleaned)
    assert "Region" in merged_df.columns
    assert "IncomeGroup" in merged_df.columns
    assert len(merged_df) == len(gdp_data_cleaned)

# Test calculating GDP growth rate
def test_calculate_gdp_growth_rate(mock_filtered_gdp):
    result = calculate_gdp_growth_rate(mock_filtered_gdp)
    assert "GDP Growth Rate" in result.columns
    assert not result["GDP Growth Rate"].isnull().all()  # Ensure some growth rates are calculated
    assert result.iloc[1]["GDP Growth Rate"] == pytest.approx(20.0, rel=1e-2)  # Example check for Brazil (1989 to 1990)

# Run tests with pytest
if __name__ == "__main__":
    pytest.main()
