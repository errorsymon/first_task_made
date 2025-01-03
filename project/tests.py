import sys
import os
import pytest

# Add the project folder to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'project')))

from pipeline import (
    load_datasets,
    preprocess_gdp_data,
    preprocess_country_metadata,
    preprocess_indicator_data,
    merge_data,
    train_and_get_importance,
    extract_top_indicators,
    save_to_sqlite,
)

@pytest.fixture
def datasets():
    """
    Fixture to load and return the datasets used for testing.
    """
    gdp_data, country_metadata, usa_data, brazil_data = load_datasets()
    gdp_data = preprocess_gdp_data(gdp_data)
    country_metadata = preprocess_country_metadata(country_metadata)
    merged_data = merge_data(gdp_data, country_metadata)

    # Preprocess USA and Brazil data
    usa_pivot = preprocess_indicator_data(usa_data)
    brazil_pivot = preprocess_indicator_data(brazil_data)

    return gdp_data, country_metadata, merged_data, usa_pivot, brazil_pivot

def test_preprocess_gdp_data(datasets):
    gdp_data, country_metadata, merged_data, usa_pivot, brazil_pivot = datasets
    assert not gdp_data.empty, "GDP data should not be empty"
    assert 'Country Name' in gdp_data.columns, "GDP data should have 'Country Name' column"
    assert 'GDP' in gdp_data.columns, "GDP data should have 'GDP' column"

def test_train_and_get_importance(datasets):
    gdp_data, country_metadata, merged_data, usa_pivot, brazil_pivot = datasets
    
    # Debugging step: Print the first few rows to verify the data
    print("USA Pivot Head:\n", usa_pivot.head())
    print("Brazil Pivot Head:\n", brazil_pivot.head())

    # Check if 'GDP' column exists in both dataframes
    assert 'GDP' in usa_pivot.columns, "'GDP' column not found in USA pivot data"
    assert 'GDP' in brazil_pivot.columns, "'GDP' column not found in Brazil pivot data"

    # Run the importance function
    usa_importance, _ = train_and_get_importance(usa_pivot, target_col="GDP")
    brazil_importance, _ = train_and_get_importance(brazil_pivot, target_col="GDP")

    assert not usa_importance.empty, "USA feature importance should not be empty"
    assert not brazil_importance.empty, "Brazil feature importance should not be empty"
    assert 'Feature' in usa_importance.columns, "USA importance data should have 'Feature' column"
    assert 'Importance' in usa_importance.columns, "USA importance data should have 'Importance' column"



def test_extract_top_indicators(datasets):
    gdp_data, country_metadata, merged_data, usa_pivot, brazil_pivot = datasets
    # Train and get feature importance
    usa_importance, _ = train_and_get_importance(usa_pivot, target_col="GDP")
    brazil_importance, _ = train_and_get_importance(brazil_pivot, target_col="GDP")

    # Extract top 5 indicators
    top_usa_indicators = extract_top_indicators(usa_importance, top_n=5)
    top_brazil_indicators = extract_top_indicators(brazil_importance, top_n=5)

    assert len(top_usa_indicators) == 5, "There should be 5 top USA indicators"
    assert len(top_brazil_indicators) == 5, "There should be 5 top Brazil indicators"
    assert isinstance(top_usa_indicators, list), "Top indicators should be a list"
    assert isinstance(top_brazil_indicators, list), "Top indicators should be a list"
    
def test_save_to_sqlite(datasets):
    gdp_data, country_metadata, merged_data, usa_pivot, brazil_pivot = datasets
    # Train and get feature importance
    usa_importance, _ = train_and_get_importance(usa_pivot, target_col="GDP")
    brazil_importance, _ = train_and_get_importance(brazil_pivot, target_col="GDP")

    # Extract top 5 indicators
    usa_top_indicators = extract_top_indicators(usa_importance, top_n=5)
    brazil_top_indicators = extract_top_indicators(brazil_importance, top_n=5)

    # Save to SQLite database
    save_to_sqlite(usa_top_indicators, brazil_top_indicators, merged_data, merged_data)
    
    # Ensure the SQLite file is created (you may want to check specific table contents as well)
    import os
    assert os.path.exists('gdp_brazil_usa.db'), "SQLite database was not created"

def test_final_data(datasets):
    gdp_data, country_metadata, merged_data, usa_pivot, brazil_pivot = datasets
    # Train and get feature importance
    usa_importance, _ = train_and_get_importance(usa_pivot, target_col="GDP")
    brazil_importance, _ = train_and_get_importance(brazil_pivot, target_col="GDP")

    # Extract top 5 indicators
    usa_top_indicators = extract_top_indicators(usa_importance, top_n=5)
    brazil_top_indicators = extract_top_indicators(brazil_importance, top_n=5)

    # Create and save final_data.csv
    final_file = 'final_data.csv'
    with open(final_file, 'w') as f:
        f.write("USA Data\n")
        usa_top_indicators_data = merged_data[['Year'] + usa_top_indicators].assign(Country='USA')
        usa_top_indicators_data.to_csv(f, index=False)
        f.write("\nBrazil Data\n")
        brazil_top_indicators_data = merged_data[['Year'] + brazil_top_indicators].assign(Country='Brazil')
        brazil_top_indicators_data.to_csv(f, index=False)

    # Check if final_data.csv was created
    assert os.path.exists(final_file), "final_data.csv was not created"
    
    # You can add more specific checks to validate the data within the CSV
    with open(final_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) > 0, "final_data.csv is empty"
