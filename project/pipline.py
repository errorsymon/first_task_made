import pandas as pd
import sqlite3
import os
from contextlib import closing


def load_datasets():
    """
    Load the GDP datasets and metadata from the provided files on the local system.
    """
    # Local file paths (use raw URLs)
    gdp_data_file = "https://raw.githubusercontent.com/errorsymon/Data/d710147cfb374060422bd86a1889d33e54fa3f2b/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_9865.csv"
    country_metadata_file = "https://raw.githubusercontent.com/errorsymon/Data/d710147cfb374060422bd86a1889d33e54fa3f2b/Metadata_Country_API_NY.GDP.MKTP.CD_DS2_en_csv_v2_9865.csv"
    indicator_metadata_file = "https://raw.githubusercontent.com/errorsymon/Data/b906c2e79b027e4ad1df257277ede4b8f9884ff2/Metadata_Indicator_API_NY.GDP.MKTP.CD_DS2_en_csv_v2_9865.csv"
    
    # Load datasets
    gdp_data = pd.read_csv(gdp_data_file, skiprows=4, on_bad_lines='skip')
    country_metadata = pd.read_csv(country_metadata_file)
    indicator_metadata = pd.read_csv(indicator_metadata_file)
    
    return gdp_data, country_metadata, indicator_metadata

def preprocess_gdp_data(gdp_data):
    """
    Preprocess the GDP data: Filter relevant columns and clean up the data.
    """
    # Keep only country name, country code, and GDP data for the years
    years = [col for col in gdp_data.columns if col.isdigit()]  # Extract year columns
    relevant_columns = ['Country Name', 'Country Code'] + years
    gdp_data = gdp_data[relevant_columns]
    
    # Melt the data to long format
    gdp_data_melted = gdp_data.melt(id_vars=['Country Name', 'Country Code'], 
                                    var_name='Year', 
                                    value_name='GDP')
    gdp_data_melted['Year'] = gdp_data_melted['Year'].astype(int)  # Convert year to integer
    gdp_data_melted.dropna(subset=['GDP'], inplace=True)  # Drop rows with missing GDP values
    
    return gdp_data_melted

def preprocess_country_metadata(country_metadata):
    """
    Preprocess the country metadata: Keep relevant columns.
    """
    relevant_columns = ['Country Code', 'Region', 'IncomeGroup']
    country_metadata = country_metadata[relevant_columns]
    return country_metadata

def merge_data(gdp_data, country_metadata):
    """
    Merge the GDP data with the country metadata.
    """
    merged_df = pd.merge(gdp_data, country_metadata, on='Country Code', how='left')
    return merged_df

def main():
    # Load datasets
    gdp_data, country_metadata, _ = load_datasets()
    
    # Preprocess datasets
    gdp_data_cleaned = preprocess_gdp_data(gdp_data)
    country_metadata_cleaned = preprocess_country_metadata(country_metadata)
    
    # Merge datasets
    merged_df = merge_data(gdp_data_cleaned, country_metadata_cleaned)
    
    # Debugging Step 1: Print a portion of the merged data to check the country names
    print("\nPreview of merged data with country names:")
    print(merged_df[['Country Name', 'Country Code']].head(20))
    
    # Remove leading/trailing spaces from country names and codes
    merged_df['Country Name'] = merged_df['Country Name'].str.strip()
    merged_df['Country Code'] = merged_df['Country Code'].str.strip()
    
    # Debugging Step 2: Ensure 'Brazil' and 'United States' exist in the dataset
    print("\nChecking if Brazil and United States are in the dataset (before filtering):")
    print(merged_df[merged_df['Country Name'].str.contains('brazil', case=False)])
    print(merged_df[merged_df['Country Name'].str.contains('united states', case=False)])
    
    # Apply both country name and country code filtering (case insensitive)
    countries_of_interest = ['brazil', 'united states']
    country_codes_of_interest = ['BRA', 'USA']

    filtered_df = merged_df[merged_df['Country Name'].str.contains('|'.join(countries_of_interest), case=False) |
                            merged_df['Country Code'].isin(country_codes_of_interest)]

    # Debugging Step 3: Check filtered DataFrame
    print("\nFiltered DataFrame (should include Brazil and USA):")
    print(filtered_df[['Country Name', 'Country Code']].head(20))
    
    # Create an SQLite database connection
    conn = sqlite3.connect('gdp_brazil_usa.db')
    cursor = conn.cursor()
    
    # Store the filtered DataFrame in the SQLite database
    filtered_df.to_sql('gdp_data', conn, if_exists='replace', index=False)
    
    # Commit and close the connection
    conn.commit()
    
    # Query all rows from the filtered data table
    query = "SELECT * FROM gdp_data LIMIT 5"
    queried_df = pd.read_sql_query(query, conn)
    
    # Display the first few rows of the queried DataFrame to confirm
    print("Queried DataFrame from SQLite:")
    print(queried_df)
    
    # Close the connection
    conn.close()
    
    # Save the filtered DataFrame to a CSV file
    output_csv = 'gdp_brazil_usa.csv'
    filtered_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Filtered dataset saved as {output_csv}")

if __name__ == "__main__":
    main()
