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
    indicator_metadata_file2 = "https://raw.githubusercontent.com/errorsymon/Data/refs/heads/main/API_USA_DS2_en_csv_v2_3173.csv"
    indicator_metadata_file3 = "https://raw.githubusercontent.com/errorsymon/Data/refs/heads/main/brazil.csv"
    
    # Load datasets
    def load_datasets():
     gdp_data = None
     country_metadata = None
     indicator_metadata = None
     indicators_gdp_usa = None
     indicators_gdp_bra = None

    try:
        gdp_data = pd.read_csv("https://raw.githubusercontent.com/errorsymon/Data/d710147cfb374060422bd86a1889d33e54fa3f2b/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_9865.csv", skiprows=4, on_bad_lines='skip')
    except Exception as e:
        print(f"Error loading GDP data: {e}")

    try:
        country_metadata = pd.read_csv("https://raw.githubusercontent.com/errorsymon/Data/d710147cfb374060422bd86a1889d33e54fa3f2b/Metadata_Country_API_NY.GDP.MKTP.CD_DS2_en_csv_v2_9865.csv")
    except Exception as e:
        print(f"Error loading country metadata: {e}")

    try:
        indicator_metadata = pd.read_csv("https://raw.githubusercontent.com/errorsymon/Data/b906c2e79b027e4ad1df257277ede4b8f9884ff2/Metadata_Indicator_API_NY.GDP.MKTP.CD_DS2_en_csv_v2_9865.csv")
    except Exception as e:
        print(f"Error loading indicator metadata: {e}")

    try:
        indicators_gdp_usa = pd.read_csv("https://raw.githubusercontent.com/errorsymon/Data/refs/heads/main/API_USA_DS2_en_csv_v2_3173.csv")
    except Exception as e:
        print(f"Error loading USA indicators: {e}")

    try:
        indicators_gdp_bra = pd.read_csv("https://raw.githubusercontent.com/errorsymon/Data/refs/heads/main/brazil.csv")
    except Exception as e:
        print(f"Error loading Brazil indicators: {e}")

    return gdp_data, country_metadata, indicator_metadata, indicators_gdp_usa, indicators_gdp_bra



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

def preprocess_usa_brazil_data(indicators_gdp_usa):
    """
    Preprocess the USA and Brazil GDP data: Remove first 5 rows from the data.
    """
    # Skip the first 5 rows
    df = indicators_gdp_usa.iloc[5:].reset_index(drop=True)
    return df

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
    gdp_data, country_metadata, indicator_metadata, indicators_gdp_bra, indicators_gdp_usa = load_datasets()
    
    # Preprocess datasets
    gdp_data_cleaned = preprocess_gdp_data(gdp_data)
    country_metadata_cleaned = preprocess_country_metadata(country_metadata)
    
    # Merge datasets
    merged_df = merge_data(gdp_data_cleaned, country_metadata_cleaned)
    
    # Debugging Step 1: Print a portion of the merged data to check the country names
    print("\nPreview of merged data with country names:")
    print(merged_df[['Country Name', 'Country Code']].head())
    
    # Remove leading/trailing spaces from country names and codes
    merged_df['Country Name'] = merged_df['Country Name'].str.strip()
    merged_df['Country Code'] = merged_df['Country Code'].str.strip()
    
    # Debugging Step 2: Ensure 'Brazil' and 'United States' exist in the dataset
    print("\nChecking if Brazil and United States are in the dataset (before filtering):")
    print(merged_df[merged_df['Country Name'].str.contains('brazil', case=False)].head())
    print(merged_df[merged_df['Country Name'].str.contains('united states', case=False)].head())
    
    # Apply both country name and country code filtering (case insensitive)
    countries_of_interest = ['brazil', 'united states']
    country_codes_of_interest = ['BRA', 'USA']

    filtered_df = merged_df[merged_df['Country Name'].str.contains('|'.join(countries_of_interest), case=False) |
                            merged_df['Country Code'].isin(country_codes_of_interest)]

    # Debugging Step 3: Check filtered DataFrame
    print("\nFiltered DataFrame (should include Brazil and USA):")
    print(filtered_df[['Country Name', 'Country Code']].head())
    
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
    print(queried_df.head())
    
    # Close the connection
    conn.close()
    
    # Save the filtered DataFrame to a CSV file
    output_csv = os.path.join(os.getcwd(), 'gdp_brazil_usa.csv')  # Save in the current working directory
    filtered_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Filtered dataset saved as {output_csv}")

    # Ensure necessary preprocessing functions are called before merging data

# Load datasets
gdp_data, country_metadata, indicator_metadata, indicators_gdp_usa, indicators_gdp_bra = load_datasets()

# Preprocess GDP data
gdp_data_cleaned = preprocess_gdp_data(gdp_data)

# Preprocess country metadata
country_metadata_cleaned = preprocess_country_metadata(country_metadata)

# Merge GDP data with country metadata
merged_df = merge_data(gdp_data_cleaned, country_metadata_cleaned)

# Filter for Brazil and the United States
countries_of_interest = ['brazil', 'united states']
country_codes_of_interest = ['BRA', 'USA']

# Apply both country name and country code filtering (case insensitive)
filtered_df = merged_df[merged_df['Country Name'].str.contains('|'.join(countries_of_interest), case=False) |
                        merged_df['Country Code'].isin(country_codes_of_interest)]

# Filter for the years 1989 to 2023
filtered_df = filtered_df[filtered_df['Year'].between(1989, 2023)]

# Print the GDP rate for USA and Brazil from 1989 to 2023
print("\nGDP for Brazil and the United States from 1989 to 2023:")
for country in ['Brazil', 'United States']:
    country_data = filtered_df[filtered_df['Country Name'].str.contains(country, case=False)]
    print(f"\nGDP for {country}:")
    print(country_data[['Year', 'GDP']])


# Calculate GDP growth rate for a specific country
def calculate_gdp_growth_rate(df):
    df.loc[:, 'GDP Growth Rate'] = df['GDP'].pct_change() * 100  # Use .loc[] to avoid SettingWithCopyWarning
    return df

# Calculate GDP growth rate for Brazil
brazil_data = filtered_df[filtered_df['Country Name'].str.contains('brazil', case=False)]
brazil_data = calculate_gdp_growth_rate(brazil_data)

# Calculate GDP growth rate for the United States
usa_data = filtered_df[filtered_df['Country Name'].str.contains('united states', case=False)]
usa_data = calculate_gdp_growth_rate(usa_data)

# Print the GDP growth rate for Brazil and United States
print("\nGDP Growth Rate for Brazil and the United States (1989 to 2023):")

# Display GDP growth rates for Brazil
print("\nGDP Growth Rate for Brazil:")
print(brazil_data[['Year', 'GDP Growth Rate']])

# Display GDP growth rates for the United States
print("\nGDP Growth Rate for United States:")
print(usa_data[['Year', 'GDP Growth Rate']])
# Sort the data to find the top major changes in GDP growth rates
# For Brazil
brazil_top_changes = brazil_data.nlargest(5, 'GDP Growth Rate')
print("Top 5 GDP Growth Rate Changes for Brazil:")
print(brazil_top_changes[['Year', 'GDP Growth Rate']])

# For the United States
usa_top_changes = usa_data.nlargest(5, 'GDP Growth Rate')
print("\nTop 5 GDP Growth Rate Changes for United States:")
print(usa_top_changes[['Year', 'GDP Growth Rate']])


if __name__ == "__main__":
    main()
