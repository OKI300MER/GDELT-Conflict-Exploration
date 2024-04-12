# Import libraies for code
import requests
from bs4 import BeautifulSoup
import os
import zipfile
import pandas as pd

'''The first part of this code is for pulling data from GDELT website and turning many files into one data set.'''

# Function to download and extract cvs files from GDELT website.

def download_and_extract(url, output_dir):
    # Download the zip file
    
    response = requests.get(url)
    zip_file_path = os.path.join(output_dir, url.split('/')[-1])
    with open(zip_file_path, 'wb') as f:
        f.write(response.content)
    
    # Extract the CSV file from the zip archive
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        csv_file = zip_ref.namelist()[0]  # Assuming there's only one CSV file in each zip
        zip_ref.extract(csv_file, output_dir)
    
    return os.path.join(output_dir, csv_file)

# URL of the webpage containing links to zip files

webpage_url = 'http://data.gdeltproject.org/events/index.html'

# Output directory to save downloaded files

output_dir = r'C:\Users\shric\Desktop\Dai\assignments\CAPSTONE\assignments\CAPSTONE\data'

# Create the output directory if it doesn't exist

os.makedirs(output_dir, exist_ok=True)

# Get the webpage content

response = requests.get(webpage_url)
if response.status_code == 200:
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    # Find all links on the page
    links = soup.find_all('a')
    for link in links:
        href = link.get('href')
        if href and href.endswith('.CSV.zip'):
            zip_url = 'http://data.gdeltproject.org/events/' + href
            csv_file_path = download_and_extract(zip_url, output_dir)
            print(f"Downloaded and extracted: {csv_file_path}")
            
# Directory containing the CSV files

csv_dir = r'C:\Users\shric\Desktop\Dai\assignments\CAPSTONE\assignments\CAPSTONE\data'

# Initialize an empty DataFrame to store the concatenated data

combined_df = pd.DataFrame()

# Iterate over CSV files in the directory

for file in os.listdir(csv_dir):
    if file.endswith('.CSV'):  # Adjust the extension as per your file format
        file_path = os.path.join(csv_dir, file)
        print(f"Reading file: {file_path}")
        
        # Read the CSV file with tab delimiter and replace empty fields with NaN values
        
        try:
            for i, chunk in enumerate(pd.read_csv(file_path, sep='\t', chunksize=5000, na_values=['', ' '])):
                print(f"Processing chunk {i+1} of file {file_path} with shape {chunk.shape}")
                combined_df = pd.concat([combined_df, chunk])
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

# Save the combined DataFrame to a single CSV file

combined_df.to_csv('combined_data.csv', index=False)
print("Combined data saved to combined_data.csv")

''' The second part of this code is for actually cleaning and working the data'''