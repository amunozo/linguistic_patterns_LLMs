import sys
import os

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import json
from datetime import datetime
import argparse
from pathlib import Path

# Try to load from .env file if it exists
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                if key == 'NYT_API_KEY':
                    os.environ[key] = value

# Get API key from environment variable
api_key = os.environ.get('NYT_API_KEY')
if not api_key or api_key == 'your_api_key_here':
    raise ValueError(
        "NYT_API_KEY is not properly configured.\n"
        "Please follow these steps:\n"
        "1. Copy .env.example to .env\n"
        "2. Edit .env and add your NYT API key\n"
        "3. Get your API key from: https://developer.nytimes.com/get-started"
    )

base_url = 'https://api.nytimes.com/svc/archive/v1/'

def download_data(start_date, end_date, output_dir):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    start_year, start_month = start_date.year, start_date.month
    end_year, end_month = end_date.year, end_date.month

    # Loop through the months and make API calls
    for year in range(start_year, end_year + 1):
        for month in range(start_month, 13 if year != end_year else end_month + 1):
            # Format the URL for the API call
            url = f'{base_url}{year}/{month}.json?api-key={api_key}'
            
            # Make the API call
            response = requests.get(url)
            
            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON response
                data = response.json()
                
                # Save the data to a file
                with open(os.path.join(output_dir, f'{year}_{month}.json'), 'w') as outfile:
                    json.dump(data, outfile)
                
                print(f'Downloaded data for {year}-{month}')
            else:
                print(f'Error downloading data for {year}-{month}. Status code: {response.status_code}')

def main():
    parser = argparse.ArgumentParser(description='Download NYT Archive Data.')
    parser.add_argument('start_date', type=lambda d: datetime.strptime(d, '%Y-%m'), help='Start date in YYYY-MM format')
    parser.add_argument('end_date', type=lambda d: datetime.strptime(d, '%Y-%m'), help='End date in YYYY-MM format')
    parser.add_argument('output_dir', type=str, help='Output directory for the downloaded data')

    args = parser.parse_args()
    
    download_data(args.start_date, args.end_date, args.output_dir)

if __name__ == '__main__':
    main()
