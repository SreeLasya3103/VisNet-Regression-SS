import requests
import pandas as pd
import time

# Define headers to mimic a browser request
HEADERS = {
    'User-Agent': 'FAA-Camera-Scraper/1.0',
    'Accept': 'application/json',
    'Referer': 'https://weathercams.faa.gov/',
}

# API endpoints
SITES_API = 'https://api.weathercams.faa.gov/sites'
SUMMARY_API = 'https://api.weathercams.faa.gov/summary'

def fetch_all_sites():
    """Fetches all FAA weather camera sites."""
    response = requests.get(SITES_API, headers=HEADERS)
    response.raise_for_status()
    sites = response.json().get('payload', [])
    return sites

def fetch_site_cameras(site_id):
    """Fetches camera data for a specific site."""
    params = {'siteId': str(site_id), 'related': 'true'}
    response = requests.get(SUMMARY_API, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json()

def main():
    print("Fetching all FAA weather camera sites...")
    sites = fetch_all_sites()
    print(f"Total sites retrieved: {len(sites)}")

    data = []

    for idx, site in enumerate(sites, 1):
        site_id = site.get('siteId')
        site_name = site.get('siteName')
        icao = site.get('icao')
        state = site.get('state')
        country = site.get('country')
        latitude = site.get('latitude')
        longitude = site.get('longitude')

        print(f"[{idx}/{len(sites)}] Processing site: {site_name} ({icao})")

        try:
            site_data = fetch_site_cameras(site_id)
            cameras = site_data.get('payload', {}).get('site', {}).get('cameras', [])
            for camera in cameras:
                direction = camera.get('cameraDirection')
                for image in camera.get('currentImages', []):
                    image_uri = image.get('imageUri')
                    image_datetime = image.get('imageDatetime')
                    data.append({
                        'ICAO': icao,
                        'Site Name': site_name,
                        'State': state,
                        'Country': country,
                        'Latitude': latitude,
                        'Longitude': longitude,
                        'Camera Direction': direction,
                        'Image URI': image_uri,
                        'Image Datetime': image_datetime
                    })
        except Exception as e:
            print(f"Error processing site {site_name} ({icao}): {e}")

        # Optional: Delay to prevent overwhelming the API
        time.sleep(0.1)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    output_file = 'faa_weather_cameras.csv'
    df.to_csv(output_file, index=False)
    print(f"\nData saved to {output_file}")

if __name__ == "__main__":
    main()
