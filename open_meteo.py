#%%
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import itertools
import numpy as np
import torch
from scipy.spatial.distance import cdist
def nsew_to_lat_long(coordinates):
    split = coordinates.split(' ')
    ns = (split[0].strip('°'))
    if 'S' in ns:
        ns = '-' + ns
    ew = (split[1].strip('°'))
    if 'W' in ew:
        ew = '-' + ew
    latitude = ns[:-2]
    longitude = ew[:-2]
    return float(latitude), float(longitude) # latitude, longitud


def get_4month_weather(lat, lon, start_date, end_date):
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Make sure all required weather variables are listed here
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            'relative_humidity_2m',
            'wind_speed_10m',
            'surface_pressure'
        ],
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location
    response = responses[0]

# print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")

    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_rel_hum_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(2).ValuesAsNumpy()
    hourly_surf_pressure = hourly.Variables(3).ValuesAsNumpy()
    # print(hourly_temperature_2m)
    h_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }
    
    h_data['temperature_2m'] = hourly_temperature_2m
    h_data['rel_hum_2m'] = hourly_rel_hum_2m
    h_data['hourly_wind_speed_10m'] = hourly_wind_speed_10m
    h_data['surf_pressure'] = hourly_surf_pressure
    h_dataframe = pd.DataFrame(data=h_data)

    return h_dataframe

def get_weather_batch(
    coords: list,
    start_d: str,
    end_d: str
) -> list:
    list_weather = []
    for coorde in coords:
        lat, lon = nsew_to_lat_long(coorde)
        
        df_w = get_4month_weather(lat, lon, start_date=start_d,
                           end_date=end_d)
        list_weather.append(df_w)
    # print(list_weather)
    return list_weather

def gen_distance_matrix(list_coords: list) -> pd.DataFrame:
    n_nodes = [f'node_{i+1}' for i in range(len(list_coords))]
    
    distance_matrix = pd.DataFrame(0, columns=n_nodes, index = n_nodes)
    for i, j in itertools.combinations(n_nodes,2):
        index = int(i.split('_')[1])-1
        index2 = int(j.split('_')[1])-1
        coord1 = np.array([nsew_to_lat_long(list_coords[index])])
        coord2 = np.array([nsew_to_lat_long(list_coords[index2])])
  
        distance_matrix.at[i,j] = 1/cdist(coord1, coord2)
        distance_matrix.at[j,i] = distance_matrix.loc[i, j]
    return distance_matrix

def format_result(list_df, 
                  map_channel):
    list_nodes = np.arange(1, len(list_df)+1)
    list_parameters = np.arange(1, len(list_df[0].columns[1:])+1)
    list_two_col = list(itertools.product(list_nodes, list_parameters))
    t = len(list_df[0].index)
    n = len(list_nodes)
    f = len(list_parameters)
    tensor_results = torch.zeros([t,n ,f], dtype = torch.float32)
    for i in list_nodes:
        # print(i)
        tensor_results[:, i-1, :] = torch.tensor(list_df[i-1].iloc[:, 1:].values, dtype = torch.float32)
    return tensor_results

        
    
    
def distance_to_connectivity(distance_matrix):
    edges = []
    edges_w = []
    
    for i, j in itertools.combinations(distance_matrix.columns, 2):
        edges.append([i,j])
        edges_w_value = distance_matrix.loc[i,j]
        edges_w.append(edges_w_value)
        
    return np.array(edges).T, np.array(edges_w)
    
#%%
if __name__ == "__main__":
    # Test parameters
    pin1 = '34.455737°S 70.925231°W'
    pin2 = '34.452217°S 70.920077°W'
    pin3 = '34.454563°S 70.911596°W'
    pin4 = '34.457393°S 70.908903°W'
    pin5 = '34.460729°S 70.914312°W'
    pin6 = '34.456189°S 70.916805°W'
    coords = [
        pin1, 
        pin2,
        pin3,
        pin4,
        pin5,
        pin6
    ]
    # lat, lon = nsew_to_lat_long('34.455737°S 70.925231°W')
    # # print(lat, long)
    start_date = "2021-11-01"
    end_date = "2024-03-01"  # 4 months later


    # Get the weather data
    w_list = get_weather_batch(coords,
                               start_d=start_date,
                               end_d=end_date)

    distance_matrix = gen_distance_matrix(coords)
    edges, edges_w = distance_to_connectivity(distance_matrix)
    
    channels_tens = format_result(w_list,
                                map_channel={
                                    1: 'temperature_2m',
                                    2: 'rel_hum_2m',
                                    3: 'hourly_wind_speed_10m',
                                    4: 'surf_pressure'
                                })
    channels_df
# %%

