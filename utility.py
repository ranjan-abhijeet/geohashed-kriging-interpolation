import os
import glob
import pygeohash
import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from pykrige.ok import OrdinaryKriging

geohashing_function = np.vectorize(pygeohash.encode)
GEOHASH_PRECISION = 8

def create_directories(directory_names: list):
    """
    Creates the directories for storing un-processed and processed data
    
    Args:
    directory_names (list)  : list of names(as string) for  
    """

    for directory in directory_names:
        if not os.path.exists(directory):
            os.makedirs(directory)

def create_clusters(input_csv_folder_path: str, output_csv_folder: str="", cluster_size: int=150):
    """
    Divides large dataset into smaller chunks using kmeans clustering, reducing the time it takes to implement
    kriging interpolation technique.
    
    Args:
    input_data_folder (str) : path of input file.
    output_data_folder (str): path of folder where clustered files will reside.
    cluster_size (int) : approximate number of data points which should be present in the cluster
    """
    input_file = glob.glob(f"{input_csv_folder_path}/*.csv")
    if len(input_file) == 0:
        return False
    df = pd.read_csv(input_file[0])
    num_clusters = len(df)//cluster_size
    coordinate_array = df.loc[:, ['latitude', 'longitude', 'pm2.5']]
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++')
    kmeans.fit(coordinate_array[coordinate_array.columns[0:2]])
    coordinate_array['cluster_label'] = kmeans.fit_predict(coordinate_array[coordinate_array.columns[0:2]]) 
    for cluster in range(num_clusters):
        df_cluster = coordinate_array[coordinate_array['cluster_label'] == cluster]
        df_cluster.to_csv(f"{output_csv_folder}/{cluster}.csv", index=False)
    return True

def apply_kriging(input_csv_files_path: str, output_csv_files_path: str, grid_space: float=0.01):
    """
    Applies Ordinary Kriging method on the clustered, smaller datasets.

    Args:
    input_csv_files_path (str) : file path of clustered csv files.
    output_csv_files_path (str) : file path where output should be written.
    grid_space (float) : grid interval for output data.
    """
    file_list = glob.glob(f"{input_csv_files_path}/*.csv")
    for index, file in enumerate(file_list):
        df = pd.read_csv(file)
        lons = np.array(df['longitude'])
        lats = np.array(df['latitude'])
        pm25 = np.array(df['pm2.5'])

        # Defining the grid
        longitude_grid = np.arange(np.amin(lons), np.amax(lons), grid_space)
        latitude_grid = np.arange(np.amin(lats), np.amax(lats), grid_space)
        
        # Perform Ordinary Kriging on given data
        kriging_model = OrdinaryKriging(lons, lats, pm25, variogram_model='spherical', verbose=False, enable_plotting=False,nlags=20)
        interpolated_data, _ = kriging_model.execute('grid', longitude_grid, latitude_grid)

        xintrp, yintrp = np.meshgrid(longitude_grid, latitude_grid)
        x_comp = np.array(xintrp)
        y_comp = np.array(yintrp)

        return_df = pd.DataFrame(columns=["latitude", "longitude", "pm2.5", "geohash"])
        return_df["longitude"] = x_comp.flatten()
        return_df["latitude"] = y_comp.flatten()
        return_df["pm2.5"] = interpolated_data.flatten()
        return_df["geohash"] = geohashing_function(return_df["latitude"], return_df["longitude"], precision=GEOHASH_PRECISION)
        return_df.to_csv(f"{output_csv_files_path}/{index}.csv", index=False)


def merge_dataframes(input_file_path: str, output_file_path: str, filename: str="krigged_data.csv"):
    file_list = glob.glob(f"{input_file_path}/*.csv")
    dataframes = []
    for file in file_list:
        df = pd.read_csv(file)
        dataframes.append(df)
    output = pd.concat(dataframes, axis=0)
    print(f"[+] Shape of output data: {output.shape}")
    output.to_csv(f"{output_file_path}/{filename}", index=False)