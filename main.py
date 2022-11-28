import utility
import time

project_structure = ["input_data", "clustered_data", "krigged_data", "merged_data"]

def interpolate_data(
                    input_data: str,
                    clustered_data: str,
                    krigged_data: str,
                    merged_data: str,
                    cluster_size: int,
                    grid_space:float
                    ):
    utility.create_directories(project_structure)
    if utility.create_clusters(input_data, clustered_data, cluster_size) is None:
        print("[-] Missing input file to interpolate data")
        return
    utility.apply_kriging(clustered_data, krigged_data, grid_space)
    utility.merge_dataframes(krigged_data, merged_data)

if __name__ == "__main__":
    start = time.time()
    interpolate_data("input_data", "clustered_data", "krigged_data", "merged_data", 100, 0.01)
    end = time.time()
    print(f"[+] Total time: {round(end-start, 2)} seconds")