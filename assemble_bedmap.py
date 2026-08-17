import argparse, time, os, re
import random
from tqdm import tqdm
import copy, math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from pyproj import Proj, Transformer, Geod

# Set the option to display all columns
pd.set_option('display.max_columns', None)

'''
We import 151 files from BedMap and create one single geodataframe
See BedMap: https://antarctica.github.io/PDC_GeophysicsBook/BEDMAP/Get_full_metadata_from_CSV_file.html
BedMap3 model: https://www.nature.com/articles/s41597-025-04672-y
Bedmap data: https://essd.copernicus.org/articles/15/2695/2023/
# - Lines: https://antarctica.github.io/PDC_GeophysicsBook/BEDMAP/Create_ShapeLines_for_BEDMAP3.html
'''

# todo: reorder_index_optimized would need to be optimized. The result applied to CRESIS_2009_Thwaites_AIR_BM3.csv
#  is not really great.

def extract_time_coverage(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        header_lines = [next(f).strip() for _ in range(18)]
    metadata = {line.split(':')[0].lstrip('#'): line.split(':')[1].strip() for line in header_lines if ':' in line}
    start_year = int(metadata.get('time_coverage_start', 0))
    end_year = int(metadata.get('time_coverage_end', 0))
    # Assertions to verify valid 4-digit years
    assert 1000 <= start_year <= 9999, f"Invalid start_year: {start_year}"
    assert 1000 <= end_year <= 9999, f"Invalid end_year: {end_year}"
    return start_year, end_year

def remove_close_points_by_eps(df, check_vars=(None, None), rel_std_thresh=0.10, epsilon=None):
    """
    Remove duplicates based on proximity in specified coordinates (lat/lon or east/north),
    and relative std of 'ice_thickness'.

    - If relative std < rel_std_thresh: average ice_thickness, keep one row.
    - If relative std >= rel_std_thresh: drop the whole group.

    Parameters:
        df : pd.DataFrame
            Must contain 'ice_thickness' and the check_vars columns.
        check_vars : tuple of str
            Column names to define duplicate proximity, e.g., ('lat', 'lon') or ('east', 'north').
        rel_std_thresh : float
            Threshold for relative standard deviation to consider values consistent.
        epsilon : float
            Maximum spatial distance to consider points as duplicates (same units as coords).

    Returns:
        pd.DataFrame: cleaned DataFrame with duplicates removed.
    """

    required_cols = set(check_vars) | {'ice_thickness'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    coords = df[list(check_vars)].values
    tree = cKDTree(coords)

    n = len(df)
    labels = -np.ones(n, dtype=int)
    cluster_id = 0

    # Assign cluster IDs to nearby points
    for i in tqdm(range(n), total=n):
        if labels[i] != -1:
            continue  # already assigned
        idxs = tree.query_ball_point(coords[i], r=epsilon)
        for idx in idxs:
            labels[idx] = cluster_id
        cluster_id += 1

    df = df.copy()
    df['cluster_id'] = labels
    indexes_to_drop = []
    updated_thickness = {}

    unique_clusters = df['cluster_id'].unique()
    print(f"Clusters to process: {len(unique_clusters)}")

    for cluster_id in tqdm(unique_clusters, desc="Processing close duplicates"):
        group = df[df['cluster_id'] == cluster_id]

        if len(group) == 1:
            continue  # not a duplicate group

        mean = group['ice_thickness'].mean()
        std = group['ice_thickness'].std()
        rel_std = std / mean if mean != 0 else np.inf

        if rel_std <= rel_std_thresh:
            to_drop = group.index[1:]
            indexes_to_drop.extend(to_drop)
            updated_thickness[group.index[0]] = mean
        else:
            indexes_to_drop.extend(group.index)  # Drop all

    cleaned_df = df.drop(index=indexes_to_drop)

    for idx, val in updated_thickness.items():
        cleaned_df.at[idx, 'ice_thickness'] = val

    cleaned_df = cleaned_df.drop(columns='cluster_id')
    cleaned_df = cleaned_df.sort_index().reset_index(drop=True)
    return cleaned_df

def remove_duplicates_by_columns(df, check_vars=(None,None), rel_std_thresh=0.10):
    """
    Remove duplicates based on a set of columns and relative std of 'ice_thickness'.

    - If relative std < rel_std_thresh: average ice_thickness, keep one row.
    - If relative std >= rel_std_thresh: drop the whole group.

    Parameters:
        df : pd.DataFrame
            Input DataFrame containing 'ice_thickness' and check_vars.
        check_vars : tuple of str
            Columns to use for detecting duplicates (e.g., ('lat', 'lon') or ('east', 'north')).
        rel_std_thresh : float
            Relative std threshold to decide if duplicates should be averaged or dropped.

    Returns:
        pd.DataFrame:
            DataFrame with duplicates resolved.
    """
    required_cols = set(check_vars) | {'ice_thickness'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    # Identify duplicated lat-lon or east-north rows
    dup_mask = df.duplicated(subset=check_vars, keep=False)
    dup_df = df[dup_mask]
    non_dup_df = df[~dup_mask]

    #print(f"\nChecking duplicates by {check_vars}")
    #print(f"Total dataset: {df.shape}")
    #print(f"Duplicates: {len(dup_df)}")
    #print(f"Non-duplicates: {len(non_dup_df)}")

    # Group duplicates by ('lat', 'lon') or ('east', 'north')
    grouped = dup_df.groupby(list(check_vars), sort=False)
    #print(f"Groups: {len(grouped)}")

    indexes_to_drop = []
    updated_thickness = {}

    for (lat, lon), group in tqdm(grouped, total=len(grouped), desc=f"Processing duplicates {check_vars}"):
        mean = group['ice_thickness'].mean()
        std = group['ice_thickness'].std()
        rel_std = std / mean if mean != 0 else np.inf

        if rel_std <= rel_std_thresh:
            # Drop all but first index, and update first with mean thickness
            to_drop = group.index[1:]
            indexes_to_drop.extend(to_drop)
            updated_thickness[group.index[0]] = mean
        else:
            # Drop all rows in this group
            indexes_to_drop.extend(group.index)

    # Drop unwanted rows
    cleaned_df = df.drop(index=indexes_to_drop)

    # Update the preserved rows with averaged thickness
    for idx, val in updated_thickness.items():
        cleaned_df.at[idx, 'ice_thickness'] = val

    cleaned_df = cleaned_df.sort_index().reset_index(drop=True)
    #print(f"Cleaned dataset size by {check_vars}: {len(cleaned_df)}")
    return cleaned_df

def reorder_index_by_precomputed_neighbors(df, neighbors_k=1000, start_idx=0):
    """
    Reorder points based on spatial proximity using a precomputed KDTree.

    Parameters:
        df (pd.DataFrame): DataFrame with 'east' and 'north' columns.
        neighbors_k (int): Number of neighbors to precompute for each point.
        start_idx (int): Index to start the traversal.

    Returns:
        pd.DataFrame: Reordered DataFrame with reset index.
    """
    coords = np.column_stack((df['east'].values, df['north'].values))
    n_points = len(df)

    # Build KDTree and precompute neighbors
    tree = cKDTree(coords)
    # This is a precomputation step. For every single point in your dataset, it finds the nearest 1,000 neighbors immediately.
    dists, indices = tree.query(coords, k=neighbors_k + 1)
    # We skip the first column because the closest neighbor to any point is always itself (distance = 0).
    neighbors = indices[:, 1:]  # skip self (first column)

    visited = np.zeros(n_points, dtype=bool)
    path = []
    current_idx = start_idx
    visited[current_idx] = True
    path.append(current_idx)

    unvisited_indices = set(range(n_points))
    unvisited_indices.remove(current_idx)

    pbar = tqdm(total=n_points - 1, desc="Reordering points")

    while len(path) < n_points:
        # Step 1: check precomputed neighbors
        neighs = neighbors[current_idx]
        next_idx = None
        for n_idx in neighs:
            if not visited[n_idx]:
                next_idx = n_idx
                break

        # Step 2: Global Fallback (The "Gap" Handler) if needed
        if next_idx is None:
            if not unvisited_indices:
                break  # All points visited

            unvisited_list = np.array(list(unvisited_indices))
            k_query = len(unvisited_list)
            dist, nearest_idx = tree.query(coords[current_idx], k=k_query)

            # Ensure nearest_idx is iterable
            nearest_idx = np.atleast_1d(nearest_idx)

            for candidate in nearest_idx:
                if not visited[candidate]:
                    next_idx = int(candidate)
                    break

            # Final guaranteed fallback: take any unvisited point
            if next_idx is None:
                next_idx = int(unvisited_indices.pop())

        # Update tracking
        visited[next_idx] = True
        unvisited_indices.discard(next_idx)  # `discard` avoids KeyError if already removed
        path.append(next_idx)
        current_idx = next_idx
        pbar.update(1)

    pbar.close()

    assert len(path) == n_points, f"Only {len(path)} of {n_points} points visited"

    return df.iloc[path].reset_index(drop=True)

def reorder_index_optimized(df, step_size=50000):
    coords = np.column_stack((df['east'].values, df['north'].values))
    n_points = len(df)

    visited = np.zeros(n_points, dtype=bool)
    path = []
    current_idx = 0  # Starting point

    path.append(current_idx)
    visited[current_idx] = True

    # We build the tree once
    tree = cKDTree(coords)

    pbar = tqdm(total=n_points, desc="Reordering points")

    while len(path) < n_points:
        # Step 1: Query the 'next few' closest points
        # We query 20 instead of 1000 to keep it fast per iteration
        dists, indices = tree.query(coords[current_idx], k=20)

        next_idx = None
        for idx in indices:
            if not visited[idx]:
                next_idx = idx
                break

        # Step 2: If local search fails, find the nearest UNVISITED point
        if next_idx is None:
            # OPTIMIZATION: Instead of querying the whole tree,
            # find the nearest point from the remaining unvisited set.
            # To keep it fast, we only do this every time the 'chain' breaks.
            unvisited_list = np.where(~visited)[0]
            if len(unvisited_list) == 0: break

            # Use a smaller 'local' tree of unvisited points to find the next jump
            # We only use a subset if the list is huge to save time
            search_indices = unvisited_list[::max(1, len(unvisited_list) // step_size)]
            temp_tree = cKDTree(coords[search_indices])
            _, nearest_temp_idx = temp_tree.query(coords[current_idx], k=1)
            next_idx = search_indices[nearest_temp_idx]

        path.append(next_idx)
        visited[next_idx] = True
        current_idx = next_idx
        pbar.update(1)

    pbar.close()
    return df.iloc[path].reset_index(drop=True)

#bedmap_folder = f"/media/maffe/nvme/polar_ice_thickness_data/bedmap"
bedmap_folder = f"/media/maffe/sturellone/bedmap_individual_csv_files/bedmap"

# Find all .csv files in the directory
files_bedmap = [
    os.path.join(root, file)
    for root, _, files in os.walk(bedmap_folder)
    for file in files if file.endswith(".csv")
]

# Process each file
total_files = len(files_bedmap)
print(f"We are dealing with {total_files} files from BedMap")

expected_columns = {
    'trajectory_id': 'flight_id',               # Line of flight ID
    'trace_number': 'point_id',                 # Trace number from the specific line given in Line_ID (note, -9999 often)
    'longitude (degree_east)': 'lon',
    'latitude (degree_north)': 'lat',
    'date': 'date',                                 # YYYY-MM-DD
    'time_UTC': 'time_utc',                         # HH:MM:SS
    'surface_altitude (m)': 'surface_elev',
    'land_ice_thickness (m)': 'ice_thickness',
    'bedrock_altitude (m)': 'bed_evel',
    'two_way_travel_time (m)': 'twtt',              # Two-way travel time in seconds
    'aircraft_altitude (m)': 'aircraft_elev',
    'along_track_distance (m)': 'atd'               # Distance in the along-track direction
}


# List to store dataframes
bedmap_list_dataframes = []

# Files not to be imported because they have equivalent BedMap3 entries
bedmap_files_not_to_import = ['BAS_2010_IMAFI_AIR_BM2.csv', 'INGV_1997_ITASE_AIR_BM2.csv',
                              'NIPR_1999_JARE40_GRN_BM2.csv', 'NIPR_2007_JARE49_GRN_BM2.csv',
                              'NIPR_2007_JASE_GRN_BM2.csv', 'RNRF_2003_48RAEap5_AIR_BM2.csv',
                              'RNRF_2004_49RAEap5_AIR_BM2.csv', 'RNRF_2005_50RAEap5_AIR_BM2.csv',
                              'RNRF_2006_51RAEap5_AIR_BM2.csv', 'RNRF_2006_KV1-area_AIR_BM2.csv',
                              'RNRF_2007_52RAEap5_AIR_BM2.csv', 'RNRF_2007_Mirny-Vostok_AIR_BM2.csv',
                              'RNRF_2008_53RAEap5_AIR_BM2.csv', 'UTIG_2008_ICECAP_AIR_BM2.csv',
                              ]

for i, file in tqdm(enumerate(files_bedmap), total=total_files, leave=True):

    #file = '/media/maffe/nvme/polar_ice_thickness_data/bedmap/bedmap/bedmap3/BAS_2010_IMAFI_AIR_BM3.csv'
    #file = '/media/maffe/nvme/polar_ice_thickness_data/bedmap/bedmap/bedmap3/CRESIS_2009_Thwaites_AIR_BM3.csv'
    #file = '/media/maffe/nvme/polar_ice_thickness_data/bedmap/bedmap/bedmap2/BEDMAP2_-_Ice_thickness__bed_and_surface_elevation_for_Antarctica_-_standardised_data_points/BEDMAP2 - Ice thickness, bed and surface elevation for Antarctica - standardised data points/BGR_1999_GANOVEX-VIII-Mertz_AIR_BM2.csv'
    #file = '/media/maffe/nvme/polar_ice_thickness_data/bedmap/bedmap/bedmap2/BEDMAP2_-_Ice_thickness__bed_and_surface_elevation_for_Antarctica_-_standardised_data_points/BEDMAP2 - Ice thickness, bed and surface elevation for Antarctica - standardised data points/RNRF_2008_Vostok-Subglacial-Lake_AIR_BM2.csv'
    #if i==1: break

    filename = file.rsplit('/', 1)[-1]
    #print(filename)

    if filename in bedmap_files_not_to_import:
        print(f"Not using {filename}")
        continue

    # read the header start and end years (int)
    yyyy_start, yyyy_end = extract_time_coverage(file)

    # Read the CSV file, skipping the header
    df_file_csv = pd.read_csv(file, skiprows=18, low_memory=False)

    # Check if expected columns exist
    missing_columns = [col for col in expected_columns.keys() if col not in df_file_csv.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # Rename columns
    df_file_csv = df_file_csv[expected_columns.keys()].rename(columns=expected_columns)

    # Add some columns: csv filename, filename number (from 0 to 151), years of start and end coverage
    df_file_csv['file'] = filename
    df_file_csv['file_no'] = i
    df_file_csv['year_start'] = yyyy_start
    df_file_csv['year_end'] = yyyy_end


    def clean_date_column(col, year=None):
        """
        Clean date column by replacing sentinel values (-9999 etc.) with a fallback YYYY-01-01.
        """
        s = col.astype(str).str.strip().replace({'-9999': pd.NA, 'nan': pd.NA, 'NaN': pd.NA})

        def fix_date(x):
            if pd.isna(x):
                return f"{year}-01-01"
            elif re.fullmatch(r'\d{4}-\d{2}-\d{2}', x):
                return x
            elif re.fullmatch(r'\d{4}', x):
                return f"{x}-01-01"
            else:
                return f"{year}-01-01"

        return s.apply(fix_date)


    def clean_time_utc_column(col):
        """
        Clean time_utc column by replacing sentinel values (-9999 etc.) with fallback 00:00:00
        and by removing milliseconds (if present)
        """
        s = col.astype(str).str.strip()
        s = s.replace({'-9999': pd.NA, 'nan': pd.NA, 'NaN': pd.NA})

        # Fill missing times with fallback
        s = s.fillna("00:00:00")

        # Remove fractional seconds (keep only HH:MM:SS)
        s = s.str.extract(r'(\d{2}:\d{2}:\d{2})')[0].fillna("00:00:00")

        return s


    # Let's deal with date and time columns. We create two temporary columns
    df_file_csv['date_str'] = clean_date_column(df_file_csv['date'], year=yyyy_start)
    df_file_csv['time_str'] = clean_time_utc_column(df_file_csv['time_utc'])
    combined = df_file_csv['date_str'] + ' ' + df_file_csv['time_str']

    # Now we generate datetime and timestamp columns
    # datetime will be type datetime64[ns], with NaT for missing/invalid entries.
    df_file_csv['datetime'] = pd.to_datetime(combined, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    # timestamp will be number of seconds since 1970-01-01 00:00:00 UTC , with np.nan for missing/invalid entries
    df_file_csv['timestamp'] = df_file_csv['datetime'].apply(
        lambda x: int(x.timestamp()) if pd.notnull(x) else np.nan
    )

    # We don't need these two columns anymore
    df_file_csv.drop(columns=['date_str', 'time_str'], inplace=True)

    # Convert to dataframe because we need east and north to run reindexing
    gdf_file_csv = gpd.GeoDataFrame(
        df_file_csv,
        geometry=gpd.points_from_xy(df_file_csv['lon'], df_file_csv['lat']), crs="EPSG:4326").to_crs("EPSG:3031")

    # Add columns
    df_file_csv['east'] = gdf_file_csv.geometry.x.values
    df_file_csv['north'] = gdf_file_csv.geometry.y.values


    # Debug plot
    debug_plot = False
    if debug_plot:
        df_file_csv = df_file_csv.loc[df_file_csv['ice_thickness'] > 0]
        #mask = (df_file_csv['lon'] >= -108.5) & (df_file_csv['lon'] <= -108.46) & \
        #       (df_file_csv['lat'] >= -77.233) & (df_file_csv['lat'] <= -77.22)
        #df_subset = df_file_csv[mask]
        #print(len(df_file_csv))
        fig, ax = plt.subplots()
        s = ax.scatter(x=df_file_csv['east'], y=df_file_csv['north'],
                   c=df_file_csv.index, cmap='jet', s=2) # df_file_csv['land_ice_thickness (m)']

        for idx, row in df_file_csv.iterrows():
            continue
            ax.annotate(str(idx),
                        (row['east'], row['north']),
                        textcoords="offset points",  # how to position the text
                        xytext=(5, 5),  # distance from text to points (x,y)
                        ha='left',  # horizontal alignment
                        fontsize=8)
        cb = plt.colorbar(s)
        plt.show()

    files_to_reindex = ['BGR_1999_GANOVEX-VIII-Mertz_AIR_BM2.csv', 'RNRF_2008_Vostok-Subglacial-Lake_AIR_BM2.csv',
                        'BGR_1999_GANOVEX-VIII-Matusevich_AIR_BM2.csv', 'NIPR_2018_JARE60_GRN_BM3.csv',
                        'NIPR_2007_JARE49_GRN_BM3.csv', 'NIPR_2017_JARE59_GRN_BM3.csv', 'BEDMAP1_1966-2000_AIR_BM1.csv',
                        'CRESIS_2009_Thwaites_AIR_BM3.csv']

    # Reindex some files
    if filename in files_to_reindex:
        print(f'Reindexing {filename}')
        #df_file_csv = reorder_index_by_precomputed_neighbors(df_file_csv)
        df_file_csv = reorder_index_optimized(df_file_csv)


    ifplot = False
    if ifplot:
        fig, ax = plt.subplots()
        df_file_csv = df_file_csv.loc[df_file_csv['ice_thickness'] > 0]
        print(len(df_file_csv))
        s = ax.scatter(x=df_file_csv['east'], y=df_file_csv['north'],
                       c=df_file_csv.index, cmap='jet', s=2)
        cb = plt.colorbar(s)
        plt.show()

    # Remove columns
    df_file_csv.drop(columns=['east', 'north'], inplace=True)

    # Append to the list of dataframes
    bedmap_list_dataframes.append(df_file_csv)

# Concatenate all dataframes into one
bedmap = pd.concat(bedmap_list_dataframes, axis=0, ignore_index=True)
print(f"Original BedMap no. points: {len(bedmap)}.")

# Drop some columns I don't need
cols_to_delete = ['twtt',  'aircraft_elev', 'surface_elev']
bedmap.drop(columns=cols_to_delete, inplace=True)
print(f"Dropped cols: {cols_to_delete}")

# Set string dtype for some columns
bedmap['flight_id'] = bedmap['flight_id'].astype('string')
bedmap['file'] = bedmap['file'].astype('string')
bedmap['date'] = bedmap['date'].astype('string')
bedmap['time_utc'] = bedmap['time_utc'].astype('string')

# ---------- PRELIMINARY CLEANING ----------------
# Directly remove the points with ice_thickness == -9999 or == 0
bedmap = bedmap[bedmap['ice_thickness'] > 0]
print(f"Dataset after removing bad ice thickness values: {len(bedmap)}")

# Remove duplicates (lat,lon) and (east, north)
bedmap = remove_duplicates_by_columns(bedmap, check_vars=('lat', 'lon'))
print(f"Dataset after duplicates lat-lon: {len(bedmap)} points")
assert not bedmap.duplicated(subset=['lat', 'lon']).any(), "There are duplicates in final bedmap."

#print(bedmap_gdf.dtypes)
#print(bedmap_gdf.sample(5))
#print(bedmap_gdf.crs)
#print(bedmap_gdf.describe())

# -----------------------------------------------
# file (string): csv filenames n=151 from BedMap
# file_no (int): 0-150
# year_start (int): 1966-2019
# year_end (int): 1972-2020
# timestamp (float64): 9.168300e+08 - 1.577656e+09. Outliers NaN.

# flight_id (string): this identified could be an int or a string.
# point_id (int): -9999 to 120000. Outliers -9999.
# Often flight_id for each file is an integer increasing from 1 onwards, and point_id is -9999.
# That makes both useless. But other times flight_id is different string identifiers (e.g. F1, F10),
# and point_id is still -9999. In such cases, flight_id can be used to identify different tracks.
# ---

bedmap_gdf = gpd.GeoDataFrame(
    bedmap,
    geometry=gpd.points_from_xy(bedmap['lon'], bedmap['lat']), crs="EPSG:4326").to_crs("EPSG:3031")
bedmap_gdf['east'] = bedmap_gdf.geometry.x.values
bedmap_gdf['north'] = bedmap_gdf.geometry.y.values
print('Generated geodataframe.')

del bedmap
print('Deleted bedmap to free some memory.')

# Check no invalid points
assert (~bedmap_gdf.geometry.is_valid).sum()==0, f"Invalid geometries found !"

bedmap_gdf = remove_duplicates_by_columns(bedmap_gdf, check_vars=('east', 'north'))
print(f"Dataset after duplicates east-north: {len(bedmap_gdf)} points")
assert not bedmap_gdf.duplicated(subset=['east', 'north']).any(), "There are duplicates in final bedmap."

#print('Begin the attempt to remove closeby points')
#bedmap_gdf = remove_close_points_by_eps(bedmap_gdf, check_vars=('east', 'north'), epsilon=1.)
#print(f"Dataset after removing closeby points east-north: {len(bedmap_gdf)} points")

# Convert to a pandas DataFrame and drop the 'geometry' column
bedmap = pd.DataFrame(bedmap_gdf.drop(columns='geometry'))

#print(f"List of columns: {bedmap.info}")

# --------- SAVE GDF ---------
save = False
if save:
    bedmap.to_parquet("/media/maffe/sturellone/gprclean/bedmap_raw_data.parquet", index=False)
    print('SAVED.')

print('ADDIO.')