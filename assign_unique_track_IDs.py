import argparse, time, os
import random
from sklearn.cluster import DBSCAN
from localtileserver.tiler import get_point
from tqdm import tqdm
import copy, math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from pyproj import Proj, Transformer, Geod
import hvplot.pandas
import panel as pn

pd.set_option('display.max_columns', None)

# 'along_track_distance (m)': 'atd' ANY USE ?

def assign_timestamp_id_segments(df, threshold_time_jump=10000):
    """
    Assigns segment IDs based on jumps in timestamp (in seconds).

    Parameters:
    - df: pandas DataFrame with a 'timestamp' column.
    - threshold_time_jump: threshold in seconds for identifying breaks. Default is 600 seconds.

    Returns:
    - A Series of segment IDs, same length as df.
      Segment IDs are 0, 1, 2... or NaN if segmentation is not useful.
    """
    if 'timestamp' not in df.columns:
        return pd.Series(np.nan, index=df.index)

    ts = df['timestamp']

    if ts.isna().any():
        return pd.Series(np.nan, index=df.index)

    diffs = ts.diff().fillna(0)

    # Identify breaks
    is_jump = (diffs < 0) | (diffs > threshold_time_jump)

    # Compute cumulative sum of jumps to assign segment IDs (starts from 0)
    segment_ids = is_jump.cumsum().astype(float)

    # If no jumps found, return NaN
    if segment_ids.nunique() == 1:
        return pd.Series(np.nan, index=df.index)

    return pd.Series(segment_ids.values, index=df.index)

def assign_point_id_segments(df, threshold_jump=500):
    """
    Returns a pd.Series of segment labels based on 'point_id' in the given df slice.

    Parameters:
        df: pandas DataFrame slice (e.g. bedmap_file)
        threshold_jump: threshold for detecting breaks in 'point_id'

    Returns:
        pd.Series with same index as df, integer segment IDs (0,1,...)
        or all zeros if no segmentation found.
    """
    if 'point_id' not in df.columns:
        return pd.Series(np.nan, index=df.index)

    point_ids = df['point_id'].where(df['point_id'] != -9999, np.nan)

    if point_ids.dropna().empty:
        return pd.Series(np.nan, index=df.index)

    diffs = point_ids.diff().fillna(0)

    # A potential new track is created if point_id has decreased or has increased by a large margin
    is_reset = (diffs < 0) | (diffs > threshold_jump)
    segment_starts = is_reset[is_reset].index.tolist()

    if len(segment_starts) == 0:
        return pd.Series(np.nan, index=df.index)

    segment_starts = [df.index[0]] + segment_starts

    segment_labels = pd.Series(np.nan, index=df.index)

    for seg_id, start_idx in enumerate(segment_starts):
        end_idx = segment_starts[seg_id + 1] if seg_id + 1 < len(segment_starts) else df.index[-1] + 1
        idx_range = df.loc[start_idx:end_idx - 1].index
        segment_labels.loc[idx_range] = seg_id

    return segment_labels

def assign_flight_id_segments(df, max_unique_fraction=0.1):
    """
    Assign integer track IDs based on 'flight_id' values if useful.

    Heuristics:
    - 'flight_id' column must exist
    - Not too many unique flight_ids (less than max_unique_fraction of total rows)
    - Not mostly invalid (-9999) values

    Parameters:
        df: pandas DataFrame slice
        max_unique_fraction: max fraction of unique flight_ids allowed

    Returns:
        pd.Series of integers (track IDs), same index as df.
        All zeros if 'flight_id' not useful.
    """
    if 'flight_id' not in df.columns:
        return pd.Series(np.nan, index=df.index)

    flight_ids = df['flight_id'].str.strip()

    if flight_ids.empty:
        return pd.Series(np.nan, index=df.index)

    # Check for invalid marker '-9999' fraction
    if (flight_ids == '-9999').mean() > 0.8:
        return pd.Series(np.nan, index=df.index)

    n_total = len(flight_ids)
    n_unique = flight_ids.nunique()

    if n_unique <= 1 or n_unique/n_total>max_unique_fraction:
        return pd.Series(np.nan, index=df.index)

    # Map each unique flight_id to an integer ID
    unique_ids = flight_ids.unique()
    id_map = {fid: i for i, fid in enumerate(unique_ids)}
    track_ids = flight_ids.map(id_map).astype("float")

    return track_ids

def assign_spatial_id_segments(df, threshold_dist=5000):
    """
    Assigns track IDs based on spatial jumps in sequential data.

    Parameters:
    - df: pandas DataFrame with columns:
        - 'east', 'north'
    - threshold_dist: distance (in meters) to detect jumps (default: 5000m)

    Returns:
    - A Series of track IDs (float), same index as df.
      Track IDs start at 0. If no segmentation is useful, returns all NaN.
    """
    if 'east' not in df.columns or 'north' not in df.columns:
        return pd.Series(np.nan, index=df.index)

    # Work only on untracked points
    x = df['east'].values
    y = df['north'].values

    if len(x) < 2:
        return pd.Series(np.nan, index=df.index)

    # Compute spatial differences
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    dists = np.sqrt(dx ** 2 + dy ** 2)

    #print(f"Mean distance between points: {dists.mean():.1f} meters")

    # Jumps define new tracks
    is_jump = dists > threshold_dist
    segment_ids = np.cumsum(is_jump).astype(float)

    return pd.Series(segment_ids, index=df.index)

def assign_angular_id_segments(df, threshold_angle_deg=60, step=7):
    """
    Assigns track IDs based on angular changes between consecutive points.

    Parameters:
    - df: DataFrame with 'east' and 'north' columns
    - threshold_angle_deg: angle threshold (in degrees) to detect jumps

    Returns:
    - A Series of track IDs (float), same index as df
      Track IDs start at 0. If no segmentation is useful, returns all NaN.
    """
    if 'east' not in df.columns or 'north' not in df.columns:
        return pd.Series(np.nan, index=df.index)

    x = df['east'].values
    y = df['north'].values
    n = len(df)

    if n < 2 * step + 1:
        return pd.Series(np.nan, index=df.index)  # Not enough points for angular analysis

    # Define start and end points for direction vectors
    v1_start = np.column_stack([x[:-2 * step], y[:-2 * step]])
    v1_end = np.column_stack([x[step:-step], y[step:-step]])
    v2_end = np.column_stack([x[2 * step:], y[2 * step:]])

    # Direction vectors
    v1 = v1_end - v1_start
    v2 = v2_end - v1_end

    # Compute angles
    dot = np.einsum('ij,ij->i', v1, v2)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    cos_angle = np.clip(dot / (norm1 * norm2 + 1e-12), -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(cos_angle))

    # Detect angular jumps
    is_jump = np.zeros(n, dtype=bool)
    is_jump[step:-step] = angles_deg > threshold_angle_deg

    # Compute segment IDs
    segment_ids = np.cumsum(is_jump).astype(float)

    if segment_ids.max() == 0:
        return pd.Series(np.nan, index=df.index)

    return pd.Series(segment_ids, index=df.index)


def compute_segmentation_index(df):
    """
    Returns the fraction of points that were assigned to at least one track
    by any of the five methods. Unassigned points are marked with NaN.
    """
    tracked_mask = (
        df['track_flight_id'].notna() |
        df['track_point_id'].notna() |
        df['track_timestamp_id'].notna() |
        df['track_spatial_id'].notna() |
        df['track_angular_id'].notna()
    )
    n_tracked = tracked_mask.sum()
    total = len(df)
    return n_tracked / total


# The goal is to assign to each point a Track ID
t0 = time.time()
bedmap = pd.read_parquet('/media/maffe/sturellone/GPRCleanup/bedmap_raw_data.parquet', engine='fastparquet')
print(f'Parquet loaded in {time.time()-t0:.1f}')

file_nos = bedmap['file_no'].unique().tolist()
file_names = bedmap['file'].unique().tolist()

bedmap['track_id'] = -9999
bedmap['track_method'] = 'none'

# Initialize id counter
running_id_track_counter = 0

# Loop over the 151 files
for n, file_name in enumerate(file_names):

    #if n == 10:
    #    break

    #file_name = 'BAS_2009_FERRIGNO_GRN_BM2.csv'
    #file_name='BAS_2001_MAMOG_AIR_BM2.csv'
    #file_name='BAS_2011_Adelaide_AIR_BM3.csv'
    #file_name = 'CRESIS_2009_Thwaites_AIR_BM3.csv' # many tracks for time
    #file_name = 'RNRF_1971_Lambert-Amery_SEI_BM3.csv'
    #file_name = 'BAS_2017_English-Coast_AIR_BM3.csv'
    #file_name = 'BAS_2010_PIG_AIR_BM2.csv'
    #file_name = 'NIPR_2017_JARE59_GRN_BM3.csv' # The index progression does not make sense
    #file_name = 'BAS_2001_Bailey-Slessor_AIR_BM2.csv'
    #file_name='STANFORD_1971_SPRI-NSF-TUD_AIR_BM3.csv' # zig-zag makes angular bananas
    #file_name = 'NIPR_1999_JARE40_GRN_BM3.csv'

    bedmap_file = bedmap.loc[bedmap['file'] == file_name].copy()

    #print(f"{bedmap_file['file_no'].iloc[0]} {file_name} n={len(bedmap_file)} t={bedmap_file['year_start'].iloc[0]}")

    flight_ids = assign_flight_id_segments(bedmap_file)
    point_ids = assign_point_id_segments(bedmap_file)
    timestamp_ids = assign_timestamp_id_segments(bedmap_file)
    spatial_ids = assign_spatial_id_segments(bedmap_file)
    angular_ids = assign_angular_id_segments(bedmap_file)

    bedmap_file['track_flight_id'] = flight_ids
    bedmap_file['track_point_id'] = point_ids
    bedmap_file['track_timestamp_id'] = timestamp_ids
    bedmap_file['track_spatial_id'] = spatial_ids
    bedmap_file['track_angular_id'] = angular_ids
    bedmap_file['track_id'] = -1 # placeholder.

    n_tracks_flight = flight_ids.nunique()
    n_tracks_point = point_ids.nunique()
    n_tracks_timestamp = timestamp_ids.nunique()
    n_tracks_spatial = spatial_ids.nunique()
    n_tracks_angular = angular_ids.nunique()

    if n_tracks_point > 0:
        chosen_method = "point"
        bedmap_file['track_id'] = point_ids.astype(int) + running_id_track_counter
        running_id_track_counter += int(point_ids.max()) + 1
    elif n_tracks_flight > 0:
        chosen_method = "flight"
        bedmap_file['track_id'] = flight_ids.astype(int) + running_id_track_counter
        running_id_track_counter += int(flight_ids.max()) + 1
    elif n_tracks_timestamp > 0:
        chosen_method = "timestamp"
        bedmap_file['track_id'] = timestamp_ids.astype(int) + running_id_track_counter
        running_id_track_counter += int(timestamp_ids.max()) + 1
    elif n_tracks_spatial > 0:
        chosen_method = "spatial"
        bedmap_file['track_id'] = spatial_ids.astype(int) + running_id_track_counter
        running_id_track_counter += int(spatial_ids.max()) + 1
    elif n_tracks_angular > 0:
        chosen_method = "angular"
        bedmap_file['track_id'] = angular_ids.astype(int) + running_id_track_counter
        running_id_track_counter += int(angular_ids.max()) + 1
    else:
        raise ValueError("No method triggered.")


    print(f"{bedmap_file['file_no'].iloc[0]} {file_name} n={len(bedmap_file)} - flight_id | point_id | timestamp | spatial | angular: {n_tracks_flight} | "
          f"{n_tracks_point} | {n_tracks_timestamp} | {n_tracks_spatial} | {n_tracks_angular} | tracks")
    seg_index = compute_segmentation_index(bedmap_file)
    #print(f"Segmentation index (fraction of tracked points): {seg_index:.4f}")

    plot = False
    if plot:
        #bedmap_file = bedmap_file.sample(int(1e5))
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3,3, figsize=(13,13))
        s1 = ax1.scatter(bedmap_file['east'], bedmap_file['north'], c=bedmap_file['track_flight_id'], cmap='nipy_spectral', s=1)
        s2 = ax2.scatter(bedmap_file['east'], bedmap_file['north'], c=bedmap_file['track_point_id'], cmap='nipy_spectral', s=1)
        s3 = ax3.scatter(bedmap_file['east'], bedmap_file['north'], c=bedmap_file['track_timestamp_id'], cmap='nipy_spectral', s=1)
        s4 = ax4.scatter(bedmap_file['east'], bedmap_file['north'], c=bedmap_file['track_spatial_id'], cmap='nipy_spectral', s=1)
        s5 = ax5.scatter(bedmap_file['east'], bedmap_file['north'], c=bedmap_file['track_angular_id'], cmap='nipy_spectral', s=1)
        s6 = ax6.scatter(bedmap_file['east'], bedmap_file['north'], c=bedmap_file['ice_thickness'], cmap='jet', s=1)
        s7 = ax7.scatter(bedmap_file['east'], bedmap_file['north'], c=bedmap_file.index, cmap='jet', s=1)
        cb1 = plt.colorbar(s1)
        cb2 = plt.colorbar(s2)
        cb3 = plt.colorbar(s3)
        cb4 = plt.colorbar(s4)
        cb5 = plt.colorbar(s5)
        cb6 = plt.colorbar(s6)
        cb7 = plt.colorbar(s7)
        ax1.set_title(f'n={len(bedmap_file)} flight n={n_tracks_flight}')
        ax2.set_title(f'point n={n_tracks_point}')
        ax3.set_title(f'timestamp n={n_tracks_timestamp}')
        ax4.set_title(f'spatial n={n_tracks_spatial}')
        ax5.set_title(f'angular n={n_tracks_angular}')
        ax6.set_title(f'ice thickness')
        ax7.set_title(f'index')
        ax8.set_axis_off()
        ax9.set_axis_off()
        plt.tight_layout()
        plt.show()


    # Update original bedmap dataframe
    bedmap.loc[bedmap_file.index, 'track_id'] = bedmap_file['track_id'].astype(int)
    bedmap.loc[bedmap_file.index, 'track_method'] = chosen_method

    #print(flight_id_offset, point_id_offset, timestamp_id_offset, spatial_id_offset, angular_id_offset)
    print(f"Method: {chosen_method}, Running track count: {running_id_track_counter}")


print('Now lets generate unique track identifiers ... ')
print(f"Data shape: {bedmap.shape}")
print(f"Data columns: {list(bedmap)}")
print(f"{bedmap['track_method'].value_counts()}")
print(f"Track min {bedmap['track_id'].min()} Track max {bedmap['track_id'].max()}")

# --------- SAVE ---------
save = True
if save:
    bedmap.to_parquet("/media/maffe/sturellone/GPRCleanup/bedmap_track_ids.parquet", index=False)
    print('SAVED.')

# --------- PLOT ---------
pn.extension()

scatter = bedmap.hvplot.scatter(
    x='east',
    y='north',
    c='track_id',
    cmap='jet',
    rasterize=True,
    width=600,
    height=600,
    title='Tracks by ID (east vs north)',
    colorbar=True,
    tools=['hover'],
    hover_cols=['track_id']
)
# Launch interactive plot in browser
pn.panel(scatter).show()



print('ADDIO.')