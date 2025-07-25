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
from shapely.plotting import plot_line
from shapely.geometry import LineString, Point, MultiPoint, MultiLineString, GeometryCollection
from pyproj import Proj, Transformer, Geod
import hvplot.pandas
import panel as pn
from itertools import combinations
import math

# todo: I need to produce also self-connections between tracks, not just connections between different tracks
t0 = time.time()
bedmap = pd.read_parquet('/media/maffe/sturellone/GPRCleanup/bedmap_track_ids.parquet', engine='fastparquet')
print(f'Parquet loaded in {time.time()-t0:.1f}')
print(list(bedmap))
print(bedmap.shape)

def compute_tracks(df):
    geometries = []
    track_ids = []
    track_files = []

    for track_id, group in tqdm(df.groupby('track_id')):
        assert group['file'].nunique() == 1, "Different files found"
        track_file = group['file'].head(1).item()
        coords = group[['east', 'north']].values
        if len(coords) == 1:
            geom = Point(coords[0])
        elif len(coords) > 1:
            geom = LineString(coords)
        else:
            raise ValueError("tracks with no valid geometry.")

        geometries.append(geom)
        track_ids.append(track_id)
        track_files.append(track_file)

    # Create geodataframe of tracks and singletons
    gdf_tracks = gpd.GeoDataFrame({'track_id': track_ids, 'track_file': track_files, 'geometry': geometries}, crs="EPSG:3031")

    return gdf_tracks


tracks_gdf = compute_tracks(bedmap)
# Some of those will be isolated Points, all others will be LineString
print(f"Created dataframe of tracks: {tracks_gdf.shape}")
#print(tracks_gdf.info)

#fig, ax = plt.subplots(figsize=(10, 8))
#sth = tracks_gdf.plot(ax=ax, column="track_id", cmap="jet", lw=1)
#plt.show()


#tracks_gdf = tracks_gdf.sample(5000, random_state=42)

# Create connectivity
print(f"Begin connectivity generation.")


def compute_track_intersections(tracks_gdf):
    # 18 minutes to run and produce 53.8m intersections
    lines_gdf = tracks_gdf[tracks_gdf.geometry.type == "LineString"].copy()
    lines_gdf = lines_gdf.reset_index(drop=True)

    sindex = lines_gdf.sindex

    intersections = []

    for i, row_i in tqdm(lines_gdf.iterrows(), total=len(lines_gdf)):
        geom_i = row_i.geometry
        track_id_i = row_i.track_id
        file_i = row_i.track_file

        # Spatial index: get candidate indices intersecting geom_i's bbox
        candidate_idxs = list(sindex.intersection(geom_i.bounds))

        for j in candidate_idxs:
            if j <= i:
                continue  # avoid self or duplicate (A-B == B-A)
            geom_j = lines_gdf.at[j, 'geometry']
            track_id_j = lines_gdf.at[j, 'track_id']
            file_j = lines_gdf.at[j, 'track_file']

            if geom_i.intersects(geom_j):
                intersection = geom_i.intersection(geom_j)
                if intersection.is_empty:
                    continue

                if isinstance(intersection, Point):
                    intersections.append({
                        "track_id_1": track_id_i,
                        "track_id_2": track_id_j,
                        "geometry": intersection
                    })

                elif isinstance(intersection, MultiPoint):

                    ifplot = len(intersection.geoms)>1000
                    #ifplot = False
                    #todo: this is a problem. many tracks lie on top of each other, resulting in many intersections
                    if ifplot:
                        print(file_i, file_j, track_id_i, track_id_j, len(intersection.geoms))
                        fig, ax = plt.subplots()
                        plot_line(geom_i, ax=ax, add_points=True, color='b', linewidth=1)
                        plot_line(geom_j, ax=ax, add_points=True, color='r', linewidth=1)
                        ax.set_title(f"{file_i} {file_j}")
                        plt.show()

                    for pt in intersection.geoms:
                        intersections.append({
                            "track_id_1": track_id_i,
                            "track_id_2": track_id_j,
                            "geometry": pt
                        })

                #elif isinstance(intersection, GeometryCollection):
                #    for part in intersection.geoms:
                #        if isinstance(part, Point):
                #            intersections.append({
                #                "track_id_1": track_id_i,
                #                "track_id_2": track_id_j,
                #                "geometry": part
                #            })

                elif isinstance(intersection, (LineString, MultiLineString, GeometryCollection)):
                    print(f"Skipping {intersection.geom_type} overlap between track {track_id_i} and {track_id_j}")
                    continue

                else:
                    raise ValueError(f"type not contemplated: {intersection.geom_type}")

    return gpd.GeoDataFrame(intersections, geometry="geometry", crs=tracks_gdf.crs)


intersections_gdf = compute_track_intersections(tracks_gdf)
print(intersections_gdf.geometry.geom_type.value_counts())


# --------- PLOT ---------
points = tracks_gdf[tracks_gdf.geometry.type == "Point"]
lines = tracks_gdf[tracks_gdf.geometry.type == "LineString"]
fig, ax = plt.subplots(figsize=(10, 8))
#lines.plot(ax=ax, column="track_id", cmap="viridis", lw=1)
#points.plot(ax=ax, color='b', markersize=5)
sth = tracks_gdf.plot(ax=ax, column="track_id", cmap="jet", lw=1)
intersections_gdf.plot(ax=ax, color='k', markersize=5, zorder=3)
plt.show()







