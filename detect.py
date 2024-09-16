import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.cluster import DBSCAN
import folium

# Step 1: Load the data (replace 'alerts.csv' with your actual dataset)
# Ensure the dataset has 'latitude' and 'longitude' columns for incidents
data = pd.read_csv('indian_incidents_extended.csv')  # Dataset of past incidents
# Display first few rows
print(data.head())

# Step 2: Prepare the data
# Ensure latitude and longitude are floats
data['latitude'] = data['latitude'].astype(float)
data['longitude'] = data['longitude'].astype(float)

# Step 3: Use DBSCAN for clustering to identify hotspots
# DBSCAN is a density-based clustering algorithm, perfect for geospatial data
coords = data[['latitude', 'longitude']].values

# Set DBSCAN parameters (distance in kilometers)
kms_per_radian = 6371.0088
epsilon = 1.5 / kms_per_radian  # 1.5 km radius for clustering
db = DBSCAN(eps=epsilon, min_samples=3, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))

# Append cluster labels to the original data
data['cluster'] = db.labels_

# Step 4: Visualize clusters/hotspots using Folium
# Create a base map centered around the mean location of incidents
center_lat, center_lon = data['latitude'].mean(), data['longitude'].mean()
map_hotspots = folium.Map(location=[center_lat, center_lon], zoom_start=13)

# Add clusters to the map
for _, row in data.iterrows():
    # Color code by cluster, -1 is noise (no cluster)
    color = 'red' if row['cluster'] == -1 else 'green'
    folium.CircleMarker(location=[row['latitude'], row['longitude']],
                        radius=5,
                        color=color,
                        fill=True,
                        fill_opacity=0.7).add_to(map_hotspots)

# Display the map
map_hotspots.save('hotspots_map.html')
print("Hotspot map has been created and saved as 'hotspots_map.html'")
