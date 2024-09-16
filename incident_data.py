import pandas as pd

# Step 1: Create a dataset of incidents with latitude, longitude, incident type, and date
data = {
    'latitude': [19.0760, 28.6139, 12.9716, 22.5726, 13.0827, 17.3850, 18.5204, 23.0225, 26.9124, 26.8467,
                 12.2958, 15.3173, 13.3400, 16.8308, 11.0168],
    'longitude': [72.8777, 77.2090, 77.5946, 88.3639, 80.2707, 78.4867, 73.8567, 72.5714, 75.7873, 80.9462,
                  76.6394, 75.7139, 77.0975, 80.2785, 76.9558],
    'incident_type': ['harassment', 'assault', 'theft', 'harassment', 'stalking', 'assault', 'theft', 'harassment', 
                      'assault', 'theft', 'harassment', 'theft', 'assault', 'stalking', 'theft'],
    'date': ['2023-09-01', '2023-09-05', '2023-09-10', '2023-09-15', '2023-09-20', '2023-09-25', '2023-09-30', 
             '2023-10-01', '2023-10-05', '2023-10-10', '2023-10-15', '2023-10-20', '2023-10-25', '2023-10-30', 
             '2023-11-01']
}

# Step 2: Create a DataFrame from the data
df = pd.DataFrame(data)

# Step 3: Display the dataset
print(df)

# Step 4: (Optional) Save the dataset as a CSV file
df.to_csv('indian_incidents_extended.csv', index=False)

print("Dataset has been saved as 'indian_incidents_extended.csv'")
