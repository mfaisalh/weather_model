import numpy as np

from netCDF4 import Dataset



# Parameters

num_latitudes = 20

num_longitudes = 40

time_steps = 50



# Generate random weather data

temperature = np.random.uniform(20, 35, size=(time_steps, num_latitudes, num_longitudes))

humidity = np.random.uniform(50, 90, size=(time_steps, num_latitudes, num_longitudes))

wind_speed = np.random.uniform(5, 25, size=(time_steps, num_latitudes, num_longitudes))



# Save data to NetCDF file

output_file = "weather_data.nc"



try:

    dataset = Dataset(output_file, 'w', format='NETCDF4_CLASSIC')



    # Define dimensions

    time_dim = dataset.createDimension('time', None)

    lat_dim = dataset.createDimension('latitude', num_latitudes)

    lon_dim = dataset.createDimension('longitude', num_longitudes)



    # Create variables

    times = dataset.createVariable('time', np.float64, ('time',))

    latitudes = dataset.createVariable('latitude', np.float32, ('latitude',))

    longitudes = dataset.createVariable('longitude', np.float32, ('longitude',))

    temp_var = dataset.createVariable('temperature', np.float32, ('time', 'latitude', 'longitude'))

    hum_var = dataset.createVariable('humidity', np.float32, ('time', 'latitude', 'longitude'))

    wind_var = dataset.createVariable('wind_speed', np.float32, ('time', 'latitude', 'longitude'))



    # Assign data to variables

    times[:] = np.arange(time_steps)

    latitudes[:] = np.linspace(-90, 90, num_latitudes)

    longitudes[:] = np.linspace(-180, 180, num_longitudes)

    temp_var[:, :, :] = temperature

    hum_var[:, :, :] = humidity

    wind_var[:, :, :] = wind_speed



    # Close the dataset

    dataset.close()



    print(f"Generated weather data saved to {output_file}")



except Exception as e:

    print(f"Error saving weather data: {e}")