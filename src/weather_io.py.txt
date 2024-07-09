from netCDF4 import Dataset

import numpy as np

import logging



logger = logging.getLogger()



def read_weather_data(filename):

    """Reads weather data from a NetCDF file."""

    try:

        dataset = Dataset(filename, 'r')

        temperature = dataset.variables['temperature'][:]

        humidity = dataset.variables['humidity'][:]

        wind_speed = dataset.variables['wind_speed'][:]

        dataset.close()

        return temperature, humidity, wind_speed

    except Exception as e:

        logger.error(f"Error reading data from file {filename}: {e}")

        raise



def write_results(output_dir, temp_mean, hum_mean, wind_mean):

    """Writes processed weather data to a NetCDF file."""

    try:

        dataset = Dataset(f"{output_dir}/processed_weather_data.nc", 'w', format='NETCDF4_CLASSIC')

        time = dataset.createDimension('time', 1)

        lat = dataset.createDimension('lat', temp_mean.shape[0])

        lon = dataset.createDimension('lon', temp_mean.shape[1])

        

        times = dataset.createVariable('time', np.float64, ('time',))

        latitudes = dataset.createVariable('latitude', np.float32, ('lat',))

        longitudes = dataset.createVariable('longitude', np.float32, ('lon',))

        temp_var = dataset.createVariable('temperature_mean', np.float32, ('time', 'lat', 'lon'))

        hum_var = dataset.createVariable('humidity_mean', np.float32, ('time', 'lat', 'lon'))

        wind_var = dataset.createVariable('wind_speed_mean', np.float32, ('time', 'lat', 'lon'))

        

        times[:] = np.array([0])

        latitudes[:] = np.linspace(-90, 90, temp_mean.shape[0])

        longitudes[:] = np.linspace(-180, 180, temp_mean.shape[1])

        temp_var[0, :, :] = temp_mean

        hum_var[0, :, :] = hum_mean

        wind_var[0, :, :] = wind_mean



        dataset.close()

    except Exception as e:

        logger.error(f"Error writing results to file: {e}")

        raise



def write_predictions(output_dir, predictions):

    """Writes predicted weather data to a NetCDF file."""

    try:

        dataset = Dataset(f"{output_dir}/predicted_weather_data.nc", 'w', format='NETCDF4_CLASSIC')

        time = dataset.createDimension('time', predictions.shape[0])

        lat = dataset.createDimension('lat', predictions.shape[1])

        lon = dataset.createDimension('lon', predictions.shape[2])

        

        times = dataset.createVariable('time', np.float64, ('time',))

        latitudes = dataset.createVariable('latitude', np.float32, ('lat',))

        longitudes = dataset.createVariable('longitude', np.float32, ('lon',))

        temp_var = dataset.createVariable('temperature', np.float32, ('time', 'lat', 'lon'))

        hum_var = dataset.createVariable('humidity', np.float32, ('time', 'lat', 'lon'))

        wind_var = dataset.createVariable('wind_speed', np.float32, ('time', 'lat', 'lon'))

        

        times[:] = np.arange(predictions.shape[0])

        latitudes[:] = np.linspace(-90, 90, predictions.shape[1])

        longitudes[:] = np.linspace(-180, 180, predictions.shape[2])

        temp_var[:, :, :] = predictions[:, :, :, 0]

        hum_var[:, :, :] = predictions[:, :, :, 1]

        wind_var[:, :, :] = predictions[:, :, :, 2]



        dataset.close()

    except Exception as e:

        logger.error(f"Error writing predictions to file: {e}")

        raise

