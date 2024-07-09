import numpy as np

import logging



logger = logging.getLogger()



def process_weather_data(temperature, humidity, wind_speed):

    """Processes weather data to compute means."""

    temp_mean = np.mean(temperature, axis=0)

    hum_mean = np.mean(humidity, axis=0)

    wind_mean = np.mean(wind_speed, axis=0)

    return temp_mean, hum_mean, wind_mean



def display_weather_data(temp_mean, hum_mean, wind_mean):

    """Displays processed weather data."""

    logger.info(f"Average Temperature: {temp_mean.mean():.2f}")

    logger.info(f"Average Humidity: {hum_mean.mean():.2f}")

    logger.info(f"Average Wind Speed: {wind_mean.mean():.2f}")

