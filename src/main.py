from mpi4py import MPI

import numpy as np

import logging

import sys

from weather_io import read_weather_data, write_results, write_predictions

from data_processing import process_weather_data, display_weather_data

from utils import setup_logging, load_config

from visualization import visualize_data

from prediction import prepare_data, train_model, predict_future

import os



# Initialize MPI

comm = MPI.COMM_WORLD

rank = comm.Get_rank()

size = comm.Get_size()



# Setup logging

setup_logging()

logger = logging.getLogger()



def main():

    if len(sys.argv) < 2:

        if rank == 0:

            logger.error("Usage: mpirun -np <num_processes> python main.py <config_file>")

        comm.Abort(1)



    config_file = sys.argv[1]

    config = load_config(config_file)



    weather_data_file = config['input_file']

    historical_data_file = config['historical_data_file']

    output_dir = config['output_dir']

    if not os.path.exists(output_dir):

        os.makedirs(output_dir)



    # Only rank 0 reads the file

    if rank == 0:

        try:

            temperature, humidity, wind_speed = read_weather_data(weather_data_file)

            hist_temperature, hist_humidity, hist_wind_speed = read_weather_data(historical_data_file)

            data_chunks = np.array_split(np.stack((temperature, humidity, wind_speed), axis=0), size, axis=1)

        except Exception as e:

            logger.error(f"Failed to read and prepare data: {e}")

            comm.Abort(1)

    else:

        data_chunks = None



    # Scatter data to all processes

    local_data = comm.scatter(data_chunks, root=0)



    # Each process processes its data

    temp_mean, hum_mean, wind_mean = process_weather_data(local_data[0], local_data[1], local_data[2])



    # Gather results from all processes

    results = comm.gather((temp_mean, hum_mean, wind_mean), root=0)



    # Rank 0 combines and displays the results

    if rank == 0:

        final_temp_mean = np.mean([result[0] for result in results], axis=0)

        final_hum_mean = np.mean([result[1] for result in results], axis=0)

        final_wind_mean = np.mean([result[2] for result in results], axis=0)

        display_weather_data(final_temp_mean, final_hum_mean, final_wind_mean)

        write_results(output_dir, final_temp_mean, final_hum_mean, final_wind_mean)



        # Prepare data for prediction

        X = prepare_data(hist_temperature, hist_humidity, hist_wind_speed)

        y_temp = hist_temperature.flatten()

        y_hum = hist_humidity.flatten()

        y_wind = hist_wind_speed.flatten()



        # Train models

        temp_model = train_model(X, y_temp, model_type='random_forest')

        hum_model = train_model(X, y_hum, model_type='random_forest')

        wind_model = train_model(X, y_wind, model_type='random_forest')



        # Predict future weather

        current_data = np.stack((final_temp_mean, final_hum_mean, final_wind_mean), axis=-1).reshape(-1, 3)

        future_temp = predict_future(temp_model, current_data).reshape(final_temp_mean.shape)

        future_hum = predict_future(hum_model, current_data).reshape(final_hum_mean.shape)

        future_wind = predict_future(wind_model, current_data).reshape(final_wind_mean.shape)



        # Combine predictions

        future_predictions = np.stack((future_temp, future_hum, future_wind), axis=-1)



        # Write predictions to file

        write_predictions(output_dir, future_predictions)



        # Visualize predictions

        visualize_data(future_temp, 'Predicted Temperature', f"{output_dir}/pred_temperature.png")

        visualize_data(future_hum, 'Predicted Humidity', f"{output_dir}/pred_humidity.png")

        visualize_data(future_wind, 'Predicted Wind Speed', f"{output_dir}/pred_wind_speed.png")



if __name__ == "__main__":

    main()

    MPI.Finalize()

