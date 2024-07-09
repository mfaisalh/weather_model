import logging

import yaml



def setup_logging():

    """Setup logging configuration."""

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')



def load_config(config_file):

    """Load configuration from a YAML file."""

    with open(config_file, 'r') as file:

        config = yaml.safe_load(file)

    return config

