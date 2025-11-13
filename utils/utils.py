import yaml
from decimal import Decimal
from datetime import datetime, timedelta
import json
import pytz


def is_today(timestamp: int) -> bool:
    tz = pytz.FixedOffset(420)  # GMT-4 is UTC-4 hours, i.e., -240 minutes
    dt = datetime.fromtimestamp(timestamp, tz)
    return dt.date() == datetime.now().date()

def is_next_day(prev_timestamp: int, new_timestamp: int) -> bool:
    tz = pytz.FixedOffset(420)  # GMT-4 is UTC-4 hours, i.e., -240 minutes

    new_dt = datetime.fromtimestamp(new_timestamp, tz)
    new_date = new_dt.date()

    previous_dt = datetime.fromtimestamp(prev_timestamp, tz)
    previous_date = previous_dt.date()

    # Check if the new date is exactly one day after the previous date
    date_difference = new_date - previous_date
    return date_difference >= timedelta(days=1)


def current_epoch() -> int:
    """return current timestamp

    Returns:
        int: timestamp in seconds
    """
    now = datetime.now()
    return int(now.timestamp())


def yaml_parser(filePath: str) -> dict:
    """
    Parses a YAML file and returns its content as a dictionary.
    The function opens the file located at `filePath` and uses the `yaml` library to parse it.

    Args:
        filePath (str): The path to the YAML file to be parsed.

    Returns:
        dict: The parsed YAML data.
    """
    with open(filePath) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def json_parser(filePath: str) -> dict:
    """
    Parses a JSON file and returns its content as a dictionary.
    The function opens the file located at `filePath` and uses the `json` library to parse it.

    Args:
        filePath (str): The path to the JSON file to be parsed.

    Returns:
        dict: The parsed JSON data.
    """
    with open(filePath) as f:
        value = json.load(f)
    return value


def json_exporter(d, filePath):
    """
    Exports a dictionary to a JSON file at the specified path with pretty formatting.
    This function writes the dictionary `d` to a file specified by `filePath`, formatting the
    JSON output with an indentation of 4 spaces.

    Args:
        d (dict): The dictionary to export.
        filePath (str): The path where the JSON file will be saved.
    """
    with open(filePath, "w") as fp:
        json.dump(d, fp, indent=4)


def format_float_by_given_granularity_into_str(
    rawNumber: float, referenceStr: str
) -> str:
    """format the raw number into a str as per referenceStr granularity

    Args:
        rawNumber (float): raw number
        referenceStr (str): refence number is string

    Returns:
        str: fotmatted number
    """
    return str(
        (
            Decimal(int(Decimal(rawNumber) / Decimal(referenceStr)))
            * Decimal(referenceStr)
        ).quantize(Decimal(referenceStr))
    )
