import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import yaml
import streamlit as st
import boto3
import os

# Connection to S3
conn = boto3.resource(
    service_name="s3",
    region_name=os.environ["AWS_DEFAULT_REGION"],
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)


@st.experimental_memo(ttl=600)
def load_data_s3(bucket_name: str, key_name: str) -> pd.DataFrame:
    """
    load_data_s3: this function load data from s3

    Args:
        bucket_name (str): path to data file list of variables to be recoded (specified by user)
        key_name (str): key name of file
    Returns:
        (DataFrame): dataframe of imported data
    """

    data_obj = conn.Bucket(bucket_name).Object(key_name).get()
    # data = pd.read_csv(data_obj["Body"], index_col=0)
    data = pd.read_csv(data_obj["Body"])

    return data


@st.cache
def load_data(file_path, file_name):
    """
    load_data: this function load data from a user-specified
    file path and file name

    Args:
        file_path (Path): path to data file list of variables to be recoded (specified by user)
        file_name (str): name of data file
    Returns:
        (DataFrame): dataframe of imported data
    """
    data_path = Path(file_path, file_name)

    if data_path.is_file():
        return pd.read_csv(data_path)
    else:
        raise Exception(f"Raw data file not found at {data_path}!")


def save_fig(
    fig_id, tight_layout=True, image_directory=None, fig_extension="png", resolution=300
):
    if image_directory is not None:
        file_name = f"{fig_id}.{fig_extension}"
        path = Path(image_directory, file_name)

        print("Saving figure", fig_id)

        if tight_layout:
            plt.tight_layout()

        plt.savefig(path, format=fig_extension, dpi=resolution)

    else:
        print("No image directory is provided, image is not saved.")


def load_yaml(filename):
    with open(filename) as f:
        return yaml.safe_load(f)
