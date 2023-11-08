"""
HDF5 File Structure Analyzer and Plotter

Description:
This script inspects the structure of an HDF5 file and visualizes it, allowing users to better 
understand and analyze the storage layout and size of various groups and datasets within the file. 

Dependencies:
    - Essential: matplotlib, h5py, pandas, seaborn, PIL
    - Additional: python-igraph, kaleido (install manually if not present)

Usage:
    python h5xray.py INPUT_FILE --output_file OUTPUT_FILE_PATH [additional_arguments]

Arguments:
    - INPUT_FILE: The path to the input HDF5 file you wish to analyze.
    - --output_file OUTPUT_FILE_PATH: The path where the generated image should be saved.
    - --annotate: Flag to include dataset names on the plot. Default is False.
    - --font_size: Size of the font for annotations on the plot. Default is 7.
    - --title: Custom title for the plot. If not specified, the name of the input file will be used.
    - --orientation: Orientation of the plot ('vertical' or 'horizontal'). Default is 'vertical'.
    - --figsize: Size of the figure for the plot specified as width,height (e.g., 10,3). Default is 6,2.

For more detailed information on available arguments, run:
    python script_name.py -h
"""

import argparse
import logging
import os
import platform
import socket
import sys
import time

import h5py
import icepyx as ipx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import s3fs
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image, ImageDraw, ImageFont


def open_hdf5_file_from_s3(s3_url, earthdata_uid, earthdata_pwd):
    """
    Opens an HDF5 file from S3 using Earthdata credentials.
    
    Args:
        s3_url (str): The S3 URL of the HDF5 file.
        earthdata_uid (str): Earthdata username.
        earthdata_pwd (str): Earthdata password.
        
    Returns:
        h5py.File: The opened HDF5 file.
    """
    
    # Create an icepyx Query Object for Earthdata login
    # Dummy values for spatial_extent and date_range since we're just using it for the login capability
    reg = ipx.Query('ATL03', [-45, 58, -35, 75], ['2019-11-30', '2019-11-30'])
    reg.earthdata_login(earthdata_uid, earthdata_pwd, s3token=True)
    
    # Set up S3 Filesystem with Earthdata credentials
    s3 = s3fs.S3FileSystem(
        key=reg._s3login_credentials['accessKeyId'],
        secret=reg._s3login_credentials['secretAccessKey'],
        token=reg._s3login_credentials['sessionToken']
    )
    
    # Open the HDF5 file from S3 and return it
    s3f = s3.open(s3_url, 'rb')
    return h5py.File(s3f, 'r')


def parse_s3_url(s3_url):
    # Parse the S3 URL to extract bucket name and key
    parts = s3_url.replace('s3://', '').split('/')
    bucket_name = parts[0]
    key = '/'.join(parts[1:])
    return bucket_name, key

def check_if_aws(verbose=False):
    """Check if we're running on an AWS EC2 instance and optionally print details."""
    metadata_url = "http://169.254.169.254/latest/meta-data/"

    def print_system_info():
        """Print information about the current system."""
        print(f"System: {platform.system()}")
        print(f"Node Name (Hostname): {platform.node()}")
        print(f"Release: {platform.release()}")
        print(f"Version: {platform.version()}")
        print(f"Machine: {platform.machine()}")
        print(f"Processor: {platform.processor()}")
        print(f"Python version: {platform.python_version()}")
        try:
            print(f"IP Address: {socket.gethostbyname(socket.gethostname())}")
        except:
            print("Could not fetch IP Address.")
        print(f"Current Working Directory: {os.getcwd()}")

    try:
        # Check if we're on EC2: a simple way is to fetch the instance-id.
        instance_id = requests.get(metadata_url + "instance-id", timeout=1).text
        if not instance_id:
            if verbose:
                print("Not running on an AWS EC2 instance. System details are as follows:")
                print_system_info()
            return False

        if verbose:
            print("Running on an AWS EC2 instance with the following details:")
            
            # Fetch all available metadata paths
            paths = requests.get(metadata_url, timeout=1).text.split("\n")
            
            for path in paths:
                # Skip directories (they end with '/')
                if path.endswith('/'):
                    continue
                
                try:
                    detail = requests.get(metadata_url + path, timeout=1).text
                    print(f"{path}: {detail}")
                except requests.RequestException as e:
                    print(f"Failed to fetch {path}: {e}")

        return True
    except requests.RequestException:
        # If we can't fetch the instance-id, we're likely not on EC2.
        if verbose:
            print("Not running on an AWS EC2 instance. System details are as follows:")
            print_system_info()
        return False


def print_report(df, file_name, elapsed_time, request_size_bytes=2*1024*1024, 
                 cost_per_request=0.0004e-3, as_str=False):
    """
    Prints or returns a report about the HDF5 file data and GET requests, 
    including cost calculation per GET request.
    
    Args:
        df (pd.DataFrame): DataFrame containing the extracted information.
        file_name (str): Name of the file.
        elapsed_time (float): Time taken to process the file.
        request_size_bytes (int): The size of each request in bytes.
        cost_per_request (float): Cost per GET request (default: $0.0004 per 1000 requests).
        as_str (bool): If True, returns the report as a string; otherwise, prints the report (default: False).
        
    Returns:
        str: The report as a string if as_str is True, otherwise None.
    """
    separator = "-" * 50
    report_lines = [
        # f"Report for {file_name}",
        # separator,
        f"Processing Time: {elapsed_time:.3f} seconds",
        f"Data Summary:",
        f"  - Total datasets: {len(df)}",
        f"  - Total requests: {df['requests_needed'].sum()}",
        f"  - Request byte size: {request_size_bytes/(1024**2)} MiB",
        f"Cost Analysis:",
        f"  - Assumed cost per 1,000 GET requests: ${cost_per_request*1000:.10f}",
        f"  - Total cost for file: ${df['requests_needed'].sum() * cost_per_request:.10f}",
        separator,
        "Top 5 datasets by number of requests:"
    ]
    top_datasets = df.nlargest(5, 'requests_needed')
    for _, row in top_datasets.iterrows():
        chunk_info = (f"Chunking: {row['chunking']} | "
                      f"Number of Chunks: {row['num_chunks']}") if row['chunking'] else "Contiguous"
        report_lines.append(f"  - {row['path']} - {row['requests_needed']} requests | {chunk_info}")

    report_lines.append(separator)
    report_lines.append("System Information:")
    system_info = {
        "OS": os.name,
        "Platform": platform.system(),
        "Platform Release": platform.release(),
        "Python Version": platform.python_version(),
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "Working Directory": os.getcwd(),
        "Host Name": platform.node(),
        "CPU Count": os.cpu_count()
    }
    for key, value in system_info.items():
        report_lines.append(f"  - {key}: {value}")

    report_lines.append(separator)
    report_str = "\n".join(report_lines)
    
    if as_str:
        return report_str
    else:
        print(report_str)

def extract_dataset_info(data_file, path='/', request_size_bytes=2*1024 * 1024):
    """
    Extracts information about datasets within an H5 file.

    This function traverses the groups and datasets starting from the specified path
    within the H5 file. It collects information on datasets including the path,
    size, chunking, and the number of requests needed to read the dataset based on 
    a specified request size.

    Parameters:
    - data_file (h5py.File): An opened H5 file object.
    - path (str): The starting path within the H5 file from which to begin extraction.
    - request_size_bytes (int): The size in bytes of a single read request, used to 
                                calculate the number of requests needed for each dataset.

    Returns:
    - List[Dict[str, Any]]: A list of dictionaries, where each dictionary contains 
                             information about a dataset. Keys include 'top', 'name', 
                             'path', 'type', 'chunking', 'num_chunks', 'bytes', 
                             'attributes', and 'requests_needed'.

    """
    results = []
    
    # Helper function to compute the number of chunks for a dataset.
    def compute_num_chunks(dataset_shape, chunk_shape):
        return [np.ceil(ds/ch) for ds, ch in zip(dataset_shape, chunk_shape)]

    # Helper function to compute the number of requests needed for a dataset.
    def compute_requests_needed(byte_size, request_size_bytes):
        return np.ceil(byte_size / request_size_bytes)

    for name, item in data_file[path].items():
        current_path = f"{path}/{name}" if path != '/' else f"/{name}"
        
        if isinstance(item, h5py.Group):
            # print('group!', current_path, item)
            results.extend(extract_dataset_info(data_file, current_path, request_size_bytes))
        elif isinstance(item, h5py.Dataset):
            # print('dataset!', current_path, item)
            chunk_shape = item.chunks
            num_chunks = compute_num_chunks(item.shape, chunk_shape) if chunk_shape else None
            
            dataset_info = {
                'top': current_path.split('/')[1],
                'name': current_path.split('/')[-1],
                'path': current_path,
                'type': 'Dataset',
                'chunking': chunk_shape,
                'num_chunks': num_chunks,
                'bytes': item.id.get_storage_size(),
                'attributes': dict(item.attrs),
                'requests_needed': compute_requests_needed(item.id.get_storage_size(), request_size_bytes)
            }
            results.append(dataset_info)
    
    return results

def plot_dataframe(df, plotting_options={}):
    """
    Plots a barcode-style visualization indicating how many requests are needed per dataset for an H5 file.

    Parameters:
    - df (DataFrame): A pandas DataFrame with at least 'bytes' and 'requests_needed' columns.
    - plotting_options (dict, optional): A dictionary containing options for plot customization such as:
      - cmap: The colormap for the plot.
      - font_size: The font size for the text labels.
      - debug: If True, will print additional plot details and a colorbar.
      - label_top_n: The number of top segments to label.
      - figsize: A tuple defining the figure size.
      - output_file: Path to save the plot image.
      - title: The title of the plot.
      - edge_color: The color of the edges of the bars.
      - linewidth: The width of the edges of the bars.

    Returns:
    - fig: The matplotlib figure object for the plot.
    """

    cmap = plotting_options.get('cmap', plt.cm.YlOrRd)
    font_size = plotting_options.get('font_size', 7)
    debug = plotting_options.get('debug', False)

    if debug:
        label_top_n = plotting_options.get('label_top_n', 10)  
    else:
        label_top_n = plotting_options.get('label_top_n', 0)  
    
    # Determine the bounds for normalization
    requests_min = 0

    # Color up to 5 requests so small files dont appear all red
    requests_max = np.max([ 5, np.ceil(df['requests_needed'].max())])
    
    if 'figsize' in plotting_options:
        figsize = plotting_options['figsize']
    elif debug:
        figsize = (8, 2)
    else:
        figsize = (6, 0.8)

    output_file = plotting_options.get('output_file', None)
    title = plotting_options.get('title', None)

    fig, ax = plt.subplots(figsize=figsize)
    stacked_value = 0

    edge_color = plotting_options.get('edge_color', 'black' if debug else None)
    linewidth = plotting_options.get('linewidth', 0.05 if debug else 0)

    top_segments_idx = df['bytes'].nlargest(label_top_n).index

    norm = Normalize(vmin=requests_min, vmax=requests_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for index, row in df.iterrows():
        color = sm.to_rgba(row['requests_needed'])

        ax.barh('Combined', row['bytes'], left=stacked_value, color=color, edgecolor=edge_color, linewidth=linewidth)

        if index in top_segments_idx:
            chunk_center = stacked_value + row['bytes'] / 2
            ax.text(chunk_center, 'Combined', row['name'], ha='center', va='center', color='black', fontsize=font_size, rotation=90)

        stacked_value += row['bytes']

    if debug:
        cb = plt.colorbar(sm, orientation='vertical', ax=ax)
        cb.set_label('Requests Needed', rotation=270, labelpad=15)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_title(f"H5XRAY - {title if title else 'Dataset'} - Total Size: {stacked_value / (1024**2):.2f} MiB")
    else:
        ax.axis('off')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {output_file}")
    else:
        try:
            plt.show()
        except Exception as e:
            print(f"Error displaying plot: {e}")

    return fig

def open_hdf5_file_from_s3(s3_url, earthdata_uid, earthdata_pwd):
    """
    Opens an HDF5 file from S3 using Earthdata credentials.

    Args:
        s3_url (str): The S3 URL of the HDF5 file.
        earthdata_uid (str): Earthdata username.
        earthdata_pwd (str): Earthdata password.

    Returns:
        h5py.File: The opened HDF5 file.
    """
    reg = ipx.Query('ATL03', [-45, 58, -35, 75], ['2019-11-30', '2019-11-30'])
    reg.earthdata_login(earthdata_uid, earthdata_pwd, s3token=True)
    s3 = s3fs.S3FileSystem(
        key=reg._s3login_credentials['accessKeyId'],
        secret=reg._s3login_credentials['secretAccessKey'],
        token=reg._s3login_credentials['sessionToken']
    )
    s3f = s3.open(s3_url, 'rb')
    return h5py.File(s3f, 'r')

def open_hdf5_file_local(file_path):
    """
    Opens a local HDF5 file.

    Args:
        file_path (str): The file path of the HDF5 file.

    Returns:
        h5py.File: The opened HDF5 file.
    """
    return h5py.File(file_path, "r")

def analyze(input_file, request_byte_size=2*1024*1024, plotting_options={}, report=True, 
            cost_per_request=0.0004e-3, aws_access_key=None, aws_secret_key=None, report_type='print'):
    """
    Main function for plotting / reporting details of an HDF5 or NetCDF file.

    Args:
        input_file (str): Path to the input HDF5 or NetCDF file or S3 URL.
        request_byte_size (int): The size of each request in bytes. Default is 2MiB (2*1024*1024 bytes).
        plotting_options (dict): A dictionary of plotting options.
        report (bool): Whether to print a report about the HDF5 or NetCDF file. Default is True.
        cost_per_request (float): Cost per GET request (default: $0.0004 per 1000 requests).
        aws_access_key (str, optional): AWS access key for S3 URLs.
        aws_secret_key (str, optional): AWS secret key for S3 URLs.
        report_type (str): Format of the report output, either 'str' for string or 'print' to print directly.

    Returns:
        matplotlib.figure.Figure: The figure object of the plot, if applicable.
        str: The report string, if report is True and report_type is 'str'.
    """

    as_str = report_type == 'str'
    report_str = None

    # Check if the input file is an S3 URL
    is_s3_url = input_file.startswith("s3://")

    try:
        start_time = time.time()
        # Context management ensures files are closed after processing
        if is_s3_url:
            # S3 URL case: handle AWS or Earthdata S3 access as before
            with open_hdf5_file_from_s3(input_file, aws_access_key, aws_secret_key) as data_file:
                data_info = extract_dataset_info(data_file, request_size_bytes=request_byte_size)
        else:
            # file_extension = os.path.splitext(input_file)[1]
            with open_hdf5_file_local(input_file) as data_file:
                data_info = extract_dataset_info(data_file, request_size_bytes=request_byte_size)

        df = pd.DataFrame(data_info)
        elapsed_time = time.time() - start_time
        
        if report:
            report_str = print_report(df, os.path.basename(input_file), elapsed_time, 
                                      request_size_bytes=request_byte_size, 
                                      cost_per_request=cost_per_request, as_str=as_str)
                
        fig = plot_dataframe(df, plotting_options)

        return fig, report_str if as_str else None

    except Exception as e:
        print(f"Error in analyzing the file: {e}")
        return None, None
