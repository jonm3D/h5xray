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
    - --byte_threshold: Minimum bytes required for a dataset to get annotated. Default is 1 MiB (1024 * 1024 bytes).
    - --title: Custom title for the plot. If not specified, the name of the input file will be used.
    - --orientation: Orientation of the plot ('vertical' or 'horizontal'). Default is 'vertical'.
    - --figsize: Size of the figure for the plot specified as width,height (e.g., 10,3). Default is 6,2.

For more detailed information on available arguments, run:
    python script_name.py -h
"""

import argparse
import os
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import platform
import time
from PIL import Image, ImageDraw, ImageFont
import s3fs

def parse_s3_url(s3_url):
    # Parse the S3 URL to extract bucket name and key
    parts = s3_url.replace('s3://', '').split('/')
    bucket_name = parts[0]
    key = '/'.join(parts[1:])
    return bucket_name, key

def open_hdf5_file_from_s3(s3_url):
    # Parse the S3 URL to extract bucket name and key
    bucket_name, key = parse_s3_url(s3_url)

    # Create an s3fs filesystem
    fs = s3fs.S3FileSystem(anon=False)  # Set anon=False if using AWS credentials

    # Open the HDF5 file directly from S3
    with fs.open(f"{bucket_name}/{key}", "rb") as s3_file:
        # Wrap the S3 file in an h5py File object
        hdf5_file = h5py.File(s3_file, "r")

    return hdf5_file

def print_report(df, file_name, elapsed_time, request_size_bytes=1024*1024, cost_per_request=0.0004):
    """
    Prints a report about the HDF5 file data and GET requests, including cost calculation per GET request.
    
    Args:
        df (pd.DataFrame): DataFrame containing the extracted information.
        file_name (str): Name of the file.
        elapsed_time (float): Time taken to process the file.
        request_size_bytes (int): The size of each request in bytes.
        cost_per_request (float): Cost per GET request (default: $0.0004 per request).
    """
    total_datasets = len(df)
    total_requests = df['requests_needed'].sum()
    
    # Calculate the cost for GET requests
    total_cost = total_requests * cost_per_request

    # Extracting top 5 datasets with most requests
    top_datasets = df.nlargest(5, 'requests_needed')
    
    # Reporting system details
    system_info = {
        "OS": os.name,
        "Platform": platform.system(),
        "Platform Version": platform.version(),
        "Python Version": platform.python_version(),
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "Current Working Directory": os.getcwd(),
        "Host Name": platform.node(),
        "Number of CPUs": os.cpu_count()
    }

    print(f"\nReport for {file_name}:")
    print("-" * 50)
    print(f"Elapsed time (s): {elapsed_time:.3f}")
    print(f"Total datasets: {total_datasets}")
    print(f"Total requests: {total_requests}")
    print(f"Request byte size: {request_size_bytes} bytes")
    print(f"Assumed cost per GET request: ${cost_per_request:.4f}")
    print(f"Total cost for file: ${total_cost:.4f}")
    print("-" * 50)
    print("Top 5 datasets with most requests:")
    for index, row in top_datasets.iterrows():
        chunk_info = f"Chunking: {row['chunking']} | Number of Chunks: {row['num_chunks']}" if row['chunking'] else "Contiguous"
        print(f"{row['path']} - {row['requests_needed']} requests | {chunk_info}")
    print("-" * 50)
    print("System Info:")
    for key, value in system_info.items():
        print(f"{key}: {value}")
    print("-" * 50)
    print("\n")



def extract_dataset_info(data_file, path='/', request_size_bytes=2*1024 * 1024):
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
            results.extend(extract_dataset_info(data_file, current_path, request_size_bytes))
        elif isinstance(item, h5py.Dataset):
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

def plot_dataframe(df, request_byte_size, plotting_options={}):
    """
    Plots h5 data as a 'barcode' showing details about dataset size and chunking.

    Args:
        df (pd.DataFrame): DataFrame containing dataset details.
        plotting_options (dict): A dictionary of plotting options. You can specify any of the following options:
            - 'figsize': Size of the figure for the plot specified as (width, height). Default is (8, 2).
            - 'cmap': Colormap for coloring bars. Default is plt.cm.coolwarm.
            - 'max_requests': Maximum number of requests for color normalization. Default is 10.
            - 'font_size': Size of the font for annotations on the plot. Default is 10.
            - 'byte_threshold': Minimum bytes required for a dataset to get annotated. Default is 5 MiB (5 * 1024 * 1024 bytes).
            - 'title': Custom title for the plot. If not specified, the name of the input file will be used.
            - 'debug': Whether to provide a detailed plot for debugging. Default is False (minimal plot).
            - 'output_file': Path to the output image file. If not provided, it defaults to '[input_filename]_xray.png'.

    Returns:
        None
    """
    cmap = plotting_options.get('cmap', plt.cm.coolwarm)
    max_requests = plotting_options.get('max_requests', 10)
    font_size = plotting_options.get('font_size', 7)
    byte_threshold = plotting_options.get('byte_threshold', 5*1024*1024)
    debug = plotting_options.get('debug', False)

    if 'figsize' in plotting_options:
        figsize = plotting_options['figsize']
    elif debug:
        figsize = (8, 2)
    else:
        figsize = (6, 0.8)
        
    output_file = plotting_options.get('output_file', None)
    title = plotting_options.get('title', None)

    df['norm_requests'] = df['requests_needed'].apply(lambda x: min(x, max_requests) / max_requests)
    df['color'] = df['norm_requests'].apply(lambda x: cmap(x))

    plt.figure(figsize=figsize)
    stacked_value = 0 

    edge_color = plotting_options.get('edge_color', 'black' if debug else None)
    linewidth = plotting_options.get('linewidth', 0.05 if debug else 0)
    
    for index, row in df.iterrows():
        plt.barh('Combined', row['bytes'], left=stacked_value, color=row['color'], edgecolor=edge_color, linewidth=linewidth)
        if debug and row['bytes'] > byte_threshold:
            chunk_center = stacked_value + row['bytes'] / 2
            plt.text(chunk_center, 'Combined', row['name'], ha='center', va='center', color='black', fontsize=font_size, rotation=90)
        stacked_value += row['bytes']

    if debug:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_requests))
        sm.set_array([])
        plt.colorbar(sm).set_label(f'{request_byte_size/1024**2:.0f} MiB Requests', rotation=270, labelpad=15)
        plt.xticks([])  
        plt.yticks([])
        plt.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.title(f"H5XRAY - {title} - Total Size: {stacked_value / (1024**2):.2f} MiB")
    else:
        plt.axis('off')
    
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {output_file}")
    else:
        try:
            plt.show()
        except Exception as e:
            print(f"Error displaying plot: {e}")

def analyze(input_file, request_byte_size=2*1024*1024, plotting_options={}, report=True, cost_per_request=0.0004):
    """
    Main function for plotting / reporting details of an HDF5 file.

    Args:
        input_file (str): Path to the input HDF5 file.
        request_byte_size (int): The size of each request in bytes. Default is 2MiB (2*1024*1024 bytes).
        plotting_options (dict): A dictionary of plotting options. Refer to the plot_dataframe function for available options.
        report (bool): Whether to print a report about the HDF5 file. Default is True.
        cost_per_request (float): Cost per GET request (default: $0.0004 per request).

    Returns:
        None
    """

    # Try to open the HDF5 file and handle potential errors
    try:
        with h5py.File(input_file, "r") as hdf5_file:
            start_time = time.time()
            file_info = extract_dataset_info(hdf5_file, request_size_bytes=request_byte_size)
            end_time = time.time()
            elapsed_time = end_time - start_time
            df = pd.DataFrame(file_info)
    except OSError as e:
        print(f"Error reading the HDF5 file: {e}")
        exit(1)
        
    # Update the title in the plotting options
    if 'title' not in plotting_options:
        file_name = os.path.splitext(os.path.basename(input_file))[0]  # Get the original filename without extension
        plotting_options['title'] = file_name

    if report:
        print_report(df, input_file, elapsed_time, request_byte_size, cost_per_request)

    plot_dataframe(df, request_byte_size, plotting_options)
    
def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize HDF5 file structures.")
    input_file_help = "Path to the input HDF5 file. This can be either a local file path or an S3 URL (e.g., s3://my-bucket/path/to/file.h5)."    
    parser.add_argument("input_file", type=str, help=input_file_help)
    parser.add_argument("--output_file", "-o", type=str, default=None, help="Path to the output image file. If not provided, it defaults to '[input_filename]_xray.png'.")
    parser.add_argument("--debug", "-d", action='store_true', help="Provide a detailed plot for debugging. Default is a minimal plot.")
    parser.add_argument("--font_size", "-f", type=int, default=7, help="Font size for annotations on the plot. Default is 7.")
    parser.add_argument("--byte_threshold", "-b", type=int, default=1024*1024, help="Minimum bytes required for a dataset to get annotated. Default is 1MiB.")
    parser.add_argument("--title", "-t", type=str, default=None, help="Title for the plot. If not specified, the name of the input file will be used.")
    parser.add_argument("--figsize", "-s", type=lambda s: [int(item) for item in s.split(',')], default=[6,2], help="Size of the figure for the plot as width,height. Default is 6,2.")
    parser.add_argument("--request_byte_size", "-r", type=int, default=2*1024*1024, help="Size of each request in bytes. Default is 2 MiB (2*1024*1024 bytes).")
    parser.add_argument("--cost_per_request", "-c", type=float, default=0.0004, help="Cost per GET request. Default is $0.0004 per request.")

    args = parser.parse_args()
    plotting_options = {
        'output_file': args.output_file,
        'debug': args.debug,
        'font_size': args.font_size,
        'byte_threshold': args.byte_threshold,
        'title': args.title,
        'figsize': args.figsize
    }

    # Check if the input file is an S3 URL
    if args.input_file.startswith('s3://'):
        # Open the HDF5 file directly
        hdf5_file = s3_utils.open_hdf5_file_from_s3(args.input_file)
    else: #local path
        hdf5_file = args.input_file

    analyze(hdf5_file, request_byte_size=args.request_byte_size, plotting_options=plotting_options, cost_per_request=args.cost_per_request)

if __name__ == "__main__":
    main()