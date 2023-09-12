"""
HDF5 File Structure Analyzer and Plotter

Description:
This script inspects the structure of an HDF5 file and visualizes it, allowing users to better 
understand and analyze the storage layout and size of various groups and datasets within the file. 
The visualization is saved as an image.

Dependencies:
    - Essential: matplotlib, h5py, pandas, seaborn, PIL
    - Optional: python-igraph, kaleido

Usage:
    python script_name.py INPUT_FILE --output_file OUTPUT_FILE_PATH [additional_arguments]

Arguments:
    - INPUT_FILE: The path to the input HDF5 file you wish to analyze.
    - --output_file OUTPUT_FILE_PATH: The path where the generated image should be saved.
    - --annotate: Flag to include dataset names on the plot. Default is False.
    - --font_size: Size of the font for annotations on the plot. Default is 5.
    - --byte_threshold: Minimum bytes required for a dataset to get annotated. Default is 0.5 MB (512 * 1024 bytes).
    - --title: Custom title for the plot. If not specified, the name of the input file will be used.
    - --orientation: Orientation of the plot ('vertical' or 'horizontal'). Default is 'vertical'.
    - --figsize: Size of the figure for the plot specified as width,height (e.g., 10,3). Default is 10,3.

For more detailed information on available arguments, run:
    python script_name.py -h
"""

import argparse
import os
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import textwrap
import platform
import time
import psutil
from PIL import Image, ImageDraw, ImageFont

# If igraph or kaleido is not installed, prompt user to manually install them.
# Typically, this is unconventional for scripts. Ideally, all dependencies should be listed in a requirements.txt file.
try:
    import igraph as ig
    import kaleido
except ImportError:
    print("Please manually install python-igraph and kaleido to proceed.")
    exit(1)

def print_report(df, file_name, elapsed_time, request_size_bytes=1024*1024):
    """
    Prints a report about the HDF5 or netCDF file based on the extracted information.
    
    Args:
        df (pd.DataFrame): DataFrame containing the extracted information.
        file_name (str): Name of the file.
        elapsed_time (float): Time taken to process the file.
        request_size_bytes (int): The size of each request in bytes.
    """
    total_datasets = len(df)
    total_requests = df['requests_needed'].sum()

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
    file_type = None
    
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

def plot_dataframe(df, plotting_options={}):
    """
    Plots the DataFrame containing dataset details.

    Args:
        df (pd.DataFrame): DataFrame containing dataset details.
        plotting_options (dict): A dictionary of plotting options. You can specify any of the following options:
            - 'figsize': Size of the figure for the plot specified as (width, height). Default is (8, 2).
            - 'cmap': Colormap for coloring bars. Default is plt.cm.Spectral_r.
            - 'max_requests': Maximum number of requests for color normalization. Default is 10.
            - 'font_size': Size of the font for annotations on the plot. Default is 10.
            - 'byte_threshold': Minimum bytes required for a dataset to get annotated. Default is 5MB (5 * 1024 * 1024 bytes).
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
    title = plotting_options.get('title', output_file)


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
        plt.colorbar(sm).set_label('Requests Needed', rotation=270, labelpad=15)
        plt.xticks([])  
        plt.yticks([])
        plt.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.title(f"H5XRAY - {title} - Total Size: {stacked_value / (1024**2):.2f} MB")
    else:
        plt.axis('off')
    
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
    else:
        try:
            plt.show()
        except Exception as e:
            print(f"Error displaying plot: {e}")

def analyze(input_file, request_byte_size=2*1024*1024, plotting_options={}, report=True):
    """
    Analyze and visualize the structure of an HDF5 file.

    Args:
        input_file (str): Path to the input HDF5 file.
        request_byte_size (int): The size of each request in bytes. Default is 2MB (2*1024*1024 bytes).
        plotting_options (dict): A dictionary of plotting options. You can specify any of the following options:
            - 'figsize': Size of the figure for the plot specified as (width, height). Default is (8, 2).
            - 'cmap': Colormap for coloring bars. Default is plt.cm.Spectral_r.
            - 'max_requests': Maximum number of requests for color normalization. Default is 10.
            - 'font_size': Size of the font for annotations on the plot. Default is 10.
            - 'byte_threshold': Minimum bytes required for a dataset to get annotated. Default is 1MB (1024 * 1024 bytes).
            - 'title': Custom title for the plot. If not specified, the name of the input file will be used.
            - 'debug': Whether to provide a detailed plot for debugging. Default is False (minimal plot).
            - 'output_file': Path to the output image file. If not provided, it defaults to '[input_filename]_xray.png'.
        report (bool): Whether to print a report about the HDF5 file. Default is True.

    Returns:
        None
    """
    # Determine the output filename if not specified by the user
    if 'output_file' not in plotting_options or plotting_options['output_file'] == 'terminal':
        base_name = os.path.splitext(input_file)[0]  # Get the name without extension
        plotting_options['output_file'] = base_name + "_xray.png"

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

    if report:
        print_report(df, input_file, elapsed_time, request_byte_size)

    plot_dataframe(df, plotting_options)


    
def main():
    #...
    parser = argparse.ArgumentParser(description="Analyze and visualize HDF5 file structures.")
    parser.add_argument("input_file", type=str, help="Path to the input HDF5 file.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to the output image file. If not provided, it defaults to '[input_filename]_xray.png'.")
    parser.add_argument("--debug", action='store_true', help="Provide a detailed plot for debugging. Default is a minimal plot.")
    parser.add_argument("--font_size", type=int, default=5, help="Font size for annotations on the plot. Default is 5.")
    parser.add_argument("--byte_threshold", type=int, default=1024*1024, help="Minimum bytes required for a dataset to get annotated. Default is 1MB.")
    parser.add_argument("--title", type=str, default=None, help="Title for the plot. If not specified, the name of the input file will be used.")
    # parser.add_argument("--figsize", type=lambda s: [int(item) for item in s.split(',')], default=[7,1], help="Size of the figure for the plot as width,height. Default is 6,2.")
    
    args = parser.parse_args()
    analyze(args.input_file, output_file=args.output_file, 
            debug=args.debug, font_size=args.font_size, byte_threshold=args.byte_threshold, title=args.title, figsize=args.figsize)

if __name__ == "__main__":
    main()