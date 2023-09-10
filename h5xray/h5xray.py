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
from PIL import Image, ImageDraw, ImageFont

# If igraph or kaleido is not installed, prompt user to manually install them.
# Typically, this is unconventional for scripts. Ideally, all dependencies should be listed in a requirements.txt file.
try:
    import igraph as ig
    import kaleido
except ImportError:
    print("Please manually install python-igraph and kaleido to proceed.")
    exit(1)

def extract_dataset_info(hdf5_file, path='/', request_size_bytes=1024 * 1024):
    """
    Recursively extract information about groups and datasets in an HDF5 file.

    Args:
        hdf5_file (h5py.File): The HDF5 file object.
        path (str): The current path in the HDF5 structure. Defaults to root ('/').
        request_size_bytes (int): The size of each request in bytes.

    Returns:
        list: A list of dictionaries, where each dictionary has details about a group or dataset,
              including the number of requests required to read the dataset.
    """
    results = []
    # Navigate the current group's items.
    for name, item in hdf5_file[path].items():
        # Construct the full path for the current item.
        current_path = f"{path}/{name}" if path != '/' else f"/{name}"
        
        if isinstance(item, h5py.Group):
            # Recursively explore the contents of the group.
            results.extend(extract_dataset_info(hdf5_file, current_path, request_size_bytes))
        elif isinstance(item, h5py.Dataset):
            # If it's a dataset, extract dataset details.
            depth = current_path.count('/') - 1  # Depth is based on the number of slashes in the path.

            # Calculate the number of requests needed to read the dataset
            dataset_size_bytes = item.size * item.dtype.itemsize
            requests_needed = (dataset_size_bytes + request_size_bytes - 1) // request_size_bytes

            dataset_info = {
                'top': current_path.split('/')[1],  # Top level group for visualization
                'name': current_path.split('/')[-1],
                'path': current_path,
                'type': 'Dataset',
                'depth': depth,
                'chunking': item.chunks,
                'bytes': item.id.get_storage_size(),
                'start_byte': item.id.get_offset(),
                'attributes': dict(item.attrs),
                'requests_needed': requests_needed,  # Number of requests required to read the dataset
            }
            results.append(dataset_info)
    return results

def plot_dataframe(df, annotate=True, font_size=10, byte_threshold=0, title="", 
                   orientation='horizontal', figsize=(7, 9), output_file=None, max_requests=15, minimal=False):
    
    cmap = plt.cm.Reds
    df['norm_requests'] = df['requests_needed'].apply(lambda x: min(x, max_requests) / max_requests)
    df['color'] = df['norm_requests'].apply(lambda x: cmap(x))

    plt.figure(figsize=figsize)
    stacked_value = 0 

    edge_color = 'black' if annotate else None
    linewidth = 0.05 if annotate else 0

    for index, row in df.iterrows():
        if orientation == 'horizontal':
            plt.bar('Combined', row['bytes'], bottom=stacked_value, color=row['color'], edgecolor=edge_color, linewidth=linewidth)
            if annotate and row['bytes'] > byte_threshold:
                chunk_center = stacked_value + row['bytes'] / 2
                plt.text('Combined', chunk_center, row['name'], ha='center', va='center', color='black', fontsize=font_size)
        else:
            plt.barh('Combined', row['bytes'], left=stacked_value, color=row['color'], edgecolor=edge_color, linewidth=linewidth)
            if annotate and row['bytes'] > byte_threshold:
                chunk_center = stacked_value + row['bytes'] / 2
                plt.text(chunk_center, 'Combined', row['name'], ha='center', va='center', color='black', fontsize=font_size, rotation=90)
        
        stacked_value += row['bytes']

    if not minimal:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_requests))
        sm.set_array([])
        cbar = plt.colorbar(sm)  
        cbar.set_label('Requests Needed', rotation=270, labelpad=15)
        
        if orientation == 'horizontal':
            plt.ylabel('Bytes (unordered)')
            plt.xticks([])  
            plt.yticks([])
            plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
        else:
            plt.xlabel('Bytes (unordered)')
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
            print("Consider providing an output location with the --output argument.")


def main():
    parser = argparse.ArgumentParser(description="Analyze HDF5 files and produce plots.")
    
    # Required input file argument
    parser.add_argument("input_file", type=str, help="Path to the input HDF5 file.")
    
    # Optional output file argument with default behavior
    parser.add_argument("--output_file", type=str, default=None, help="Path to the output image file. If not provided, it defaults to '[input_filename]_xray.png'.")
    
    # Optional arguments
    parser.add_argument("--annotate", action='store_true', help="Whether to annotate on the plot or not. Default is False.")
    parser.add_argument("--font_size", type=int, default=5, help="Font size for annotations on the plot. Default is 5.")
    parser.add_argument("--byte_threshold", type=int, default=512*1024, help="Minimum bytes required for a dataset to get annotated. Default is 0.5 MB (512 * 1024 bytes).")
    parser.add_argument("--title", type=str, default=None, help="Title for the plot. If not specified, the name of the input file will be used.")
    parser.add_argument("--orientation", type=str, choices=['vertical', 'horizontal'], default='vertical', help="Orientation of the plot ('vertical' or 'horizontal'). Default is 'vertical'.")
    parser.add_argument("--figsize", type=lambda s: [int(item) for item in s.split(',')], default=[10,3], help="Size of the figure for the plot as width,height. Default is 10,3.")
    
    # Add the debug argument
    parser.add_argument("--debug", action='store_true', help="Provide a detailed plot for debugging. Default is a minimal plot.")
    
    args = parser.parse_args()
    
    # Determine the default output filename if not specified
    if args.output_file is None:
        base_name = os.path.splitext(args.input_file)[0]  # Get the name without extension
        args.output_file = base_name + "_xray.png"
        
    # Try to open the HDF5 file and handle potential errors
    try:
        with h5py.File(args.input_file, "r") as hdf5_file:
            file_info = extract_dataset_info(hdf5_file)
            df = pd.DataFrame(file_info)
    except OSError as e:
        print(f"Error reading the HDF5 file: {e}")
        exit(1)
    
    # Set title based on input if not provided
    if args.title is None:
        args.title = os.path.basename(args.input_file)
    
    # Infer the file format from the output file path if not provided
    file_format = os.path.splitext(args.output_file)[1].replace(".", "")
    
    plot_dataframe(df, 
                   annotate=args.annotate, 
                   font_size=args.font_size, 
                   byte_threshold=args.byte_threshold, 
                   title=args.title, 
                   orientation=args.orientation, 
                   figsize=args.figsize,
                   output_file=args.output_file,
                   minimal=not args.debug)  # Set the minimal parameter based on the debug argument


if __name__ == "__main__":
    main()