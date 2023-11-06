import h5py
import igraph as ig
import os
import plotly.graph_objects as go
from PIL import Image, ImageDraw
import textwrap
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from PIL import Image, ImageDraw

def scale_to_range(arr, feature_range=(5, 20)):
    scaler = MinMaxScaler(feature_range=(0, 1))  # We start by scaling to the 0-1 range.
    scaled_data = scaler.fit_transform(arr.reshape(-1, 1)).flatten()  # Fit and transform the data.

    # Now scale to 5-20 range.
    min_val, max_val = feature_range
    scaled_data = scaled_data * (max_val - min_val) + min_val
    return scaled_data

class H5Tree:
    def __init__(self, filepath, max_depth=None, show_text=False, figure_size=(500, 800), export_path=None, group_path='/'):
        """
        Constructor for the H5XRay class.

        Args:
            filepath (str): Path to the HDF5 file to visualize.
            max_depth (int, optional): Maximum depth to visualize in the tree. If None, visualize entire tree.
            show_text (bool, optional): Whether to display text labels on nodes in the graph.
            figure_size (tuple, optional): Dimensions for the output visualization (width, height).
        """
        self.filepath = filepath
        self.G = None
        self.show_text = show_text
        self.figure_size = figure_size
        self.group_path = group_path
        self.generate_igraph_tree(max_depth=max_depth, group_path=group_path)


    def hdf5_to_igraph_tree(self, group, parent_name='/', depth=0, max_depth=None):
        """
        Converts an HDF5 group into an igraph tree representation. Used for visualization.

        Args:
            group (h5py.Group): The current HDF5 group.
            parent_name (str): The hierarchical name of the current group.
            depth (int): Current depth in the hierarchy.
            max_depth (int, optional): Maximum depth for visualization.
        """
        if self.G is None:
            self.G = ig.Graph(directed=True)
            self.G.add_vertex(name=parent_name, label=parent_name.split('/')[-1] or '/')

        if max_depth is not None and depth > max_depth:
            return

        for key, item in group.items():
            vertex_name = f"{parent_name}/{key}" if parent_name != '/' else f"{parent_name}{key}"

            attrs = {k: v for k, v in item.attrs.items()}  # Extract attributes
            dtype, shape, size = 'na', 'na', 'na'
            if isinstance(item, h5py.Dataset):
                size = item.size * item.dtype.itemsize
                dtype = str(item.dtype)
                shape = str(item.shape)

            self.G.add_vertex(name=vertex_name, label=key, depth=depth, size=str(size), dtype=dtype, shape=shape, attrs=attrs)
            self.G.add_edge(parent_name, vertex_name)

            if isinstance(item, h5py.Group):
                self.hdf5_to_igraph_tree(item, vertex_name, depth + 1, max_depth)


    def generate_igraph_tree(self, group_path='/', max_depth=None):
        """
        Generates the igraph tree representation of the HDF5 file.

        Args:
            group_path (str): The path to the group within the HDF5 file. Must start with '/'.
            max_depth (int, optional): Maximum depth for visualization.
        """

        assert isinstance(group_path, str) and group_path.startswith('/'), "group_path must be a string starting with '/'"

        if self.G is None:
            with h5py.File(self.filepath, 'r') as f:
                # Check if the group_path is a valid group in the HDF5 file
                if group_path not in f:
                    raise ValueError(f"{group_path} is not a valid group in the HDF5 file.")
                self.G = None
                self.hdf5_to_igraph_tree(f[group_path], max_depth=max_depth)

    def create_plotly_figure(self):
        """
        Create the Plotly figure for visualization.

        Returns:
            go.Figure: The Plotly figure object.
        """
        if self.G is None:
            raise ValueError("No graph data available. Please run generate_igraph_tree first.")

        layout = self.G.layout_reingold_tilford(root=[0])

        x = [coord[0] for coord in layout]
        y = [coord[1] for coord in layout]

        unique_depths = set(v['depth'] + 1 if v['depth'] is not None else 0 for v in self.G.vs)
        max_depth = max(unique_depths)
        categorical_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        color_palette = [categorical_colors[i % len(categorical_colors)] for i in range(max_depth + 1)]

        edge_trace = go.Scatter(
            x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

        for edge in self.G.es:
            source, target = edge.tuple
            x0, y0 = layout[source]
            x1, y1 = layout[target]
            edge_trace['y'] += tuple([x0, x1, None])
            edge_trace['x'] += tuple([y0, y1, None])
             
        size_arr = np.array([v['size'] for v in self.G.vs])

        # set value to numeric if possible, otherwise set to 0 (ie just a folder level)
        size_arr = np.array([int(s) if isinstance(s, str) and s.replace('.','',1).isdigit() else 0 for s in size_arr])
        
        # scale sizes to between 5 and 20
        size_arr = scale_to_range(size_arr, feature_range=(5, 20))
        
        hover_texts = []
        for v in self.G.vs:
            base_text = v['label']
            additional_info = []

            # Add size if dataset
            if v['size'] != 'na':
                additional_info.append(f"Size: {v['size']} bytes")

            # Add dtype and shape if dataset
            if v['dtype'] != 'na' and v['shape'] != 'na':
                additional_info.append(f"Type: {v['dtype']}")
                additional_info.append(f"Shape: {v['shape']}")

            # Add attributes (metadata) in a structured manner
            if v['attrs']:
                attr_texts = []
                for k, attr_value in v['attrs'].items():
                    # Check if the value is a byte string and decode if necessary
                    formatted_value = attr_value.decode() if isinstance(attr_value, bytes) else str(attr_value)
                    # Wrap long attribute values
                    wrapped_value = self.wrap_text(formatted_value)
                    attr_text = f"{k}: {wrapped_value}"
                    attr_texts.append(attr_text)

                additional_info.append("Attributes:")
                additional_info.extend(attr_texts)

            full_text = base_text + ("<br>" + "<br>".join(additional_info) if additional_info else "")
            hover_texts.append(full_text)

        node_trace = go.Scatter(
            x=y, y=x,
            mode='markers+text' if self.show_text else 'markers',
            text=[v['label'] for v in self.G.vs] if self.show_text else None,
            hovertext=hover_texts,
            hoverinfo='text',
            marker=dict(
                showscale=False,
                color=[color_palette[v['depth'] + 1 if v['depth'] is not None else 0] for v in self.G.vs],
                size=size_arr,
                opacity=0.6,
                line_width=0,
            ))
        
        
        
        if self.group_path == '/':
            title = f'h5xray-tree: {os.path.basename(self.filepath)}'
        else:
            title = f'h5xray-tree: {os.path.basename(self.filepath)} \n{self.group_path}'

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(title=title,
                                         showlegend=False, hovermode='closest',
                                         width=self.figure_size[0], height=self.figure_size[1],
                                         margin=dict(b=10, l=10, r=50, t=50),
                                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                         plot_bgcolor='white', paper_bgcolor='white'))
        

        return fig




    def explore(self):
        """
        Visualizes the tree structure of the HDF5 file using Plotly.

        Args:
            export_path (str, optional): The full path to the output image file, including filename and extension.
                                         If provided, the plot will be exported as an image.
        Raises:
            ValueError: If the graph data isn't available.
        """
        fig = self.create_plotly_figure()  # Create the Plotly figure

        return fig

    def wrap_text(self, text, max_len=50, indent="    "):  # using 4 spaces as indentation by default
        """
        Wrap the given text into lines of no more than max_len characters with indentation for wrapped lines.
        
        Args:
            text (str): The input text.
            max_len (int, optional): Maximum number of characters for each line. Default is 50.
            indent (str, optional): Characters to use for line indentation. Default is 4 spaces.
        
        Returns:
            str: The wrapped text with indentation.
        """
        lines = textwrap.wrap(text, max_len)
        # Only indent from the second line onwards
        indented_lines = [lines[0]] + [indent + line for line in lines[1:]]
        return "<br>".join(indented_lines)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Visualize HDF5 file structures.")
    parser.add_argument("filepath", type=str, help="Path to the HDF5 file to visualize.")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth to visualize in the tree.")
    parser.add_argument("--show_text", type=bool, default=True, help="Whether to display text labels on nodes in the graph.")
    parser.add_argument("--figure_size", type=int, nargs=2, default=(1000, 1000), help="Dimensions for the output visualization (width, height).")
    parser.add_argument("--export_path", type=str, default=None, help="The full path to the output image file, including filename and extension.")
    
    args = parser.parse_args()

    tree = H5Tree(filepath=args.filepath, 
                     max_depth=args.max_depth, 
                     show_text=args.show_text, 
                     figure_size=args.figure_size, 
                     export_path=args.export_path)
    