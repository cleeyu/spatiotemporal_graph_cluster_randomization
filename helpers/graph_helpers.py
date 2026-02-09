import numpy as np
from scipy.spatial.distance import cdist
from typing import Tuple, Dict
from . import utils

# ======================================
# GENERATE LATTICE GRAPHS AND CLUSTERS
# ======================================

def lattice_neighbor(x,y,sqrt_n,kappa):
    return (np.abs(x % sqrt_n - y % sqrt_n) + np.abs(x // sqrt_n - y / sqrt_n) <= kappa).astype(int)
    
def generate_interference_graph_from_lattice(
        sqrt_n: int, 
        kappa: float, 
        include_self_loops: bool=True
        ) -> np.ndarray:

    adj_matrix = np.fromfunction(lambda x,y: lattice_neighbor(x,y,sqrt_n,kappa), (sqrt_n**2, sqrt_n**2), dtype=int)
    if not include_self_loops:
        adj_matrix -= np.eye(sqrt_n**2, dtype=int)

    return adj_matrix

def cluster_membership(x,sqrt_n,square_width,sqrt_num_clusters):
    return ((x // sqrt_n) // square_width) * sqrt_num_clusters + ((x % sqrt_n) // square_width)

def generate_clusters_from_lattice(
        sqrt_n: int, 
        num_cells_per_dim: int, 
        ) -> np.ndarray:
    
    square_width = int(np.ceil(sqrt_n/num_cells_per_dim))

    cluster_map = np.fromfunction(lambda x: cluster_membership(x,sqrt_n,square_width,num_cells_per_dim), (sqrt_n**2,), dtype=int)
    cluster_matrix = np.zeros((sqrt_n**2, num_cells_per_dim**2))
    cluster_matrix[np.arange(sqrt_n**2), cluster_map] = 1
    return cluster_matrix

# =================== FUNCTIONS TO GENERATE GENERAL SPATIAL GRAPHS AND CLUSTERS =========================

def generate_linear_chain_adj_map(n: int) -> np.ndarray:
    """ Generate adjacency matrix for a linear chain graph with self-loops. """
    return np.tril(np.ones((n, n)), k=1) - np.tril(np.ones((n, n)), k=-1)

def generate_random_points(num_pts: int, seed: int = 42) -> np.ndarray:
    """
    Generate n random pts in [0,1]^2
    Parameters:
    - num_pts: int, number of individuals
    - seed: int, random seed for reproducibility
    Returns:
    - coords_array: np.ndarray (n x 2), array of (x,y) coordinates
    """
    np.random.seed(seed)
    coords_array = np.random.uniform(0, 1, size=(num_pts, 2))
    return coords_array

def generate_lattice_points(sqrt_n: int) -> np.ndarray:
    """
    Generate lattice grid pts in [0,1]^2
    Parameters:
    - sqrt_n: int, grid dimension (total n = sqrt_n^2)
    Returns:
    - coords_array: np.ndarray (n x 2), array of (x,y) coordinates on lattice
    """
    n = sqrt_n**2
    coords_array = np.zeros((n, 2))
    grid_points = np.linspace(0, 1, sqrt_n) # Create evenly spaced grid in [0,1]
    coords_array[:,0] = np.flatten(np.outer(grid_points,np.ones(sqrt_n)))
    coords_array[:,1] = np.flatten(np.outer(np.ones(sqrt_n),grid_points))
    return coords_array

### ---

def build_adjacency_matrix_from_coords(
        coords_array: np.ndarray, 
        kappa: float
        ) -> np.ndarray:
    """ Build adjacency matrix based on distance threshold (WITH self-loops) """
    distance_matrix = cdist(coords_array, coords_array, metric='euclidean')

    return distance_matrix <= kappa

def spatial_clustering_map(
        coords_array: np.ndarray,
        num_cells_per_dim: int
        ) -> np.ndarray:
    """
    Partition space into square cells and map each node to its cell
    Parameters:
    - coords_array: np.ndarray (n x 2), node coordinates in [0,1]^2
    - num_cells_per_dim: int, number of cells per dimension
    Returns:
    - cluster_matrix: np.ndarray
    """
    n = coords_array.shape[0]
    cell_width = 1.0 / num_cells_per_dim

    x_coord = coords_array[:, 0]
    y_coord = coords_array[:, 1]

    cluster_map = (np.floor(x_coord / cell_width) * num_cells_per_dim + np.floor(y_coord / cell_width)).astype(int)
    cluster_map[cluster_map == num_cells_per_dim] = num_cells_per_dim - 1  # Handle boundary case where coord = 1.0

    cluster_matrix = np.zeros((n, num_cells_per_dim**2))
    cluster_matrix[np.arange(n), cluster_map] = 1
    return cluster_matrix