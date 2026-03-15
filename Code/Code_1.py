"""
Code_1: Voxel-Based Heterogeneous Ballast Generation Script

This script generates 147 independent voxelized ballast sub-blocks
using procedurally placed 3D stone geometries with stochastic rigid-body
transformations and overlap-controlled assembly.

Author: Sepideh Harajchi
Affiliation: Delft University of Technology, Department of Geoscience and Engineering

Associated manuscript:
"Numerical Investigation of GPR Performance Over Voxelized Heterogeneous Railway Ballast:
Influence of Layer Composition and Antenna Frequency"
"""

import os
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from scipy.spatial import Delaunay
import pickle
import matplotlib.pyplot as plt
import h5py

# Base repository directory
base_address = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input and output directories
input_dir = os.path.join(base_address, "Input")
output_dir = os.path.join(base_address, "Output")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Function to load vertices from an OBJ file
def load_obj(filename):
    vertices = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertices.append([float(coord) for coord in line.strip().split()[1:4]])
    print(f"Loaded {len(vertices)} vertices from {filename}")
    return np.array(vertices)


# Function to create a voxel grid based on the box size
def create_voxel_grid(box_min, box_max, voxel_size):
    box_size = box_max - box_min
    grid_shape = np.ceil(box_size / voxel_size).astype(int)
    grid = np.zeros(grid_shape, dtype=bool)
    return grid


# Function to voxelize the stone
def voxelize(stone_vertices, grid, box_min, voxel_size):
    tri = Delaunay(stone_vertices)
    grid_shape = grid.shape
    filled_voxels = 0

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            for k in range(grid_shape[2]):
                point = box_min + np.array([i, j, k]) * np.array(voxel_size) + np.array(voxel_size) / 2
                if tri.find_simplex(point) >= 0:
                    grid[i, j, k] = True
                    filled_voxels += 1

    return grid


# Function to check overlap before placing the stone
def calculate_overlap(stone_voxels, global_voxel_grid):
    overlap_count = np.sum(np.logical_and(stone_voxels, global_voxel_grid))
    total_voxels = np.sum(stone_voxels)
    overlap_fraction = overlap_count / total_voxels if total_voxels > 0 else 0
    return overlap_count, overlap_fraction


# Function to apply a small random rotation with additional precision
def apply_precise_random_rotation(vertices):
    base_angles = np.random.uniform(-5, 5, 3)
    small_decimals = np.random.uniform(0, 1, 3)
    small_decimals = np.round(small_decimals, 3)
    precise_angles = base_angles + small_decimals
    rotation = R.from_euler('xyz', precise_angles, degrees=True)
    rotated_vertices = rotation.apply(vertices)
    return rotated_vertices


# Function to calculate the filled fraction for a specific Z-level
def calculate_filled_fraction_per_z_level(global_grid, z_min_idx, z_max_idx):
    z_level_slice = global_grid[:, :, z_min_idx:z_max_idx]
    total_voxels_in_z_level = np.prod(z_level_slice.shape)
    filled_voxels_in_z_level = np.sum(z_level_slice)

    filled_fraction = filled_voxels_in_z_level / total_voxels_in_z_level
    print(f"Current filled fraction for Z-level ({z_min_idx}, {z_max_idx}): {filled_fraction:.4f}")

    return filled_fraction


# Function to visualize 2D projection
def visualize_hdf5_data_2d(data, output_png_path, voxel_size, box_min):
    projection = np.sum(data != -1, axis=2)
    x_extent = np.array(data.shape[0]) * voxel_size[0]
    y_extent = np.array(data.shape[1]) * voxel_size[1]

    fig, ax = plt.subplots()
    cax = ax.imshow(
        projection.T,
        cmap='viridis',
        extent=[0, x_extent, 0, y_extent],
        origin='lower',
        aspect='auto'
    )
    fig.colorbar(cax, ax=ax, label='Occupied Voxel Count')
    ax.set_title('2D Projection of 3D Box')
    ax.set_xlabel('X-axis (meters)')
    ax.set_ylabel('Y-axis (meters)')
    plt.savefig(output_png_path)
    plt.close(fig)


# Function to visualize 3D voxel data with colors
def visualize_3d_voxel_data_colored(voxel_data, voxel_size, box_min, box_max, output_png_path, all_stones):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for idx, (stone_grid, color) in enumerate(all_stones):
        filled_voxels = np.argwhere(stone_grid)
        if filled_voxels.shape[1] != 3:
            print(f"Error: filled_voxels has shape {filled_voxels.shape}, expected shape (n, 3). Skipping this stone.")
            continue
        ax.scatter(
            filled_voxels[:, 0] * voxel_size[0] + box_min[0],
            filled_voxels[:, 1] * voxel_size[1] + box_min[1],
            filled_voxels[:, 2] * voxel_size[2] + box_min[2],
            c=color,
            marker='o',
            s=1
        )

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_xlim([0, box_max[0]])
    ax.set_ylim([0, box_max[1]])
    ax.set_zlim([0, box_max[2]])
    plt.savefig(output_png_path)
    plt.close(fig)


# Systematic placement with varied random rotations and overlap tracking
def systematic_z_level_placement_with_colors(
    all_vertices,
    all_colors,
    global_voxel_grid,
    voxel_size,
    box_min,
    box_max,
    grid_size_x,
    grid_size_y,
    x_spacing,
    y_spacing,
    z_ranges,
    fill_threshold=0.95,
    max_stones_per_level=25,
    iteration_num=1
):
    total_stones_placed = 0
    all_stones = []
    voxel_grids = []
    overlap_data = []

    for z_idx, (z_min, z_max) in enumerate(z_ranges):
        z_min_idx = int(z_min / voxel_size[2])
        z_max_idx = int(z_max / voxel_size[2])
        stones_placed_in_level = 0

        current_z_level_voxel_grids = []
        overlap_list = []

        for x_idx in range(grid_size_x):
            for y_idx in range(grid_size_y):
                if stones_placed_in_level >= max_stones_per_level:
                    print(f"Reached max stones per level: {max_stones_per_level}. Moving to next level.")
                    break

                stone_index = random.randint(0, len(all_vertices) - 1)
                vertices = all_vertices[stone_index]
                color = all_colors[stone_index]

                rotated_vertices = apply_precise_random_rotation(vertices)
                stone_min = np.min(rotated_vertices, axis=0)

                translation = np.array([x_idx * x_spacing, y_idx * y_spacing, z_min - stone_min[2]])
                transformed_vertices = rotated_vertices + translation

                print(f"Trying to place stone {total_stones_placed + 1} at X: {x_idx}, Y: {y_idx}, Z: {z_min}")

                stone_grid = create_voxel_grid(box_min, box_max, voxel_size)
                voxel_grid = voxelize(transformed_vertices, stone_grid, box_min, voxel_size)

                if voxel_grid.ndim != 3:
                    print(f"Error: voxel_grid has shape {voxel_grid.shape} instead of expected 3D shape.")
                    continue

                overlap_count, overlap_fraction = calculate_overlap(voxel_grid, global_voxel_grid)

                if overlap_fraction > 0.3:
                    print(f"Overlap too high for stone at X: {x_idx}, Y: {y_idx}, Z: {z_min}. Skipping this placement.")
                    continue

                filled_voxels_count = np.sum(voxel_grid)
                if filled_voxels_count > 0:
                    global_voxel_grid = np.logical_or(global_voxel_grid, voxel_grid)
                    all_stones.append((voxel_grid, color))
                    voxel_grids.append(voxel_grid)
                    current_z_level_voxel_grids.append(voxel_grid)
                    overlap_list.append(overlap_count)
                    total_stones_placed += 1
                    stones_placed_in_level += 1
                    print(
                        f"Placed stone {total_stones_placed} at Z level {z_min}. "
                        f"Stones in this level: {stones_placed_in_level}, "
                        f"Filled voxels: {filled_voxels_count}, Overlap: {overlap_count}"
                    )
                else:
                    print(f"Warning: No voxels filled for this stone at X: {x_idx}, Y: {y_idx}, Z: {z_min}. Skipping this placement.")

        max_overlap = max(overlap_list) if overlap_list else 0
        min_overlap = min(overlap_list) if overlap_list else 0
        overlap_data.append((max_overlap, min_overlap))
        print(f"Z-level {z_min} - {z_max}: Max Overlap: {max_overlap}, Min Overlap: {min_overlap}")

        current_filled_fraction = calculate_filled_fraction_per_z_level(global_voxel_grid, z_min_idx, z_max_idx)
        print(f"Current filled fraction for Z-level ({z_min}, {z_max}): {current_filled_fraction:.4f}")

        z_level_data = np.full(global_voxel_grid.shape, -1, dtype=np.int16)
        z_level_data[global_voxel_grid] = 0

        z_level_pkl_file = os.path.join(output_dir, f'voxel_grid_iteration_{iteration_num}_z_{z_idx}.pkl')
        with open(z_level_pkl_file, 'wb') as f:
            pickle.dump(current_z_level_voxel_grids, f)
        print(f"Saved voxel grid for Z-level {z_idx} to {z_level_pkl_file}")

        output_2d_png = os.path.join(output_dir, f'systematic_voxel_projection_iteration_{iteration_num}_z_{z_idx}.png')
        output_3d_png = os.path.join(output_dir, f'systematic_voxel_3d_iteration_{iteration_num}_z_{z_idx}.png')
        visualize_hdf5_data_2d(z_level_data, output_2d_png, voxel_size, box_min)
        visualize_3d_voxel_data_colored(current_z_level_voxel_grids, voxel_size, box_min, box_max, output_3d_png, all_stones)

    return global_voxel_grid, all_stones, voxel_grids, overlap_data


# Function to create final HDF5 file
def create_hdf5_file(file_path, box_min, box_max, voxel_size, material_indices, voxel_data):
    box_size = box_max - box_min
    data_shape = np.ceil(box_size / voxel_size).astype(int)

    with h5py.File(file_path, 'w') as f:
        f.attrs['dx_dy_dz'] = voxel_size
        f.attrs['box_size'] = box_size

        data = np.full(data_shape, -1, dtype=np.int16)
        print(f"Initialized HDF5 data with shape: {data.shape} and default value -1")

        for stone_index, stone_grid in enumerate(voxel_data):
            filled_voxels = np.argwhere(stone_grid)
            print(f"Stone {stone_index + 1} has {len(filled_voxels)} filled voxels.")
            for voxel in filled_voxels:
                ix, iy, iz = voxel
                data[ix, iy, iz] = material_indices["stone"]

        f.create_dataset("data", data=data, dtype=np.int16)

        print(f"HDF5 file created at {file_path}")
        return data


# Main execution with iterations
def main(number_of_iterations=147):
    voxel_size = (0.002, 0.002, 0.002)
    box_min = np.array([0, 0, 0])
    box_max = np.array([0.1, 0.1, 0.1])
    fill_threshold = 0.95
    max_stones_per_level = 25
    grid_size_x, grid_size_y = 5, 5
    x_spacing, y_spacing = 0.022, 0.022
    z_ranges = [(0, 0.025), (0.025, 0.05), (0.05, 0.075), (0.075, 0.1)]

    obj_file_paths = [os.path.join(input_dir, f'D{i}.obj') for i in range(1, 11)]
    all_vertices = [load_obj(path) for path in obj_file_paths]

    obj_colors = [
        '#d9d9d9', '#ffcc99', '#99ff99', '#66ccff', '#ff9999',
        '#9999ff', '#ffff99', '#c2c2f0', '#ffb3e6', '#ffccff'
    ]
    all_colors = {i: obj_colors[i] for i in range(len(obj_file_paths))}

    final_report_path = os.path.join(output_dir, 'final_report.txt')

    with open(final_report_path, 'w') as report_file:
        report_file.write("Iteration Report\n")
        report_file.write("================\n\n")

    for iteration in range(1, number_of_iterations + 1):
        global_voxel_grid = create_voxel_grid(box_min, box_max, voxel_size)
        global_voxel_grid, all_stones_systematic, voxel_grids_systematic, overlap_data = systematic_z_level_placement_with_colors(
            all_vertices,
            all_colors,
            global_voxel_grid,
            voxel_size,
            box_min,
            box_max,
            grid_size_x,
            grid_size_y,
            x_spacing,
            y_spacing,
            z_ranges,
            fill_threshold,
            max_stones_per_level,
            iteration
        )

        final_pkl_file = os.path.join(output_dir, f'filled_box_systematic_with_colors_{iteration}.pkl')
        with open(final_pkl_file, 'wb') as f:
            pickle.dump(voxel_grids_systematic, f)
        print(f"Saved final voxel grid to {final_pkl_file}")

        hdf5_file_path = os.path.join(output_dir, f'systematic_voxel_data_with_colors_{iteration}.h5')
        data = create_hdf5_file(hdf5_file_path, box_min, box_max, voxel_size, {"stone": 0}, voxel_grids_systematic)

        with open(final_report_path, 'a') as report_file:
            report_file.write(f"Iteration {iteration}\n")
            report_file.write("----------\n")
            for z_idx, (max_overlap, min_overlap) in enumerate(overlap_data):
                filled_fraction = calculate_filled_fraction_per_z_level(
                    global_voxel_grid,
                    int(z_ranges[z_idx][0] / voxel_size[2]),
                    int(z_ranges[z_idx][1] / voxel_size[2])
                )
                report_file.write(
                    f"Z-level {z_ranges[z_idx][0]} - {z_ranges[z_idx][1]}: "
                    f"Max Overlap: {max_overlap}, Min Overlap: {min_overlap}, "
                    f"Filled Fraction: {filled_fraction:.4f}\n"
                )
            report_file.write("\n")


if __name__ == "__main__":
    main(number_of_iterations=147)
