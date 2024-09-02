import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ins_tools.util import *
import ins_tools.visualize as visualize
from ins_tools.INS import INS
import os
import logging
import glob
import scipy.io as sio  # Import scipy.io for loading .mat files
from scipy.signal import medfilt  # Import median filter

# Directory containing your Vicon data files
vicon_data_dir = 'data/vicon/processed/'
vicon_data_files = glob.glob(os.path.join(vicon_data_dir, '*.mat'))

# Set up logging
output_dir = "results/figs/vicon/"
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, 'output.log')
logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# Optimal Detectors are obtained by running data\vicon\processed\extract_optimal_detectors.m 
detector = ['shoe', 'ared', 'shoe', 'shoe', 'shoe', 'ared', 'shoe', 'shoe',
            'vicon', 'shoe', 'shoe', 'vicon', 'vicon', 'shoe', 'vicon', 'mbgtd',
            'shoe', 'shoe', 'ared', 'vicon', 'shoe', 'shoe', 'vicon', 'shoe',
            'vicon', 'shoe', 'shoe', 'shoe', 'vicon', 'vicon', 'vicon', 'shoe',
            'shoe', 'vicon', 'vicon', 'shoe', 'shoe', 'shoe', 'shoe', 'ared',
            'shoe', 'shoe', 'ared', 'shoe', 'shoe', 'shoe', 'ared', 'shoe',
            'shoe', 'ared', 'mbgtd', 'shoe', 'vicon', 'shoe', 'shoe', 'vicon']
thresh = [2750000, 0.1, 6250000, 15000000, 5500000, 0.08, 3000000, 3250000,
          0.02, 97500000, 20000000, 0.0825, 0.1, 30000000, 0.0625, 0.225,
          92500000, 9000000, 0.015, 0.05, 3250000, 4500000, 0.1, 100000000,
          0.0725, 100000000, 15000000, 250000000, 0.0875, 0.0825, 0.0925, 70000000,
          525000000, 0.4, 0.375, 150000000, 175000000, 70000000, 27500000, 1.1,
          12500000, 65000000, 0.725, 67500000, 300000000, 650000000, 1, 4250000,
          725000, 0.0175, 0.125, 42500000, 0.0675, 9750000, 3500000, 0.175]

# Function to align trajectory with ground truth
def align_trajectory(predicted, ground_truth):
    predicted_aligned, ground_truth_aligned = align_plots(predicted, ground_truth)
    return predicted_aligned, ground_truth_aligned

i = 0  # experiment index
# Process each Vicon data file
for file in vicon_data_files:
    logging.info(f"Processing file: {file}")
    data = sio.loadmat(file)

    # Extract the relevant columns
    imu_data = np.column_stack((data['imu'][:, :3], data['imu'][:, 3:6]))  # Accel and Gyro data
    timestamps = data['ts'][0]
    gt = data['gt']  # Ground truth from Vicon dataset

    # Initialize INS object with correct parameters
    ins = INS(imu_data, sigma_a=0.00098, sigma_w=8.7266463e-5, T=1.0 / 200)

    logging.info(f"Processing {detector[i]} detector for file: {file}")
    ins.Localizer.set_gt(gt)  # Set the ground truth data required by 'vicon' detector
    ins.Localizer.set_ts(timestamps)  # Set the sampling time required by 'vicon' detector
    zv = ins.Localizer.compute_zv_lrt(W=5 if detector[i] != 'mbgtd' else 2, G=thresh[i], detector=detector[i])
    x = ins.baseline(zv=zv)

    # Align the trajectory with the ground truth
    x_aligned, gt_aligned = align_trajectory(x, gt)

    # Apply median filter to zero velocity detection
    logging.info(f"Applying median filter to {detector[i]} zero velocity detection")
    kernel_size = 15  # Starting kernel size for median filter
    zv_filtered = medfilt(zv, kernel_size)
    zv_filtered[:100] = 1  # Ensure all labels are zero at the beginning as the foot is stationary

    # Automatically detect the stride indices using one-to-zero transitions
    n, strideIndex = count_one_to_zero_transitions(zv_filtered)
    strideIndex = strideIndex - 1 # make all stride indexes the last samples of the respective ZUPT phase
    strideIndex[0] = 0 # first sample is the first stride index
    strideIndex = np.append(strideIndex, len(timestamps)-1) # last sample is the last stride index
    logging.info(f"Detected {n} strides in the data.")

    # Remove the '.mat' extension from the filename
    base_filename = os.path.splitext(os.path.basename(file))[0]
    # Plotting the trajectory and the ground truth with stride indices marked
    plt.figure()
    visualize.plot_topdown([x_aligned, gt_aligned], title=f'{os.path.basename(base_filename)}',
                           legend=[detector[i], 'Ground Truth'])
    plt.scatter(-x_aligned[strideIndex, 0], x_aligned[strideIndex, 1], c='r',
                marker='x')  # Mark the stride points on the trajectory
    plt.savefig(os.path.join(output_dir, f'vicon_data_trajectories_optimal_{os.path.basename(base_filename)}.png'), bbox_inches='tight')

    plt.figure()
    plt.plot(timestamps[:len(x_aligned)], x_aligned[:, 2], label=detector[i])  # Plot INS Z positions
    plt.plot(timestamps[:len(gt_aligned)], gt_aligned[:, 2], label='Ground Truth')  # Plot GT Z positions
    plt.title(f'Vertical Trajectories - {os.path.basename(file)}')
    plt.xlabel('Time')
    plt.ylabel('Z Position')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'vicon_data_vertical_lstm_{os.path.basename(base_filename)}.png'), bbox_inches='tight')

    # Plotting the zero velocity detection for median filtered data with stride indices marked
    plt.figure()
    plt.plot(timestamps[:len(zv)], zv, label='Original')
    plt.plot(timestamps[:len(zv_filtered)], zv_filtered, label='Median Filtered')
    plt.scatter(timestamps[strideIndex], zv_filtered[strideIndex], c='r', marker='x')  # Mark the stride points
    plt.title(f'Zero Velocity Detection - {os.path.basename(base_filename)}')
    plt.xlabel('Time')
    plt.ylabel('Zero Velocity')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'vicon_data_zv_optimal_{os.path.basename(base_filename)}.png'), bbox_inches='tight')

    i += 1  # Move to the next experiment

logging.info("Processing complete for all files.")
