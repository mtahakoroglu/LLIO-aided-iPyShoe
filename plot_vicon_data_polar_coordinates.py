import numpy as np
import matplotlib.pyplot as plt
from ins_tools.util import *
import ins_tools.visualize as visualize
from ins_tools.INS import INS
import os
import logging
import glob
import scipy.io as sio
from scipy.signal import medfilt

# Directory containing your Vicon data files
vicon_data_dir = 'data/vicon/processed/'
vicon_data_files = glob.glob(os.path.join(vicon_data_dir, '*.mat'))

# Set up logging
output_dir = "results/figs/vicon_polar_coordinates/"
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, 'output.log')
logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# Detector thresholds
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

# Function to calculate displacement and heading change between stride points
def calculate_displacement_and_heading(gt, strideIndex):
    displacements = []
    heading_changes = []
    for j in range(1, len(strideIndex)):
        delta_position = gt[strideIndex[j], :2] - gt[strideIndex[j - 1], :2]
        displacement = np.linalg.norm(delta_position)
        heading_change = np.arctan2(delta_position[1], delta_position[0])
        displacements.append(displacement)
        heading_changes.append(heading_change)
    return np.array(displacements), np.array(heading_changes)


# Function to reconstruct trajectory from displacements and heading changes
def reconstruct_trajectory(displacements, heading_changes, initial_position):
    trajectory = [initial_position]
    current_heading = 0.0

    for i in range(len(displacements)):
        delta_position = np.array([
            displacements[i] * np.cos(heading_changes[i]),
            displacements[i] * np.sin(heading_changes[i])
        ])
        new_position = trajectory[-1] + delta_position
        trajectory.append(new_position)
        current_heading += heading_changes[i]

    return np.array(trajectory)


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

    # Apply median filter to zero velocity detection
    logging.info(f"Applying heuristic filter to {detector[i]} zero velocity detection")
    k = 65 # temporal window size for checking if detected strides are too close or not
    print(f"zv size: {zv.shape}")
    print(f"zv content: {zv}")
    zv_filtered, n, strideIndex = heuristic_zv_filter_and_stride_detector(zv, k)
    logging.info(f"Detected {n} strides in the data.")

    # Calculate displacement and heading changes between stride points based on ground truth
    displacements, heading_changes = calculate_displacement_and_heading(gt[:, :2], strideIndex)

    # Reconstruct the trajectory from displacements and heading changes
    initial_position = gt[0, :2]  # Starting point from the GT trajectory
    reconstructed_traj = reconstruct_trajectory(displacements, heading_changes, initial_position)

    # Remove the '.mat' extension from the filename
    base_filename = os.path.splitext(os.path.basename(file))[0]

    # Plotting the reconstructed trajectory and the ground truth without stride indices
    plt.figure()
    visualize.plot_topdown([reconstructed_traj, gt[:, :2]], title=base_filename,
                           legend=['Stride & Heading', 'GT (sample-wise)'])
    plt.scatter(-reconstructed_traj[:, 0], reconstructed_traj[:, 1], c='b', marker='x')
    plt.grid(True, which='both', linestyle='--', linewidth=1.5)
    plt.savefig(os.path.join(output_dir, f'stride_and_heading_{base_filename}.png'), dpi=600, bbox_inches='tight')

    plt.figure()
    plt.plot(timestamps[:len(gt)], gt[:, 2], label='GT (sample-wise)')  # Plot GT Z positions
    plt.plot(timestamps[:len(reconstructed_traj)], reconstructed_traj[:, 1],
             label='Stride & Heading')  # Plot reconstructed Z positions (use Y axis for visualization)
    plt.title(f'Vertical Trajectories - {base_filename}')
    plt.grid(True, which='both', linestyle='--', linewidth=1.5)
    plt.xlabel('Time')
    plt.ylabel('Z Position')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'vicon_data_vertical_lstm_{base_filename}.png'), dpi=600, bbox_inches='tight')

    # Plotting the zero velocity detection for median filtered data without stride indices
    plt.figure()
    plt.plot(timestamps[:len(zv)], zv, label='Original')
    plt.plot(timestamps[:len(zv_filtered)], zv_filtered, label='Heuristically Filtered')
    plt.scatter(timestamps[strideIndex], zv_filtered[strideIndex], c='r', marker='x')
    plt.title(f'Zero Velocity Detection - {base_filename}')
    plt.xlabel('Time')
    plt.ylabel('Zero Velocity')
    plt.grid(True, which='both', linestyle='--', linewidth=1.5)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'vicon_data_zv_optimal_{base_filename}.png'), dpi=600, bbox_inches='tight')

    i += 1  # Move to the next experiment

logging.info("Processing complete for all files.")
