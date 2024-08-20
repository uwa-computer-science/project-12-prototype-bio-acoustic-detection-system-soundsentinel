"""
This script is designed to visualize MFCC features for randomly selected segments of audio data corresponding to specific labels (classes of meerkat). 

Input:
- all_mfcc_features.npy: A numpy file containing the MFCC features extracted from audio segments. 
  - Shape: (n_mfcc, n_time_frames), where n_mfcc represents the number of MFCC features, and n_time_frames represents the total number of time frames.
  
- all_labels.npy: A numpy file containing the labels for each time frame.
  - Shape: (n_time_frames,), where each element is an integer representing the label (event) associated with that time frame.

Output:
- A visualization (heatmap) of the MFCC features centered around the selected event. 
  - The visualization shows the MFCC feature dimensions on the y-axis and time frames on the x-axis.
  - A red dashed line highlights the center of the event being visualized.

Key Parameters:
- The script uses predefined time frame ranges for each label to ensure the visualizations appropriately capture the duration of events with varying lengths.
"""


import numpy as np
import matplotlib.pyplot as plt
import random

# Load preprocessed MFCC features and labels data
all_mfcc_features = np.load('all_mfcc_features.npy')
all_labels = np.load('all_labels.npy')

def visualize_random_segment(mfcc_features, labels, target_label):
    """
    Visualize a randomly selected MFCC segment with the specified target label.
    """
    # Define the time frame range based on the label
    # Time frame range is calculated based on the average duration of each label. Shorter events get fewer frames to capture their features, while longer events get more frames to ensure full coverage.
    time_ranges = {
        1: 2,  
        2: 3,  
        3: 7,  
        4: 5   
    }

    # Find all indices where the label matches the target_label
    target_indices = np.where(labels == target_label)[0]

    if len(target_indices) > 0:
        # Randomly select one of the target label positions
        idx = random.choice(target_indices)

        # Determine the time frame range around the selected index
        range_size = time_ranges.get(target_label, 3)  # Default to 3 if label is not in the dict
        start_frame = max(0, idx - range_size)  # Take range_size frames before the current index
        end_frame = min(mfcc_features.shape[1], idx + range_size)  # Take range_size frames after the current index

        # Check the shape of mfcc_features and selected data
        print(f"MFCC features shape: {mfcc_features.shape}")
        print(f"Selected range shape: {mfcc_features[:, start_frame:end_frame].shape}")
        print(f"Start frame: {start_frame}, End frame: {end_frame}")

        # Ensure the selected range is not empty and is 2D
        if mfcc_features[:, start_frame:end_frame].size == 0 or len(mfcc_features[:, start_frame:end_frame].shape) != 2:
            print("Error: Selected MFCC data is empty or not 2D.")
            return
        
        # Visualize the MFCC features for the selected range
        plt.figure(figsize=(10, 4))
        plt.imshow(mfcc_features[:, start_frame:end_frame], aspect='auto', interpolation='none')
        plt.title(f'MFCC Features for label {target_label} around frame {idx}')
        plt.xlabel('Time Frame')
        plt.ylabel('MFCC Feature Dimension')
        plt.colorbar(label='Feature Value')

        # Highlight the central frame
        plt.axvline(x=range_size, color='red', linestyle='--', label='Event Center')
        plt.legend()

        plt.show()
    else:
        print(f"No labels found for target label {target_label}")

# Example: Visualize a segment with label 2
visualize_random_segment(all_mfcc_features, all_labels, target_label=1)
visualize_random_segment(all_mfcc_features, all_labels, target_label=2)
visualize_random_segment(all_mfcc_features, all_labels, target_label=3)
visualize_random_segment(all_mfcc_features, all_labels, target_label=4)

