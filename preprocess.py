# this file of code load orinigal wav file, extract audio features (MFCC) using librosa library ,and assigns corresponding labels. 
# outputs are mfcc_features_tensor and labels_tensor saved in dataloader.
# label: 0: No meerkat detected; 1: Detected SNMK; 2: Detected CCMK; 3: Detected AGGM; 4: Detected SOCM.  
#events: {0: 68026, 1: 78, 2: 2249, 3: 341, 4: 170}

#parameters:
#n_mfcc=30 Increasing n_mfcc will extract more spectral features
#hop_length=512 Reducing hop_length increases time resolution, easier to capture short events
#target_sr=8000 Target Sampling Rate, Increasing target_sr will increase the number of frames and improve time resolution, helping the model capture short events
#batch_size=32 Increasing batch_size can speed up the training process 
#each frame covers 512 / 8000 = 0.064 seconds.



import pandas as pd
import librosa
import numpy as np
from scipy.signal import resample

# load file
audio_files = ['/Users/xiaoyuliu/Documents/school/capstone/MT/dcase_MK1.wav', '/Users/xiaoyuliu/Documents/school/capstone/MT/dcase_MK2.wav']  
csv_files = ['/Users/xiaoyuliu/Documents/school/capstone/MT/dcase_MK1.csv', '/Users/xiaoyuliu/Documents/school/capstone/MT/dcase_MK2.csv'] 

all_mfcc_features = []
all_labels = []


# Loop through each audio file and its corresponding CSV file
for audio_path, csv_path in zip(audio_files, csv_files):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Load the CSV file with annotations
    annotations = pd.read_csv(csv_path)

    # Extract MFCC features from the audio file
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
    mfcc_normalized = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / np.std(mfcc, axis=1, keepdims=True)

    # Initialize an array to hold the labels for each time frame
    labels = np.zeros(mfcc.shape[1])  # Number of time frames

    
    # Iterate through the annotations to label each frame based on the annotations
    for _, row in annotations.iterrows():
        start_time = row['Starttime']
        end_time = row['Endtime']

        # Convert times to sample indices
        start_frame = int(start_time * sr // 512)  # 512 is the default hop_length in librosa.feature.mfcc
        end_frame = int(end_time * sr // 512)

        # Determine which class is positive

        if row['SNMK'] == 'POS':
            label = 1
        elif row['CCMK'] == 'POS':
            label = 2
        elif row['AGGM'] == 'POS':
            label = 3
        elif row['SOCM'] == 'POS':
            label = 4
        else:
            label = 0  # No Meerkat detected

        # Label the corresponding time frames
        labels[start_frame:end_frame] = label



    # Resample MFCC and labels if needed to match SNN input requirements
    target_sr = 8000  # Target sampling rate for SNN
    num_samples = int(mfcc_normalized.shape[1] * target_sr / sr)

    # Resample the MFCC features and labels
    mfcc_resampled = resample(mfcc_normalized, num_samples, axis=1)
    labels_resampled = resample(labels, num_samples)

    # Ensure labels are integers after resampling
    labels_resampled = np.round(labels_resampled).astype(int)

    # Append the resampled features and labels to the lists
    all_mfcc_features.append(mfcc_resampled)
    all_labels.append(labels_resampled)



# concatenate all features and labels into single arrays
all_mfcc_features = np.concatenate(all_mfcc_features, axis=1)
all_labels = np.concatenate(all_labels)

# Save the processed data 
np.save('all_mfcc_features.npy', all_mfcc_features)
np.save('all_labels.npy', all_labels)

# check shape
unique, counts = np.unique(all_labels, return_counts=True)
event_counts = dict(zip(unique, counts))
#print(event_counts) #{0: 67976, 1: 8, 2: 2724, 4: 156}
#print(all_mfcc_features.shape) # (30, 70864)
#print( all_labels.shape) #(70864,)

# Transpose mfcc_resampled to make the first dimension correspond to the number of time frames
all_mfcc_features = all_mfcc_features.T 
#print(all_mfcc_features.shape) #（70864, 30）（time_frames， mfcc）

#create tensordataset and dataloader，To load the preprocessed audio features and labels in batches, so that they can be passed to SNN
import torch
from torch.utils.data import DataLoader, TensorDataset
mfcc_features_tensor = torch.tensor(all_mfcc_features, dtype=torch.float32)
labels_tensor = torch.tensor(all_labels, dtype=torch.long)
dataset = TensorDataset(mfcc_features_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_idx, (inputs, targets) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1} shape: {inputs.shape}") 
    print(f"Sample shape: {inputs[0].shape}")
    break


#Batch 1 shape: torch.Size([32, 30])
#Sample shape: torch.Size([30])