"""
This code processes audio files by segmenting them into 1-second clips based on annotated meerkat call events,
labeling them according to the type of event. The segmented clips are saved with filenames include label.
0: background noise; 1: SNMK; 2: CCMK; 3: AGGM; 4: SOCM.  7: unknown 
"""


import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

# File paths for the audio files and corresponding CSV annotations
audio_files = ['/Users/xiaoyuliu/Documents/school/capstone/MT/dcase_MK1.wav', '/Users/xiaoyuliu/Documents/school/capstone/MT/dcase_MK2.wav']
csv_files = ['/Users/xiaoyuliu/Documents/school/capstone/MT/dcase_MK1.csv', '/Users/xiaoyuliu/Documents/school/capstone/MT/dcase_MK2.csv']

# Output directory for AFE segments
output_dir = 'processed_segments'
os.makedirs(output_dir, exist_ok=True)

# Target sampling rate
sr = 8000

# Iterate through each audio file and its corresponding CSV annotation file
for audio_file, csv_file in zip(audio_files, csv_files):
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=sr)

    # Load the CSV annotations
    annotations = pd.read_csv(csv_file)

    # Iterate over each row in the CSV to extract annotation details
    for i, annotation in annotations.iterrows():
        start_time = annotation['Starttime']
        end_time = annotation['Endtime']

        # Calculate the event center time
        event_center = (start_time + end_time) / 2

        # Calculate the segment start and end times to ensure a 1-second duration
        segment_start = max(0, event_center - 0.5)  # Ensure the start time is not negative
        segment_end = segment_start + 1.0  # End time is 1 second after the start time

        # Extract the segment from the audio file
        start_sample = int(segment_start * sr)
        end_sample = int(segment_end * sr)
        segment = y[start_sample:end_sample]

        # If the segment is less than 1 second, pad it with zeros
        if len(segment) < sr:
            segment = np.pad(segment, (0, sr - len(segment)), 'constant')

        # Determine the label based on the annotation
        if annotation[3] == 'POS':
            label = 1  # SNMK
        elif annotation[4] == 'POS':
            label = 2  # CCMK
        elif annotation[5] == 'POS':
            label = 3  # AGGM
        elif annotation[6] == 'POS':
            label = 4  # SOCM
        else:
            label = 7  # unknown

        # Create the output filename based on the audio file name, label, and segment index
        output_filename = os.path.join(output_dir, f'{os.path.basename(audio_file).replace(".wav", "")}_segment_{i+1}_label_{label}.wav')

        # Save the segment as a WAV file using the soundfile library
        sf.write(output_filename, segment, sr)

    # Process background noise segments
    no_event_segments = []
    for i in range(len(y) // sr):
        segment_start = i * sr
        segment_end = segment_start + sr
        segment = y[segment_start:segment_end]
        
        # Check if the segment overlaps with any event
        overlaps = any(start_sample <= segment_start <= end_sample or start_sample <= segment_end <= end_sample
                       for start_sample, end_sample in zip(annotations['Starttime'] * sr, annotations['Endtime'] * sr))
        
        if not overlaps:
            no_event_segments.append(segment)

    # Randomly select the same number of no-event segments as there are event segments
    np.random.shuffle(no_event_segments)
    selected_no_event_segments = no_event_segments[:len(annotations)]

    # Save the no-event segments with label 0 (background noise)
    for i, segment in enumerate(selected_no_event_segments):
        output_filename = os.path.join(output_dir, f'segment_{os.path.basename(audio_file).replace(".wav", "")}_{i+1}_label_0.wav')
        sf.write(output_filename, segment, sr)
