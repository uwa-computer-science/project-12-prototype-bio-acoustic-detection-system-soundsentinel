# Bio-Acoustic Detection System


#### Project Objectives

This project focuses on creating a bio-acoustic event detection system using the Xylo Audio processor to monitor biological pests in outdoor industrial settings, such as solar farms. 

Once trained, the model is simulated and deployed on the Xylo hardware for real-world performance evaluation. The project also includes clear data visualizations, uses Weights & Biases for performance tracking, and leverages GitHub for version control to ensure reproducibility.

#### Data Description

We used the DCASE2024 Meerkat vocalization dataset, which includes 70 minutes of audio data. The dataset features four vocalization types:
- **SNMK** (95 instances)
- **CCMK** (1046 instances)
- **AGGM** (53 instances)
- **SOCM** (40 instances)
- 24 unclassified vocalizations (UNK)

These recordings were made with GPS/audio collars at the Kalahari Meerkat Project, providing a complex acoustic environment for training a Spiking Neural Network (SNN).

#### Approach

We adopted a collaborative approach to tackle challenges such as unfamiliarity with audio data and SNNs. The project focuses on binary classification due to class imbalance in the dataset, with an emphasis on distinguishing major vocalization classes from background noise. We addressed the dataset's low sampling rate through linear interpolation, resampling the audio to meet simulation requirements. Shifting the model's output from spikes to membrane potentials improved both the accuracy and AUC significantly.

#### Files
processed_segments.py: Python script for processing segments of the dataset and resampling of sampling rate.
AFE_Simulation.ipynb: Simulation notebook containing the code for transforming raw audio data into neuromorphic format, compatible for Spiking Neural Networks. 
data_loader_to_model_training.ipynb: This notebook handles custom data loading and datasets and initial model training using spikes as outputs.
final_training.ipynb: The notebook for performing the final stage of training.
final_model.ipynb: Notebook responsible for the finalized model, metric visualisation, and simulation. 

### Dependencies and Key Libraries

#### Rockpool

This project relies heavily on **Rockpool**, a Python package designed for developing signal processing applications with Spiking Neural Networks (SNNs). Rockpool makes it easy to build, simulate, train, and test SNNs, and can also deploy models either in simulation or on event-driven neuromorphic hardware.

Rockpool supports multiple backends such as **Brian2**, **Torch**, **JAX**, **Numba**, and **NumPy**, and was chosen for this project because it simplifies the implementation of machine learning models based on SNNs. 

#### How Rockpool Was Used

In this project, **Rockpool** was utilized for:
- Converting raw audio data into neuromorphic spikes using its AFE module
- Building, training, and testing the Spiking Neural Network (SNN) models
- Simulating model performance and preparing it for deployment on hardware

Specifically, we leveraged Rockpool's support for **Torch** backends for simulation and training, ensuring compatibility with the Xylo Audio hardware.

### Installation Instructions

To replicate the setup and run the project, you will need to install **Rockpool** and its dependencies. Follow the instructions below to install Rockpool:

#### 6.1 Installing Rockpool

You can install Rockpool via `pip`. For a basic installation, run:

```bash
pip install rockpool --user
pip install rockpool[all] --user
git clone https://github.com/SynSense/rockpool.git
cd rockpool
pip install -e .[all] --user
```

For more detailed documentation and tutorials, visit the official Rockpool documentation.



