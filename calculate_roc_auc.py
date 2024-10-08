"run training loop and then run this"
import torch
import numpy as np

# Switch to spiking output after training
net.output = 'spike'

# Define a range of thresholds to test
threshold_vals = np.linspace(0, 10, num=50)  # Define your threshold range

# Initialize lists to store TPR and FDR for each threshold
tprs = []
fdrs = []

# Iterate through each threshold value
for th in threshold_vals:
    # Set output firing threshold
    net.seq[-1].threshold = torch.tensor(th, dtype=torch.float32)  # Set the threshold for spiking

    all_predictions = []
    all_labels = []

    # Run the test dataset through your model with the current threshold
    for inputs, labels in test_loader:  #  test_loader or validation
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            print("Input data contains NaN or infinity.")

        inputs = inputs.transpose(1, 2)  # Adjust input shape if necessary
        
        # Forward pass with the current threshold
        outputs, _, _ = net(inputs)  # Model outputs spikes based on the threshold
        if torch.isnan(outputs).any():
            print("NaN detected in outputs")
        #print(f"Model outputs at threshold {th}:", outputs)
        
        # Collect predictions and labels
        firing = (outputs > th).float()  # Apply threshold to generate spikes
        all_predictions.extend(firing.cpu().numpy())  # Sum spikes over time dimension
        all_labels.extend(labels.cpu().numpy())  # Collect ground truth labels

        # all_predictions.extend(firing.sum(dim=1).cpu().numpy())  # Sum spikes over time dimension
        # all_labels.extend(labels.cpu().numpy())  # Collect ground truth labels

    # Calculate TPR (True Positive Rate) and FDR (False Detection Rate)
    binary_predictions = (np.array(all_predictions) > 0.5).astype(int)  # Convert predictions to binary

    tp = np.sum((binary_predictions == 1) & (np.array(all_labels) == 1))  # True Positives
    fp = np.sum((binary_predictions == 1) & (np.array(all_labels) == 0))  # False Positives
    fn = np.sum((binary_predictions == 0) & (np.array(all_labels) == 1))  # False Negatives
    tn = np.sum((binary_predictions == 0) & (np.array(all_labels) == 0))  # True Negatives

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
    fdr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Detection Rate (similar to FPR)

    # Append results to lists
    tprs.append(tpr)
    fdrs.append(fdr)

# After the loop, plot the ROC curve (TPR vs FDR)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(fdrs, tprs, label='ROC Curve', color='b')
plt.plot([0, 1], [0, 1], 'r--')  # Random classifier line
plt.xlabel('False Detection Rate (FDR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve based on Spiking Threshold')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Optionally, calculate AUC (using FDR and TPR)
auc = np.trapz(tprs, fdrs)
print(f"AUC: {auc:.3f}")
