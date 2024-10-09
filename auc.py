import numpy as np
import matplotlib.pyplot as plt

# Set the output to spike
net.output = 'spike'

# Define a range of thresholds to test
threshold_vals = np.linspace(0, 1, num=50)  # Adjusted to a more realistic range

# Initialize lists to store TPR and FDR for each threshold
tprs = []
fdrs = []

# Iterate through each threshold value
for th in threshold_vals:
    # Set output firing threshold
    net.seq[-1].threshold = torch.tensor(th, dtype=torch.float32).to(device)

    all_predictions = []
    all_labels = []

    # Run the test dataset through your model
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Adjust input shape if necessary
        inputs = inputs.transpose(1, 2)

        # Forward pass with the current threshold
        outputs, _, _ = net(inputs)

        # Perform prediction
        firing = (outputs > th).float()  # Apply threshold to generate spikes

        # Accumulate predictions
        all_predictions.extend(firing.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

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
