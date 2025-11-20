import numpy as np
def get_FWHM(self,
                 x_data,
                 y_data):
        half_max = np.max(y_data) / 2.0
        indices_above_half_max = np.where(y_data >= half_max)[0]
        if len(indices_above_half_max) < 2:
            return 0  # FWHM cannot be determined
        fwhm = x_data[indices_above_half_max[-1]] - x_data[indices_above_half_max[0]]
        return fwhm

def compute_AC_from_spectrum_and_phase_data(spectrum_frequencies_Hz, spectrum, phase_values_rad):
    # Given a spectrum and phase, produce the autocorrelation
    E_f = np.sqrt(spectrum) * np.exp(1j * phase_values_rad)
    E_t = np.fft.ifftshift(np.fft.ifft(E_f))
    time_domain = np.arange(len(spectrum_frequencies_Hz)) / (2 * np.max(spectrum_frequencies_Hz))  # time grid
    delays_fs = np.linspace(-200, 200, 2**12)  # delay grid in fs
    autocorr_values = []
    for delay_fs in delays_fs:
        delay_s = delay_fs * 1e-15
        E_t_delayed = np.interp(time_domain - delay_s, time_domain, E_t, left=0, right=0)
        integrand_function = np.abs( (E_t + E_t_delayed)**2 )**2
        # plt.plot(time_domain, np.real(E_t+ E_t_delayed))
        # plt.plot(time_domain, np.real(E_t), label='E_t')
        # plt.plot(time_domain, np.real(E_t_delayed), label='E_t_delayed')
        # plt.legend()
        # plt.show()
        autocorr_value = np.trapz(integrand_function, time_domain)
        autocorr_values.append(autocorr_value)
    return 8*np.array(autocorr_values)/np.max(np.array(autocorr_values)), delays_fs

# # Load data: 
# maximum_frequency = 3e8 / (100 * 1e-9)  # Hz
# sampling_frequency = 2.0 * maximum_frequency
# number_of_points_in_grid = 2**12
# dt = 1.0 / sampling_frequency  # s
# # Frequency grid
# frequency_domain = np.fft.fftfreq(n=number_of_points_in_grid, d=dt) # Hz, unshifted
# spectrum = np.exp(-4 * np.log(2) * ((frequency_domain - 3.75e14) / (75e12))**2)  # Gaussian spectrum
# phase_values_rad = (1e-30) * (2 * np.pi * (frequency_domain - 3.75e14))**2  # GDD phase

# import matplotlib.pyplot as plt
# # plt.plot(frequency_domain, phase_values_rad)
# # plt.show()

# autocorr_values, delays_fs = compute_AC_from_spectrum_and_phase_data(
#     spectrum_frequencies_Hz=frequency_domain,
#     spectrum=spectrum,
#     phase_values_rad=phase_values_rad
# )
# plt.plot(delays_fs, autocorr_values)
# plt.show()


# ---------------- #
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

class InterferometricAutocorrelationModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(InterferometricAutocorrelationModel, self).__init__()
        # Define layers here
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential( 
            nn.Linear(input_size, 512), # Input layer
            nn.ReLU(), # Activation function, introducing non-linearity so the network can learn complex patterns
            nn.Linear(512, 512), # Hidden layer, only layer here
            nn.ReLU(), # Activation function
            nn.Linear(512, num_classes), # Output layer
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 28*28
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Load dataset (example with MNIST, replace with actual dataset)
train_dataset = datasets.MNIST(
    root='data',
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)
train_loader = DataLoader(
     dataset=train_dataset,
     batch_size=batch_size,
     shuffle=True # Shuffle the order of the data between epochs
) 

test_dataset = datasets.MNIST(
    root='data',
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# Initialize network
model = InterferometricAutocorrelationModel(input_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
     for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Get to correct shape
        data = data.reshape(data.shape[0], -1)

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward
        optimizer.zero_grad() # Set all gradients to zero for each batch, so it doesnt remember the gradients calulated for forward pass of previous batch
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

# Check accuracy
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad(): # When testing, we don't need to calculate gradients (for memory efficiency)
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}%')
    
    model.train()
    return float(num_correct)/float(num_samples)

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
         
