import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('./')
import Generate_Training_Data as gtd
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
import AutocorrelationModelling as acm

class InterferometricAutocorrelationModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(InterferometricAutocorrelationModel, self).__init__()
        # Define layers here
        # self.flatten = nn.Flatten()
        # self.linear_relu_stack = nn.Sequential( 
        #     nn.Linear(input_size, 2048), # Input layer
        #     nn.ReLU(), # Activation function, introducing non-linearity so the network can learn complex patterns
        #     nn.Linear(2048, 2048), # Hidden layer, only layer here
        #     nn.SiLU(), # Activation function
        #     nn.Linear(2048, 2048), # Hidden layer, only layer here
        #     nn.ReLU(), # Activation function
        #     nn.Linear(2048, num_classes), # Output layer
        # )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.SiLU(),
            nn.LayerNorm(2048),

            nn.Linear(2048, 1536),
            nn.SiLU(),
            nn.LayerNorm(1536),

            nn.Linear(1536, 1024),
            nn.SiLU(),
            nn.LayerNorm(1024),

            nn.Linear(1024, num_classes)
        )


    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 1000*4
num_classes = 1000
learning_rate = 0.001
# batch_size = 64
num_epochs = 10000

setting = ""
plot = False

if setting == "train":
    # Initialize network
    model = InterferometricAutocorrelationModel(input_size, num_classes).to(device)
    print("Model initialized and moved to device: ", device)
    # Loss and optimizer
    criterion = criterion = nn.MSELoss() # nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create a figure to plot the loss
    if plot == True:
        plt.figure()
        plt.ion()  # Turn on interactive mode for live updating
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()
    epochs_list = []
    loss_list = []
    # Training loop
    for epoch in range(num_epochs):
        # a = data.to(device=device)
        # targets = targets.to(device=device)

        # Get to correct shape
        input_array, target_array = gtd.generate_dataset(num_samples=1, noise_amplitude=0.03, save_directory=r'CNN\Training-Data-1')

        data = torch.from_numpy(input_array).float().to(device=device)
        target_array = torch.from_numpy(target_array).float().to(device=device)

        # Forward
        scores = model(data)
        loss = criterion(scores, target_array)

        # Backward
        optimizer.zero_grad() # Set all gradients to zero for each batch, so it doesnt remember the gradients calulated for forward pass of previous batch
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

        # Update the plot
        epochs_list.append(epoch)
        loss_list.append(loss.item())
        if plot == True:
            plt.plot(epochs_list, loss_list, 'b-')
            plt.pause(0.1)
        print(f"\nPercentage done = {(epoch/num_epochs)*100:.2f}")
    # plt.show()
    # input("Press enter..")
    print("Training complete.")
    torch.save(model.state_dict(), r"C:\\Users\\jdm24\\OneDrive - Imperial College London\\Documents\\Diagnostics\\LED Autocorrelator\\Interferometric Autocorrelation Modelling\\Interferometric-Autocorrelation-Modelling\\CNN\\model_weights.pth")
    print("Model weights saved.")
    print("saving loss...")
    import pandas as pd
    df = pd.DataFrame({'epoch': epochs_list, 'loss': loss_list})
    save_path = r"C:\Users\jdm24\OneDrive - Imperial College London\Documents\Diagnostics\LED Autocorrelator\Interferometric Autocorrelation Modelling\Interferometric-Autocorrelation-Modelling\CNN\loss_history.csv"
    df.to_csv(save_path, index=False)
    print(f"Loss history saved to {save_path}")



else:
    model = InterferometricAutocorrelationModel(input_size, num_classes).to(device)
    model.load_state_dict(torch.load(r"C:\\Users\\jdm24\\OneDrive - Imperial College London\\Documents\\Diagnostics\\LED Autocorrelator\\Interferometric Autocorrelation Modelling\\Interferometric-Autocorrelation-Modelling\\CNN\\model_weights.pth"))
    model.eval()


    def check_accuracy(input_data, target_data, model, individual_array_length=1000):
        ac =acm.Autocorrelation()
        freqs, spectrum, delays, autocorrelation = ac.unflatten_CNN_input(CNN_input=input_data, individual_array_length=individual_array_length)
        with torch.no_grad(): # When testing, we don't need to calculate gradients (for memory efficiency)
            output = model(input_data)
            output_array = output.cpu().numpy()
        print("Plotting!")
        plt.plot(freqs, spectrum)
        plt.show()
        plt.cla()
        plt.plot(delays, autocorrelation/8)
        plt.show()
        plt.cla()
        plt.plot(freqs, target_data, label='Phase target')
        plt.plot(freqs, output_array, label='Predicted phase')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Phase (a.u.)")
        plt.legend()
        plt.show()
        return
        
    

    input_array, target_array = gtd.generate_dataset(num_samples=1, noise_amplitude=0.03, save_directory=r'CNN\Training-Data-1')
    check_accuracy(torch.from_numpy(input_array).float().to(device=device), target_array, model)

    # Read and plot loss history
    print("Plotting history")
    import pandas as pd
    loss_path = r"C:\Users\jdm24\OneDrive - Imperial College London\Documents\Diagnostics\LED Autocorrelator\Interferometric Autocorrelation Modelling\Interferometric-Autocorrelation-Modelling\CNN\loss_history.csv"
    loss_df = pd.read_csv(loss_path)
    plt.figure()
    plt.plot(loss_df['epoch'], loss_df['loss'])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.show()