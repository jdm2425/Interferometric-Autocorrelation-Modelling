# Script to generate training data for CNN from simulated autocorrelation data
import os
from random import random
import sys
import time
sys.path.append('./')  # To allow import from parent directory
import AutocorrelationModelling as acm
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

# Create an instance of the AutocorrelationModelling class
ac_model = acm.Autocorrelation()

# synthetic spectrum similar to Oscillator
ac_model.create_domains(
    min_wavelength_nm=100, # Note: Temporal resolution depends on min wavelength
    max_wavelength_nm=1100,
    number_of_points_in_grid=2**14,
    nyquist_sampling_factor=2.0
) 

def generate_dataset(num_samples,
                     number_gaussians_per_spectrum_range=(10, 25),
                     gaussian_bandwidth_range_Hz=(1e12, 25e12),
                     central_frequency_range_Hz=(3.53e14, 4.0e14),
                     amplitude_range=(0.2, 1.0),
                     spectral_phase_gaussian_bandwidth_range_Hz=(50e12, 100e12),
                     noise_amplitude=0.01,
                     normalise=True,
                     save_data=False,
                     save_directory=None):
    # plt.figure(figsize=(10, 6))
    for i in range(num_samples):

        # Generate the gaussian parameters
        number_gaussians_per_spectrum = np.random.randint(
            number_gaussians_per_spectrum_range[0],
            number_gaussians_per_spectrum_range[1]+1
        )
        central_frequencies = np.random.uniform(
            central_frequency_range_Hz[0],
            central_frequency_range_Hz[1],
            number_gaussians_per_spectrum
        )
        spectral_bandwidths = np.random.uniform(
            gaussian_bandwidth_range_Hz[0],
            gaussian_bandwidth_range_Hz[1],
            number_gaussians_per_spectrum
        )
        amplitudes = np.random.uniform(
            amplitude_range[0], amplitude_range[1],
            number_gaussians_per_spectrum
        )
        # Alternative method From Paper
        # lambda_0 = np.random.uniform(600e-9, 900e-9)
        # gamma_0  = np.random.uniform(150e-9, 220e-9)
        # dlambda_0 = np.array([-gamma_0/4 + np.random.uniform(0,1)* (gamma_0/2) 
        #                       for _ in range(number_gaussians_per_spectrum)])
        # lambdas = lambda_0*np.ones_like(dlambda_0) + dlambda_0
        # central_frequencies = ac_model.c / lambdas
        
        # bandwidths_nm = np.array([4 + np.random.uniform(0, 1) * 30 
        #                           for _ in range(number_gaussians_per_spectrum)])
        # bandwidths = bandwidths_nm * 1e-9
        # spectral_bandwidths = []
        # for i in range(len(bandwidths)):
        #     spectral_bandwidths.append(np.abs(ac_model.c / (lambdas[i] + bandwidths[i]/2) - ac_model.c / (lambdas[i] - bandwidths[i]/2)))
        # spectral_bandwidths = np.array(spectral_bandwidths)
        
        # amplitudes = np.array([amplitude_range[0] + np.random.uniform(0, 1) 
        #                        * (amplitude_range[1] - amplitude_range[0]) 
        #                        for _ in range(number_gaussians_per_spectrum)])

        # # Create a synthetic spectrum
        # print(f"spectral_bandwidths: {spectral_bandwidths}")
        # print(f"central_frequencies: {central_frequencies}")
        # print(f"amplitudes: {amplitudes}")
        ac_model.create_synthetic_spectrum(
            central_frequency_Hz=central_frequencies,
            spectral_bandwidth_Hz=spectral_bandwidths,
            amplitude=amplitudes,
            noise_amplitude=noise_amplitude,
            normalise=normalise
        )
        
        # Generate a random phase
        # phases = np.random.uniform(-np.pi, np.pi, size=len(ac_model.frequency_domain))
        # sigma = np.random.uniform(spectral_phase_gaussian_bandwidth_range_Hz[0],
        #                           spectral_phase_gaussian_bandwidth_range_Hz[1]) / (ac_model.frequency_domain[1] - ac_model.frequency_domain[0])
        # # print(f"spectral phase sigma: {sigma}")
        # ac_model.spectral_phase = gaussian_filter1d(phases, sigma=sigma)
        # # Stretch phase to full range
        # ac_model.spectral_phase *= 1000/(np.max(np.abs(ac_model.spectral_phase)))  # Stretch phase to full range
        # remove linear component
        # p = np.polyfit(ac_model.frequency_domain, ac_model.spectral_phase, 1)
        # ac_model.spectral_phase -= np.polyval(p, ac_model.frequency_domain)


        # Alternatively, use spectral phase
        ac_model.create_spectral_phase(np.random.uniform(central_frequency_range_Hz[0],
                              central_frequency_range_Hz[1]),
                              CEP=np.random.uniform(-np.pi, np.pi),
                              group_delay=0, # fs
                              group_delay_dispersion=np.random.uniform(0, 550e-30), # fs2
                              third_order_dispersion=np.random.uniform(0, 350e-45), # fs3
                              fourth_order_dispersion=np.random.uniform(0, 250e-60), # fs4
                              )

        # Produce the autocorrelation
        ac_model.create_field_in_frequency()

        # ac_model.plot_field_in_frequency()

        ac_model.compute_field_in_time(
            hann_filter_bandwidth_Hz=95e12,  # Example value
            hann_filter_central_frequency_Hz=3.75e14,  # Example value
            plot=False
        )
        ac_model.autocorrelation_delays(
                                      min_delay_fs=-150,
                                      max_delay_fs=150,
                                      number_of_delay_points=1000
                                    )
        ac_model.autocorrelate(intensity_AC=False,
                               print_statements=True
                               )
        # ac_model.plot_autocorrelation(measured=False,
                                    #  simulated=True)
        if save_data is True:
            if save_directory is not None:
                # Check if sub-directories exist
                if os.path.exists(save_directory) == False:
                    user_input = input(f"\nDo you want to create directory '{save_directory}'? (y/n): ")
                    if user_input.lower() != 'y':
                        return
                    os.makedirs(save_directory)
                if not os.path.exists(os.path.join(save_directory, 'Autocorrelations')):  # Check for sub-directory
                    os.makedirs(os.path.join(save_directory, 'Autocorrelations'))
                if not os.path.exists(os.path.join(save_directory, 'Spectra')):  # Check for sub-directory
                    os.makedirs(os.path.join(save_directory, 'Spectra'))
                if not os.path.exists(os.path.join(save_directory, 'SpectralPhases')):  # Check for sub-directory
                    os.makedirs(os.path.join(save_directory, 'SpectralPhases'))
                # Save the data
                np.savetxt(os.path.join(save_directory, 'Autocorrelations', f'autocorr_{i:05d}.txt'),
                        np.column_stack((ac_model.simulated_autocorrelation_delays_fs,   
                                            ac_model.simulated_autocorrelation)),
                        header='Delay (fs), Autocorrelation (a.u.)')
                np.savetxt(os.path.join(save_directory, 'Spectra', f'spectrum_{i:05d}.txt'),
                        np.column_stack((ac_model.frequency_domain, ac_model.spectrum_in_frequency)),
                        header='Frequency (Hz), Spectrum (a.u.)')
                np.savetxt(os.path.join(save_directory, 'SpectralPhases', f'spectral_phase_{i:05d}.txt'),
                        np.column_stack((ac_model.frequency_domain, ac_model.spectral_phase)),
                        header='Frequency (Hz), Spectral Phase (rad)')
        
        # Need to get spectral data on common grid for CNN training
        # Num points 100
        # Get the FWHM and use that to define the grid
        fwhm = ac_model.get_FWHM(ac_model.frequency_domain, ac_model.spectrum_in_frequency)
        # print(f"\nFWHM: {fwhm*1e-12} THz")
        central_frequency = ac_model.get_FWHM_centre(ac_model.frequency_domain, ac_model.spectrum_in_frequency)
        # print(f"Central frequency: {central_frequency*1e-12} THz")
        grid = np.linspace(central_frequency - fwhm,
                           central_frequency + fwhm,
                           1000)
        spectrum_interp = interp1d(
            ac_model.frequency_domain,
            ac_model.spectrum_in_frequency
        )
        spectrum_resampled = spectrum_interp(grid)
        phase_interp = interp1d(
            ac_model.frequency_domain,
            ac_model.spectral_phase
        )
        spectral_phase_resampled = phase_interp(grid)
        # plt.figure()
        # # plt.plot(grid, spectrum_resampled)
        # plt.plot(ac_model.frequency_domain, ac_model.spectrum_in_frequency)
        # plt.plot(grid, spectrum_resampled)
        # plt.show()
        # plt.figure()
        # plt.plot(grid, spectral_phase_resampled)
        # # plt.plot(grid, spectral_phase_resampled)
        # plt.show()
        
        # CNN training data would be:
        # Inputs: frequency_grid, spectrum_resampled, delays, autocorr_values
        # Targets: spectral_phase_resampled, on the frequency_grid
        # Flattened input array
        input_array = np.concatenate((
            grid*1e-15, # len 1000 (PHz)
            spectrum_resampled, # len 1000
            ac_model.simulated_autocorrelation_delays_fs*1e-3, # len 1000 (in attoseconds)
            ac_model.simulated_autocorrelation # len 1000
        ))
        target_array = spectral_phase_resampled  # len 1000
        return input_array, target_array


# input_array, target_array = generate_dataset(num_samples=1, noise_amplitude=0.03, save_directory=r'CNN\Training-Data-1')

# frequencies, spectrum, delays, autocorr_values = ac_model.unflatten_CNN_input(input_array, 1000)  # To check unflattening works
# plt.figure()
# plt.plot(frequencies, spectrum, label='Spectrum')
# plt.show()
# plt.figure()
# plt.plot(delays, autocorr_values, label='Autocorrelation')
# plt.show()
# plt.figure()
# plt.plot(frequencies, target_array, label='Spectral Phase Target')
# plt.show()