# Code for fitting autocoorrelation data
# Jack Morse
# Created: Oct 2025
# Imperial College London
# Version 1.0

import numpy as np
import matplotlib.pyplot as plt
import AutocorrelationModelling as acm

# User must provide paths for a measured spectrum file and autocorrelation data file
spectrum_file = {'path': r"C:\Users\jdm24\OneDrive - Imperial College London\Documents\Chimera\Oscillator\Spectra\chimera_spectrum_27Oct_2025.txt",
                 'delimiter': '\t',
                 'skip_header': 18,
                 'wavelength_col': 0,
                 'signal_col': 1}

autocorr_file = {'path': r"D:\AC_Chimera_osc_AC_27oct2025_4.csv",
                 'delimiter': ',',
                 'skip_header': 1,
                 'delay_col': 0,
                 'signal_col': 1}

# Create an instance of the AutocorrelationModelling class
ac_model = acm.Autocorrelation()

# Load the measured spectrum
ac_model.load_spectrum(**spectrum_file)
# Or create a synthetic spectrum

# ac_model.create_domains(
#     min_wavelength_nm=100, # Note: Temporal resolution depends on min wavelength
#     max_wavelength_nm=1100,
#     number_of_points_in_grid=2**14,
#     nyquist_sampling_factor=2.0
# ) 

# ac_model.create_synthetic_spectrum(
#     central_frequency_Hz=[3.75e14, 3.58e14],
#     spectral_bandwidth_Hz=[75e12, 10e12],
#     amplitude=[1.3, 0.7]
# )

ac_model.show_spectrum()

ac_model.create_spectral_phase(
    central_frequency=3.75e14, # Hz
    CEP=0,
    group_delay=0e-15, # fs
    group_delay_dispersion=100e-30, # fs2
    third_order_dispersion=0e-45, # fs3
    fourth_order_dispersion=0e-60, # fs4
)

ac_model.create_field_in_frequency()

ac_model.plot_field_in_frequency(
    plot_spectral_phase=True,
    plot_amplitude=True,
    plot_field=True
)

ac_model.compute_field_in_time(
    hann_filter_bandwidth_Hz=75e12,  # Example value
    hann_filter_central_frequency_Hz=3.75e14,  # Example value
    plot=True
)

ac_model.autocorrelation_delays(
                              min_delay_fs=-200,
                              max_delay_fs=200,
                              number_of_delay_points=2**12
                            )

ac_model.autocorrelate(intensity_AC=False,
                       print_statements=True,
                       normalise=True
                       )

ac_model.load_autocorrelation(**autocorr_file)

ac_model.plot_autocorrelation(measured=True,
                             simulated=True)
