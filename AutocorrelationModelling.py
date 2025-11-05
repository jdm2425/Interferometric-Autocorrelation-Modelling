# InterferometricAC.py
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import simpson, trapezoid

class Conversions:
    def __init__(self):
        self.c = 299792458  # Speed of light in m/s

    def SIChangePrefix(self, value, input_prefix="", output_prefix=""):
        """
        Converts a value from one SI unit prefix to another.

        Parameters:
        value (float): The value to convert.
        input_prefix (str): The SI prefix of the input value. Default is the base SI unit (no prefix).
        output_prefix (str): The desired SI prefix for the output value. Default is the base SI unit.

        Returns:
        float: The value converted to the desired SI unit.
        """

        # Dictionary of SI prefixes and their corresponding powers of 10
        prefix_dict = {
            "": 0,         # Base SI unit
            "m": -3,       # milli
            "mu": -6,      # micro
            "n": -9,       # nano
            "p": -12,      # pico
            "f": -15,      # femto
            "a": -18,      # atto
            "z": -21,      # zepto
            "y": -24,      # yocto
            "k": 3,        # kilo
            "M": 6,        # mega
            "G": 9,        # giga
            "T": 12,       # tera
            "P": 15,       # peta
        }

        # Check if input and output prefixes are valid
        if input_prefix not in prefix_dict:
            raise ValueError(f"Invalid input prefix: {input_prefix}")
        if output_prefix not in prefix_dict:
          
            raise ValueError(f"Invalid output prefix: {output_prefix}")

        # Convert the input value to base SI (no prefix)
        base_value = value * (10 ** prefix_dict[input_prefix])

        # Convert from base SI to the desired output prefix
        output_value = base_value / (10 ** prefix_dict[output_prefix])

        return output_value

    def AngularFrequencyToWavelength(self, omega, input_SI_prefix="T", output_SI_prefix="n"):
        """
        Converts angular frequency (omega) to wavelength (lambda). 
        
        Parameters:
        - omega: The angular frequency (in radians per second).
        - input_SI_prefix: The SI prefix of the input angular frequency. Default is the base SI unit (no prefix).
        - output_SI_prefix: The desired SI prefix for the output wavelength. Default is the base SI unit.

        Returns:
        - wavelength: The wavelength (default nm).
        """
        # Calculate the wavelength
        omega = self.SIChangePrefix(omega, input_prefix=input_SI_prefix, output_prefix="")
        wavelengths = 2 * np.pi * self.c / np.abs(omega)
        return self.SIChangePrefix(wavelengths, input_prefix="", output_prefix=output_SI_prefix)

    def WavelengthToAngularFrequency(self, wavelength, input_SI_prefix="n", output_SI_prefix="T"):
        """
        Converts wavelength to angular frequency. 
        
        Parameters:
        - wavelength: The wavelength (default nm).
        - input_SI_prefix: The SI prefix of the input wavelength. Default is the base SI unit (no prefix).
        - output_SI_prefix: The desired SI prefix for the output angular frequency. Default is the base SI unit.

        Returns:
        - omega: The angular frequency (default in rad THz).
        """
        wavelength = self.SIChangePrefix(wavelength, input_prefix=input_SI_prefix, output_prefix="")
        omega = 2 * np.pi * self.c / wavelength
        return self.SIChangePrefix(omega, input_prefix="", output_prefix=output_SI_prefix)
    
    def Normalise(y):
        """
        Normalizes y values to range [0, 1] for real values,
        and scales complex values separately for magnitude.
        
        Parameters:
        y (array-like): Input y values, can be complex.
        
        Returns:
        np.ndarray: Normalized y values.
        """
        y = np.array(y)
        if np.iscomplexobj(y):
            magnitude = np.abs(y)
            normalized_magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))
            return normalized_magnitude * np.exp(1j * np.angle(y))
        else:
            return (y - np.min(y)) / (np.max(y) - np.min(y))


    
class PulseEquations:
    c = 299792458  # Speed of light in m/s

    def __init__(self):
        pass


    def get_initial_guess_for_gaussian_fit(x_data, y_data):
        """
        Get initial guess for Gaussian fit parameters based on the input data.
        
        Parameters:
        - x: The x values of the data.
        - y: The y values of the data.
        """
        idx_max = np.argmax(y_data)  # Index of max value
        std_dev_idx = np.argmin(np.abs(y_data - y_data[idx_max] * np.exp(-0.5)))  # Estimate width
        stddev = np.abs(x_data[idx_max] - x_data[std_dev_idx])  # Estimate stddev
        offset = y_data[0] # np.min(y_data)  # Estimate offset 
        amplitude = np.max(y_data) - offset  # Adjust amplitude to exclude offset
        return [amplitude, float(x_data[idx_max]), float(stddev), float(offset)]  

    def gaussian(x, amplitude, mean, stddev, offset):
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2)) + offset

    def gaussian_fit(x_data, y_data, initial_guess=None, print_params=True):
        if initial_guess is None:
            initial_guess = PulseEquations.get_initial_guess_for_gaussian_fit(x_data, y_data)

        params, covariance = curve_fit(PulseEquations.gaussian, x_data, y_data, p0=initial_guess)
        amplitude, mean, stddev, offset = params

        if print_params:
            print(f"Gaussian fit parameters:\n"
                f"Amplitude = {amplitude:.2f}\n"
                f"Mean = {mean:.2f}\n"
                f"Standard deviation = {stddev:.2f}\n"
                f"Offset = {offset:.2f}\n"
                f"FWHM = {2 * np.sqrt(2 * np.log(2)) * stddev:.2f}")

        y_fit = PulseEquations.gaussian(x_data, *params)
        return y_fit, params
    
    def XlimAroundPeakFeature(x, y, window_size=10e-15, print_statements=False):
        """
        Assuming a gaussian-esque signal with a single dominant peak, this returns the x_lim around the feature for a specified width, useful for zooming in.
        
        Parameters:
        - x [array]: x domain
        - y [array]: y domain
        - window_size [float]: total size of the x_range
        
        Returns:
        - xlim: [xmin, xmax]
        """
        idx_of_peak_feature = np.argmax(np.abs(y))

        number_of_indices = round(window_size / np.diff(x)[0])
        if print_statements:
            print(f"Index of time feature (max): {idx_of_peak_feature}\nNumber of indices: {number_of_indices}")
            # print(f"Length of y array: {len(y)}")
            print(f"Xloc of peak: {x[idx_of_peak_feature]}")

        if (idx_of_peak_feature - number_of_indices) >= 0:
            min_time = x[idx_of_peak_feature - number_of_indices]
        else: 
            min_time = x[0]
        if (idx_of_peak_feature + number_of_indices) <= len(x):
            max_time = x[idx_of_peak_feature + number_of_indices]
        else:
            max_time = x[len(x) - 1]
        return [min_time, max_time]

    

    
class Autocorrelation:
    
    def __init__(self, 
        ):
        self.frequency_domain = np.ndarray([]) # Hz
        self.spectrum_in_frequency = np.ndarray([]) # a.u.
        self.wavelength_domain = np.ndarray([])
        self.spectrum_in_wavelength = np.ndarray([])
        self.spectral_phase = np.ndarray([])

        self.field_in_frequency = np.ndarray([])
        self.field_in_time = np.ndarray([])
        self.time_domain = np.ndarray([])

        self.simulated_autocorrelation_delays = np.ndarray([])
        self.simulated_autocorrelation = np.ndarray([])

        self.measured_autocorrelation_delays = np.ndarray([])
        self.measured_autocorrelation = np.ndarray([])

        self.c = 299792458  # Speed of light in m/s
                
    def load_spectrum(self, 
                      path,
                      delimiter='\t',
                      skip_header=1,
                      wavelength_col=0,
                      signal_col=1,
                      skip_footer=1,
                      normalise=True):
        df = pd.read_csv(
            path,
            delimiter=delimiter,
            skiprows=skip_header,
            header=None,                 # Treat all rows as data
            comment=None,                # Don’t skip lines starting with #
            engine='python',             # More forgiving parser
        )
        # Remove footer rows if specified
        spectrometer_wavelengths = df[wavelength_col]
        spectrometer_signal = df[signal_col]
        if skip_footer > 0:
            spectrometer_wavelengths = np.array(spectrometer_wavelengths[0:-1], dtype=float)
            spectrometer_signal = np.array(spectrometer_signal[0:-1], dtype=float)
        self.wavelengths = np.array(spectrometer_wavelengths, dtype=float)
        self.spectrum_in_wavelength = np.array(spectrometer_signal, dtype=float)
        if normalise:
            self.spectrum_in_wavelength = Conversions.Normalise(self.spectrum_in_wavelength)
        # Interpolate to frequency domain
        if self.frequency_domain is None:
            pass

    def interpolate_spectrum_to_wavelength_domain(self):
        pass
    def interpolate_spectrum_to_frequency_domain(self):
        pass

    def create_domains(self,
                                min_wavelength_nm=400,
                                max_wavelength_nm=700,
                                number_of_points_in_grid=2048,
                                nyquist_sampling_factor=2.0):
        """
        Creates frequency and wavelength domains for modelling.
        """
        maximum_frequency = self.c / (min_wavelength_nm * 1e-9)  # Hz
        sampling_frequency = nyquist_sampling_factor * maximum_frequency
        dt = 1.0 / sampling_frequency  # s
        # Frequency grid
        self.frequency_domain = np.fft.fftfreq(n=number_of_points_in_grid, d=dt) # Hz, unshifted
        self.wavelength_domain = np.linspace(min_wavelength_nm, max_wavelength_nm, number_of_points_in_grid)  # nm

    def convert_FWHM_spectral_bandwidth_nm_to_Hz(self,
                                                 spectral_bandwidth_nm,
                                                 central_wavelength_nm):
        spectral_bandwidth_Hz = (self.c / 1e-9) * (1 / (central_wavelength_nm - spectral_bandwidth_nm/2) - 1 / (central_wavelength_nm + spectral_bandwidth_nm/2))
        return spectral_bandwidth_Hz

    def create_synthetic_spectrum(self,
                                   amplitude=1.0,
                                   central_frequency_Hz=3.58e14,
                                   spectral_bandwidth_Hz=75,
                                   normalise=True):
        """
        Creates a synthetic Gaussian spectrum in the frequency domain. Can make a single Gaussian or a sum of multiple Gaussians.
        Parameters:
        - frequency_domain: Array of frequency values (Hz).
        - amplitude: Peak amplitude of the spectrum.
        - central_frequency_Hz: Central frequency of the spectrum (Hz).
        - spectral_bandwidth_Hz: Full width at half maximum (FWHM) bandwidth of the spectrum (Hz).
        """
        if self.frequency_domain is None or self.wavelength_domain is None:
            raise ValueError("Frequency and wavelength domains must be initialised! Call create_domains() first.")
        
        # Check to see if amplitude, central_frequency_Hz, spectral_bandwidth_Hz are arrays or scalars
        if np.isscalar(amplitude) and np.isscalar(central_frequency_Hz) and np.isscalar(spectral_bandwidth_Hz):
            self.spectrum_in_frequency = amplitude * np.exp(-(4 * np.log(2) * (self.frequency_domain - central_frequency_Hz)**2) /
                                    (spectral_bandwidth_Hz**2))
        elif isinstance(amplitude, (list, np.ndarray)) and isinstance(central_frequency_Hz, (list, np.ndarray)) and isinstance(spectral_bandwidth_Hz, (list, np.ndarray)):
            assert(len(amplitude) == len(central_frequency_Hz) == len(spectral_bandwidth_Hz)), "amplitude, central_frequency_Hz, and spectral_bandwidth_Hz must have the same length."
            self.spectrum_in_frequency = sum(
                a * np.exp(-(4 * np.log(2) * (self.frequency_domain - c)**2) /
                                (b**2))
                for a, c, b in zip(amplitude, central_frequency_Hz, spectral_bandwidth_Hz)
            )
        else:
            raise ValueError("amplitude, central_frequency_Hz, and spectral_bandwidth_Hz must be either all scalars or all lists/arrays of the same length.")
        
        # Interpolate to wavelength domain
        indices_of_only_positive_frequencies = np.where(self.frequency_domain > 0)
        positive_frequencies = self.frequency_domain[indices_of_only_positive_frequencies]
        spectrum_at_positive_frequencies = self.spectrum_in_frequency[indices_of_only_positive_frequencies]
        wavelength_from_frequencies = self.c / (positive_frequencies * 1e-9)  # Convert frequency to wavelength in nm
        # spectrum_in_wavelength = spectrum_at_positive_frequencies[::-1]  # Reverse frequency array for correct mapping
        spectrum_in_wavelength = spectrum_at_positive_frequencies

        interp_func = interp1d(
            wavelength_from_frequencies,
            spectrum_in_wavelength,
            kind='linear',
            bounds_error=False,
            fill_value=(spectrum_in_wavelength[0], spectrum_in_wavelength[-1])         # use first/last value beyond range
        )
        self.spectrum_in_wavelength = interp_func(self.wavelength_domain)
        if normalise:
            self.spectrum_in_wavelength = Conversions.Normalise(self.spectrum_in_wavelength)
            self.spectrum_in_frequency = Conversions.Normalise(self.spectrum_in_frequency)

    def create_spectral_phase(self):  
        pass

    def create_field_in_frequency(self):
        pass
    

    def show_spectrum(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.wavelength_domain, self.spectrum_in_wavelength, '-', lw=1, color='C0')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (a.u.)')
        plt.title('Stored Spectrum')
        plt.grid(True)
        plt.show()

    def GetDetectorSignal(pulse_1_electric_field_in_t, pulse_2_electric_field_in_t, time_domain, method="trapezoid", intensity_AC=False):
        if intensity_AC:
            integrand_function = (np.abs(pulse_1_electric_field_in_t)**2)*np.abs(pulse_2_electric_field_in_t**2)
        else:
            integrand_function = np.abs( (pulse_1_electric_field_in_t + pulse_2_electric_field_in_t)**2 )**2
        # integrand_function =  np.abs( pulse_1_electric_field_in_t + pulse_2_electric_field_in_t )**2
        if not isinstance(method, str):
            raise ValueError("method should be string from list ['trapezoid', 'simpson']")
        if method.lower() == "trapezoid":
            integrated_signal = trapezoid(integrand_function, time_domain)
        elif method.lower() == "simpson":
            integrated_signal = simpson(integrand_function, time_domain)
        else: print("Unrecognised method value, should be string from list ['trapezoid', 'simpson']")  
        # print(f"Diff: {integrated_signal - integrated_signal_simpson}")
        return integrated_signal
    
    def Autocorrelate(time_domain, pulse_1_in_time, autocorrelation_delays, intensity_AC=False):
        interferometric_autocorrelation_values = []
        from scipy.interpolate import interp1d
        for delay in  autocorrelation_delays: # Varying delay from 0 to 10 femtoseconds
            interpolator = interp1d(time_domain, pulse_1_in_time, kind='linear', fill_value="extrapolate")
            t_delayed = time_domain - delay  # Shifted time vector
            delayed_signal = interpolator(t_delayed)
            pulse_2_in_time = delayed_signal
            # pulse_sum = pulse_1_in_time + pulse_2_in_time
            autocorrelation_value = Autocorrelation.GetDetectorSignal(pulse_1_in_time, pulse_2_in_time, time_domain, intensity_AC=intensity_AC)
            interferometric_autocorrelation_values.append(autocorrelation_value)
        return np.array(interferometric_autocorrelation_values)


class LaserPulse:
    def __init__(self):
        pass