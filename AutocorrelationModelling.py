# InterferometricAC.py
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
from scipy.optimize import curve_fit
import sys
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
    def NormaliseByArea(x, y):
        """
        Normalizes y values so that the area under the curve is 1.
        
        Parameters:
        y (array-like): Input y values.
        x (array-like): Corresponding x values.
        
        Returns:
        np.ndarray: Area-normalized y values.
        """
        area = simpson(y, x)
        if area == 0:
            return y
        y /= area
        # check
        # print(f"Area after normalisation: {simpson(y, x)}")
        return y
        


    
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
        self.sampling_frequency = None  # Hz
        self.number_of_points_in_grid = None # integer
        self.spectrum_in_frequency = np.ndarray([]) # a.u.
        self.wavelength_domain = np.ndarray([])
        self.spectrum_in_wavelength = np.ndarray([])
        self.spectral_phase = np.ndarray([])

        self.field_in_frequency = np.ndarray([])
        self.field_in_time = np.ndarray([])
        self.time_domain = np.ndarray([])

        self.simulated_autocorrelation_delays_fs = np.ndarray([])
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
        self.wavelength_domain = np.array(spectrometer_wavelengths, dtype=float)
        self.spectrum_in_wavelength = np.array(spectrometer_signal, dtype=float)
        self.number_of_points_in_grid = len(self.wavelength_domain)
        if normalise:
            self.spectrum_in_wavelength = Conversions.Normalise(self.spectrum_in_wavelength)
        # Interpolate to frequency domain
        frequency_from_wavelengths = self.c / (self.wavelength_domain * 1e-9)  # Convert wavelength to frequency in Hz
        interp_func = interp1d(frequency_from_wavelengths,
                               self.spectrum_in_wavelength,
                               bounds_error=False,
                               fill_value=(self.spectrum_in_wavelength[0], self.spectrum_in_wavelength[-1]))        # use first/last value beyond range)
        self.create_frequency_domain(min_wavelength_nm=np.min(self.wavelength_domain),
                                      number_of_points_in_grid=self.number_of_points_in_grid)
        self.time_domain = np.arange(self.number_of_points_in_grid) / self.sampling_frequency  # time grid
        self.spectrum_in_frequency = interp_func(self.frequency_domain)
        # plt.plot(self.frequency_domain, self.spectrum_in_frequency)
        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel("Intensity (a.u.)")
        # plt.title("Spectrum in Frequency Domain")
        # plt.show()

    def interpolate_spectrum_to_wavelength_domain(self):
        pass

    def interpolate_spectrum_to_frequency_domain(self):
        pass
    
    def create_frequency_domain(self, min_wavelength_nm=400, number_of_points_in_grid=2048, nyquist_sampling_factor=2.0):
        maximum_frequency = self.c / (min_wavelength_nm * 1e-9)  # Hz
        self.sampling_frequency = nyquist_sampling_factor * maximum_frequency
        self.number_of_points_in_grid = number_of_points_in_grid
        dt = 1.0 / self.sampling_frequency  # s
        # Frequency grid
        self.frequency_domain = np.fft.fftfreq(n=number_of_points_in_grid, d=dt) # Hz, unshifted
    
    def create_domains(self,
                                min_wavelength_nm=400,
                                max_wavelength_nm=700,
                                number_of_points_in_grid=2048,
                                nyquist_sampling_factor=2.0):
        """
        Creates frequency and wavelength domains for modelling.
        """
        self.create_frequency_domain(min_wavelength_nm=min_wavelength_nm,
                                      number_of_points_in_grid=number_of_points_in_grid,
                                      nyquist_sampling_factor=nyquist_sampling_factor)
        
        self.wavelength_domain = np.linspace(min_wavelength_nm, max_wavelength_nm, number_of_points_in_grid)  # nm
        self.time_domain = np.arange(self.number_of_points_in_grid) / self.sampling_frequency  # time grid

    def convert_FWHM_spectral_bandwidth_nm_to_Hz(self,
                                                 spectral_bandwidth_nm,
                                                 central_wavelength_nm):
        spectral_bandwidth_Hz = (self.c / 1e-9) * (1 / (central_wavelength_nm - spectral_bandwidth_nm/2) - 1 / (central_wavelength_nm + spectral_bandwidth_nm/2))
        return spectral_bandwidth_Hz
    
    def get_FWHM(self,
                 x_data,
                 y_data):
        half_max = np.max(y_data) / 2.0
        indices_above_half_max = np.where(y_data >= half_max)[0]
        if len(indices_above_half_max) < 2:
            return 0  # FWHM cannot be determined
        fwhm = x_data[indices_above_half_max[-1]] - x_data[indices_above_half_max[0]]
        return fwhm
    
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

    def create_spectral_phase(self,
                              central_frequency, # Hz
                              CEP=0,
                              group_delay=0e-15, # fs
                              group_delay_dispersion=0e-30, # fs2
                              third_order_dispersion=0e-45, # fs3
                              fourth_order_dispersion=0e-60, # fs4
                              ):  
        # Check frequency domain is initialised
        if self.frequency_domain is np.array([]):
            raise ValueError("Frequency domain must be initialised! Call create_domains() first.")
        
        # Spectral phase (Taylor expansion about central frequency)
        phi_0 = CEP
        phi_1 = group_delay * 2 * np.pi
        phi_2 = group_delay_dispersion * (2 * np.pi)**2
        phi_3 = third_order_dispersion * (2 * np.pi)**3
        phi_4 = fourth_order_dispersion * (2 * np.pi)**4

        self.spectral_phase = (phi_0
                + phi_1 * (self.frequency_domain - central_frequency)
                + 0.5 * phi_2 * (self.frequency_domain - central_frequency)**2
                + (1.0/6.0) * phi_3 * (self.frequency_domain - central_frequency)**3
                + (1.0/24.0) * phi_4 * (self.frequency_domain - central_frequency)**4)

    def create_field_in_frequency(self):
        self.field_in_frequency = np.sqrt(self.spectrum_in_frequency) * np.exp(1j * self.spectral_phase)

    def plot_field_in_frequency(self,
                                plot_spectral_phase=True,
                                plot_amplitude=False,
                                plot_field=True):
        # Create figure with twin axes
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        position_of_peak = np.where(self.field_in_frequency == np.max(self.field_in_frequency))[0]
        fwhm = self.get_FWHM(self.frequency_domain, np.abs(self.field_in_frequency))
        ax1.set_xlim([self.frequency_domain[position_of_peak] - 3*fwhm, self.frequency_domain[position_of_peak] + 3*fwhm])
        # Plot amplitude/field on first axis
        if plot_amplitude:
            ax1.plot(self.frequency_domain, np.abs(self.field_in_frequency), 'b-', label='Amplitude')
        if plot_field:
            ax1.plot(self.frequency_domain, np.real(self.field_in_frequency), 'g-', label='Real Field')
            ax1.plot(self.frequency_domain, np.imag(self.field_in_frequency), 'r-', label='Imaginary Field')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Amplitude', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # Plot phase on second axis if requested
        if plot_spectral_phase:
            ax2.plot(self.frequency_domain, np.unwrap(np.angle(self.field_in_frequency)), 'k--', label='Phase')
            ax2.set_ylabel('Phase (rad)', color='k')
            ax2.tick_params(axis='y', labelcolor='k')
            min_y_value = np.min(np.unwrap(np.angle(self.field_in_frequency))[(self.frequency_domain >= ax1.get_xlim()[0]) & (self.frequency_domain <= ax1.get_xlim()[1])])
            max_y_value = np.max(np.unwrap(np.angle(self.field_in_frequency))[(self.frequency_domain >= ax1.get_xlim()[0]) & (self.frequency_domain <= ax1.get_xlim()[1])])
            print(f"Phase min/max in view: {min_y_value}, {max_y_value}")
            ax2.set_ylim([min_y_value, max_y_value])
             

        # Add legend and show plot
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2)
        plt.title('Field in Frequency Domain')
        plt.show()

    def apply_hann_window(self,
                          filter_central_frequency_Hz,
                          filter_bandwidth_Hz):
        """
        Applies a Hann window centered at `center_freq` with width `bandwidth`.
        Tapers smoothly to zero outside the region.
        """
        # Half-width in Hz
        half_bw = filter_bandwidth_Hz / 2.0

        #  Create normalized frequency distance from center
        f_offset = self.frequency_domain - filter_central_frequency_Hz
        # Initialize window with zeros
        window = np.zeros_like(self.frequency_domain)
        # Identify region inside the spectral window
        inside = np.abs(f_offset) <= half_bw
        # Apply Hann taper within the region
        # Hann = 0.5 * (1 + cos(pi * x / half_bw)) for |x| <= half_bw
        window[inside] = 0.5 * (1 + np.cos(np.pi * f_offset[inside] / half_bw))
        # Apply window to spectrum
        E_f_windowed = self.field_in_frequency * window
        return E_f_windowed, window

    def compute_field_in_time(self,
                             hann_filter_central_frequency_Hz=None,
                             hann_filter_bandwidth_Hz=None,
                             normalisation=None,
                             plot=False
                             ):
        """
        Computes the time-domain electric field via inverse FFT.
        Optionally applies a Hann window filter in the frequency domain before transformation.
        Parameters:
        - hann_filter_central_frequency_Hz: Central frequency of the Hann window filter (Hz)
        - hann_filter_bandwidth_Hz: Bandwidth of the Hann window filter (Hz)
        - normalise: Whether to normalise the time-domain field ('area', 'amplitude' or None)
        - plot: Whether to plot the frequency and time domain fields
        """
        assert self.field_in_frequency.size > 1, \
            "Field in frequency domain must be initialised! Call create_field_in_frequency() first."
        hann_window = None
        if hann_filter_central_frequency_Hz is not None and hann_filter_bandwidth_Hz is not None:
            field_in_frequency, hann_window = self.apply_hann_window(hann_filter_central_frequency_Hz, hann_filter_bandwidth_Hz)
        else:
            field_in_frequency = self.field_in_frequency

        self.field_in_time = self.sampling_frequency * np.fft.ifft(field_in_frequency)   # includes Δf factor
        if normalisation is not None:
            if normalisation.lower() == 'area':
                self.field_in_time = Conversions.NormaliseByArea(self.time_domain, self.field_in_time)
            elif normalisation.lower() == 'amplitude':
                self.field_in_time = Conversions.Normalise(self.field_in_time)
            else:
                raise ValueError(f"Unknown normalisation method: {normalisation}. Choose from ['area', 'amplitude', None].")
        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
            
            # Plot spectrum and filter
            ax1.plot(self.frequency_domain, np.abs(self.field_in_frequency), 'b-', label='Original Spectrum')
            if hann_window is not None:
                ax1.plot(self.frequency_domain, np.abs(field_in_frequency), 'r-', label='Filtered Spectrum')
                ax1.plot(self.frequency_domain, hann_window, 'g--', label='Filter')
            ax1.set_xlabel('Frequency (Hz)')
            ax1.set_ylabel('Amplitude')
            position_of_peak = np.where(self.field_in_frequency == np.max(self.field_in_frequency))[0]
            fwhm = self.get_FWHM(self.frequency_domain, np.abs(self.field_in_frequency))
            ax1.set_xlim([self.frequency_domain[position_of_peak] - 3*fwhm, self.frequency_domain[position_of_peak] + 3*fwhm])
            ax1.set_ylim([0, 1.1 * np.max(np.abs(self.field_in_frequency))])
            ax1.legend()
            ax1.grid(True)
            
            # Plot time domain field
            ax2.plot(self.time_domain*1e15, np.real(np.fft.fftshift(self.field_in_time)), 'b-', label='Field')
            ax2.set_xlabel('Time (fs)')
            ax2.set_ylabel('Amplitude')
            position_of_peak = np.where(np.real(np.fft.fftshift(self.field_in_time)) == np.max(np.real(np.fft.fftshift(self.field_in_time))))[0]
            fwhm = self.get_FWHM(self.time_domain, np.real(np.fft.fftshift(self.field_in_time)))
            ax2.set_xlim([(self.time_domain[position_of_peak] - 3*fwhm)*1e15, (self.time_domain[position_of_peak] + 3*fwhm)*1e15])
            ax2.set_ylim([-1.1 * np.max(np.abs(self.field_in_time)), 1.1 * np.max(np.abs(self.field_in_time))])
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
        pass

    def show_spectrum(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.wavelength_domain, self.spectrum_in_wavelength, '-', lw=1, color='C0')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (a.u.)')
        plt.title('Stored Spectrum')
        plt.grid(True)
        plt.show()
    
    def get_detector_signal(pulse_1_electric_field_in_t,
                            pulse_2_electric_field_in_t,
                            time_domain, method="trapezoid",
                            intensity_AC=False):
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
    
    def autocorrelation_delays(self,
                              min_delay_fs=0,
                              max_delay_fs=100,
                              number_of_delay_points=500):
        self.simulated_autocorrelation_delays_fs = np.linspace(min_delay_fs, max_delay_fs, number_of_delay_points)

    def autocorrelate(self, intensity_AC=False,
                      print_statements=False,
                      normalise=True):
        interferometric_autocorrelation_values = []

        if self.simulated_autocorrelation_delays_fs.size <= 1:
            raise ValueError("Autocorrelation delays must be initialised! Call autocorrelation_delays() first.")
        
        if print_statements: print("Calculating autocorrelation...")
        n_delays = len(self.simulated_autocorrelation_delays_fs)
        bar_length = 30  # characters wide

        pulse_1_in_time = np.fft.fftshift(self.field_in_time)

        for i, delay in enumerate(self.simulated_autocorrelation_delays_fs): # Varying delay from 0 to 10 femtoseconds
            interpolator = interp1d(self.time_domain, pulse_1_in_time, kind='linear', fill_value="extrapolate")
            t_delayed = self.time_domain - delay*1e-15  # Shifted time vector
            delayed_signal = interpolator(t_delayed)
            pulse_2_in_time = delayed_signal
            # pulse_sum = pulse_1_in_time + pulse_2_in_time
            autocorrelation_value = Autocorrelation.get_detector_signal(pulse_1_in_time, pulse_2_in_time, self.time_domain, intensity_AC=intensity_AC)
            interferometric_autocorrelation_values.append(autocorrelation_value)
                    # --- Progress bar update ---
            if print_statements:
                progress = (i+1) / n_delays
                percent = int(round(progress * 100))
                filled_len = int(bar_length * progress)
                bar = '█' * filled_len + '-' * (bar_length - filled_len)
                sys.stdout.write(f'\rProgress: |{bar}| {percent}%')
                sys.stdout.flush()
        if normalise:
            self.simulated_autocorrelation = 8 * Conversions.Normalise(np.array(interferometric_autocorrelation_values))
            return
        self.simulated_autocorrelation = np.array(interferometric_autocorrelation_values)

    def plot_autocorrelation(self, measured=True, simulated=True):
        plt.figure(figsize=(8, 4))
        if measured:
            if self.measured_autocorrelation.size > 1:
                plt.plot(self.measured_autocorrelation_delays, self.measured_autocorrelation, 'o-', label='Measured AC', color='C0')
        if simulated:
            if self.simulated_autocorrelation.size > 1:
                plt.plot(self.simulated_autocorrelation_delays_fs, self.simulated_autocorrelation, '-', label='Simulated AC', color='C1')
        plt.xlabel('Delay (fs)')
        plt.ylabel('Autocorrelation Signal (a.u.)')
        plt.title('Autocorrelation')
        plt.legend()
        plt.grid(True)
        plt.show()

