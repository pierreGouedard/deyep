# Global import
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import unit_impulse
from scipy.fftpack import fft, ifft, fftshift, fftfreq

# Local import


def sine_wave(f, overSampRate, nCyl):
    """
    Generate sine wave signal with the following parameters
        Parameters:
            f : frequency of sine wave in Hertz
            overSampRate : oversampling rate (integer)
            phase : desired phase shift in radians
            nCyl : number of cycles of sine wave to generate
        Returns:
            (t,g) : time base (t) and the signal g(t) as tuple
        Example:
            f=10; overSampRate=30;
            phase = 1/3*np.pi;nCyl = 5;
            (t,g) = sine_wave(f,overSampRate,phase,nCyl)
    """
    fs = overSampRate*f # sampling frequency
    t = np.arange(0, nCyl*1/f-1/fs, 1/fs) # time base
    g = np.sin(2*np.pi*f*t) # replace with cos if a cosine wave is desired
    return t, g


def dirac_comb(f, overSampRate, nCyl):
    """
    Generate sine wave signal with the following parameters
        Parameters:
            f : frequency of sine wave in Hertz
            overSampRate : oversampling rate (integer)
            phase : desired phase shift in radians
            nCyl : number of cycles of sine wave to generate
        Returns:
            (t,g) : time base (t) and the signal g(t) as tuple
        Example:
            f=10; overSampRate=30;
            phase = 1/3*np.pi;nCyl = 5;
            (t,g) = sine_wave(f,overSampRate,phase,nCyl)
    """
    fs = overSampRate * f # sampling frequency
    t = np.arange(0, nCyl * 1/f, 1/fs) # time base
    g = np.hstack([unit_impulse(overSampRate) for _ in range(nCyl)]) # replace with cos if a cosine wave is desired
    return t, g


if __name__ == '__main__':
    """
    Usage:
    python deyep/scripts/sp.py

    """
    # Parameters
    overSampRate = 100
    f = 100
    nCyl = 10000

    # Plot sinewave
    (t, sin_x) = sine_wave(f, overSampRate, nCyl)  # function call

    plt.plot(t, sin_x)  # plot using pyplot library from matplotlib package
    plt.title('Sine wave f=' + str(f) + ' Hz')  # plot title
    plt.xlabel('Time (s)')  # x-axis label
    plt.ylabel('Amplitude')  # y-axis label
    plt.show()  # display the figure

    # Plot dirac comb
    (t, dirac_x) = dirac_comb(f, overSampRate, nCyl)  # function call

    plt.plot(t, dirac_x)  # plot using pyplot library from matplotlib package
    plt.title('Dirac comb f=' + str(f) + ' Hz')  # plot title
    plt.xlabel('Time (s)')  # x-axis label
    plt.ylabel('Amplitude')  # y-axis label
    plt.show()  # display the figure

    # FFT basics with plot
    NFFT = 1024 * 2 * 2
    sin_X = fft(sin_x, NFFT)
    dirac_X = fft(dirac_x, NFFT)

    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure handle
    nVals = np.arange(start=0, stop=NFFT)  # raw index for FFT plot
    ax.plot(nVals, np.abs(sin_X))
    ax.set_title('Double Sided FFT - without FFTShift - sin wave')
    ax.set_xlabel('Sample points (N-point DFT)')
    ax.set_ylabel('DFT Values')

    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure handle
    nVals = np.arange(start=0, stop=NFFT)  # raw index for FFT plot
    ax.plot(nVals, np.abs(dirac_X))
    ax.set_title('Double Sided FFT - without FFTShift - dirac comb')
    ax.set_xlabel('Sample points (N-point DFT)')
    ax.set_ylabel('DFT Values')

    plt.show()

    # ABsolute freqs ° freq centered
    sin_X = fftshift(fft(sin_x, NFFT))  # compute DFT using FFT
    dirac_X = fftshift(fft(dirac_x, NFFT))  # compute DFT using FFT

    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure handle
    fVals = np.arange(start=-NFFT / 2, stop=NFFT / 2) * overSampRate * f / NFFT
    ax.plot(fVals, np.abs(sin_X), 'b')
    ax.set_title('Double Sided FFT - with FFTShift - sin wave')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('|DFT Values|')
    ax.set_xlim(-500, 500)
    ax.set_xticks(np.arange(-500, 500 + 10, 100))

    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure handle
    fVals = np.arange(start=-NFFT / 2, stop=NFFT / 2) * overSampRate * f / NFFT
    ax.plot(fVals, np.abs(dirac_X), 'b')
    ax.set_title('Double Sided FFT - with FFTShift - dirac comb')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('|DFT Values|')
    ax.set_xlim(-500, 500)
    ax.set_xticks(np.arange(-500, 500 + 10, 100))

    plt.show()
    import IPython
    IPython.embed()