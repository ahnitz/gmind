""" This module contains classes and tools to generate time series and images
containing noise and or signals
"""
import numpy

from pycbc.noise import frequency_noise_from_psd
from scipy.interpolate import interp1d
from pycbc.types import FrequencySeries

class GaussianNoiseGenerator(object):
    def __init__(self, buffer_duration, sample_rate, psd, flow, output_type='frequency', seed=0):
        self.buffer_duration = buffer_duration
        self.sample_rate = sample_rate
        self.current_seed = seed        
        self.type = output_type
        self.flow = flow

        # Make PSD match the duration and sample rate choice
        tlen = int(sample_rate / psd.delta_f)
        flen = tlen / 2 + 1
        psd = psd.copy()

        if flen < len(psd):
            psd.resize(flen)
        else:
            raise ValueError("PSD does not have content at high enough"
                             "frequency for this sample rate")

        flog = numpy.log(psd.sample_frequencies.numpy())
        slog = numpy.log(psd.numpy())
        psd_interp = interp1d(flog, slog)
        kmin = int(flow * buffer_duration)

        nflen = (buffer_duration * sample_rate) / 2 + 1
        fvals = numpy.log(numpy.arange(0, nflen, 1) / float(self.buffer_duration))
        pdata = numpy.exp(psd_interp(fvals))
        pdata[0:kmin] = 0
        self.psd = FrequencySeries(pdata, delta_f=1.0 / self.buffer_duration)

        self.psd[0:kmin].clear()
        self.psd[len(self.psd)-1] = 0

    def frequency(self):
        print self.current_seed
        return frequency_noise_from_psd(self.psd, seed=self.current_seed)
        
    def next(self):
        self.current_seed += 1
        if self.type == 'frequency':
            return self.frequency()
