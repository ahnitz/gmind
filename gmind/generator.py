""" This module contains classes and tools to generate time series and images
containing noise and or signals
"""
import numpy.random

from pycbc.noise import frequency_noise_from_psd
from scipy.interpolate import interp1d
from pycbc.types import FrequencySeries

from pycbc.inference.option_utils import read_args_from_config
from pycbc.distributions import read_distributions_from_config
from pycbc.transforms import read_transforms_from_config, apply_transforms
from pycbc.inference import prior

from pycbc.workflow import WorkflowConfigParser

from pycbc.waveform import get_fd_waveform
from pycbc.filter import sigma, matched_filter

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
            raise ValueError("PSD does not have content at high enough "
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
        return frequency_noise_from_psd(self.psd, seed=self.current_seed)
        
    def next(self):
        self.current_seed += 1
        if self.type == 'frequency':
            return self.frequency()

class WFParamGenerator(object):
    def __init__(self, config_file, seed=0):
        numpy.random.seed(seed)
        config_file = WorkflowConfigParser(config_file, None)
        var_args, self.static, constraints = read_args_from_config(config_file)
        dist = read_distributions_from_config(config_file)

        self.trans = read_transforms_from_config(config_file)
        self.pval = prior.PriorEvaluator(var_args, *dist, 
                                **{"constraints": constraints})   

    def draw(self):
        return apply_transforms(self.pval.rvs(), self.trans)[0]

# This should have more options for output format and qtile configuration
class GaussianSignalQImageGenerator(object):
    def __init__(self, noise_generator, param_generator, duration, image_dim,
                 whitening_truncation=4, q=10, batch=10):
        self.noise = noise_generator
        self.param = param_generator
        self.image_dim = image_dim
        self.whitening_truncation = whitening_truncation
        self.window = 0.5
        self.q = q
        self.duration = duration
        self.batch = batch

    def next(self):
        images = []
        targets = []
        for i in range(self.batch):
            n = self.noise.next()
            p = self.param.draw()
            hp, _ = get_fd_waveform(p, delta_f=n.delta_f,
                                    f_lower=self.noise.flow, **self.param.static)
            hp.resize(len(n))
            sg = sigma(hp, psd=self.noise.psd, low_frequency_cutoff=self.noise.flow)
            n += hp.cyclic_time_shift(p.tc) / sg * p.snr
            
            msnr = matched_filter(hp, n, psd=self.noise.psd,
                                  low_frequency_cutoff=self.noise.flow)
            snr = abs(msnr.crop(self.whitening_truncation,
                                self.whitening_truncation)).max()

            n = n.to_timeseries()
            w = n.whiten(self.whitening_truncation, self.whitening_truncation)

            dt = self.duration / float(self.image_dim[0])
            fhigh = self.noise.sample_rate * 0.3
            t, f, p = w.qtransform(dt, logfsteps=self.image_dim[1],
                                       frange=(self.noise.flow, fhigh),
                                       qrange=(self.q, self.q),
                                       return_complex=True)

            kmin = int((w.duration / 2 - self.duration / 2) / dt)
            kmax = kmin + int(self.duration / dt)
            p = p[:, kmin:kmax].transpose()
            
            amp = numpy.abs(p)
            p = numpy.stack([p.real, p.imag, amp], axis=2)
            images.append(p)
            targets.append(snr)
        
        return numpy.stack(images, axis=0), numpy.array(targets)







