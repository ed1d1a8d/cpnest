import unittest
import numpy as np
from scipy.fftpack import fft
import cpnest.model

def sin_gaussian(t, params):
    var, freq, amp = params
    return amp * np.sin(t * freq) * np.exp(- t * t / var / 2.0)

def generate_freq_sin_gaussian(t_low, t_high, sample_freq, sg_params):
    times = np.linspace(t_low, t_high, sample_freq * (t_high - t_low))
    t_data = [sin_gaussian(t, sg_params) for t in times]
    f_data = fft(t_data)
    return f_data

def log_gaussian_likelihood(a1, a2, var):
    diff = a1 - a2
    norm = diff.real * diff.real + diff.imag * diff.imag
    ret = -(norm / var + np.log(2 * np.pi * var)) / 2
    return ret.real

class SineGaussianModel(cpnest.model.Model):
    t_low = -1
    t_high = 1
    sample_freq = 100

    names = ['var']
    bounds = [[0, 1]]

    data_params = [0.01, 20, 2] # What should be estimated
    data = generate_freq_sin_gaussian(t_low, t_high, sample_freq, data_params)
    noise = np.full_like(data, 1)

    cnt = 0

    def log_likelihood(self, sample_params_dict):
        sample_params = [sample_params_dict[name] for name in self.names]
        sample_params = sample_params + [20, 2]
        sample_data = generate_freq_sin_gaussian(self.t_low,
                                                 self.t_high,
                                                 self.sample_freq,
                                                 sample_params)
        tot = 0
        for i in range(len(sample_data)):
            tot = tot + log_gaussian_likelihood(self.data[i],
                                                sample_data[i],
                                                self.noise[i])

        # To more easily track progress
        self.cnt = self.cnt + 1
        if self.cnt % 1000 == 0:
            print(self.cnt)

        return tot

class SineGaussianTestCase(unittest.TestCase):
    """
    Test the sine gaussian model
    """
    def setUp(self):
        self.work=cpnest.CPNest(SineGaussianModel(),verbose=2,Nthreads=8,Nlive=500,maxmcmc=1000)

    def test_run(self):
        self.work.run()

def test_all():
    unittest.main(verbosity=2)

if __name__=='__main__':
   unittest.main(verbosity=2)

