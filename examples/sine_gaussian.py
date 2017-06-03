import cpnest.model
import matplotlib.pyplot as plt
import numpy as np
import unittest

# returns amplitude of sin gaussian defined by params at freq
# corresponds to h_{+}(freq) in equation (18) of LIGO-T1400734-0
def freq_domain_sin_gaussian(freq, params):
    alpha = params['alpha']
    phi_0 = params['phi_0']
    f_0 = params['f_0']
    q = params['q']
    h_rss = params['h_rss']

    tau = q / np.sqrt(2.0) / np.pi / f_0
    f_minus = freq - f_0
    rootpi = np.sqrt(np.pi)

    rt = q / 4.0 / rootpi / f_0 * (1 + np.cos(2 * phi_0) * np.exp(-q * q))
    exp = -f_minus * f_minus * np.pi * np.pi * tau * tau + 1j * phi_0
    ret = np.cos(alpha) * np.exp(exp) * h_rss * rootpi * tau / 2.0 / np.sqrt(rt)

    return ret

# returns p(a2),
# where p is the pdf function of a gaussian centered at a1 with variance var
def log_gaussian_likelihood(a1, a2, var):
    diff = a1 - a2
    norm = diff.real * diff.real + diff.imag * diff.imag
    ret = -(norm / var + np.log(2 * np.pi * var)) / 2.0
    return ret.real

class SineGaussianModel(cpnest.model.Model):

    # What cpnest is trying to guess
    data_params = {
        'alpha': 0,
        'phi_0': 0,
        'f_0': 2,
        'q': 5,
        'h_rss': 10
    }

    names = ['f_0', 'q', 'h_rss']
    bounds = [[1, 3], [4, 6], [9, 11]]

    freq_low = 0
    freq_high = 4
    freq_step = 0.01

    # We assume noise for each frequency is gaussian.
    # This function returns the variance of the gaussian for a given frequency.
    def noise_variance(self, freq):
        return 0.02 #same noise for all distributions

    # returns a random value from a complex gaussian with variance defined by
    # function noise_variance
    def noise(self, freq):
        var = self.noise_variance(freq)
        real = np.random.normal(scale=np.sqrt(var / 2.0))
        imag = np.random.normal(scale=np.sqrt(var / 2.0))
        ret = real + 1j * imag
        return ret

    # need init function so can use class variables in list comprehension
    def __init__(self):
        np.random.seed(12345)

        self.freqs = np.arange(self.freq_low, self.freq_high, self.freq_step)

        self.data = [freq_domain_sin_gaussian(freq, self.data_params)
                     + self.noise(freq)
                     for freq in self.freqs]

        magnitude_data = [np.sqrt(a.real * a.real + a.imag * a.imag).real
                          for a in self.data]
        plt.plot(self.freqs, magnitude_data)
        plt.savefig('sine-gaussian-data.png')

    def log_likelihood(self, live_point):
        sample_params = self.data_params.copy()
        for name in live_point.names:
            sample_params[name] = live_point[name]

        tot = 0
        freq = self.freq_low
        for i in range(len(self.freqs)):
            freq = self.freqs[i]
            data_val = self.data[i]
            sample_val = freq_domain_sin_gaussian(freq, sample_params)
            noise_var = self.noise_variance(freq)

            tot = tot + log_gaussian_likelihood(data_val,
                                                sample_val,
                                                noise_var)

        return tot

class SineGaussianTestCase(unittest.TestCase):
    """
    Test the sine gaussian model
    """
    def setUp(self):
        self.work=cpnest.CPNest(SineGaussianModel(),verbose=4,Nthreads=8,Nlive=100,maxmcmc=1000)

    def test_run(self):
        self.work.run()

def test_all():
    unittest.main(verbosity=4)

if __name__=='__main__':
   unittest.main(verbosity=4)
