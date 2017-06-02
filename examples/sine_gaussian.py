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

params1 = {
    'alpha': 0,
    'phi_0': 0,
    'f_0': 2,
    'q': 5,
    'h_rss': 10
}

params2 = {'alpha': 0, 'phi_0': 0, 'f_0': 2.3069462171112063, 'q': 4.38411564667209, 'h_rss': 10.35212712432972}
params3 = {'alpha': 0, 'phi_0': 0, 'f_0': 2.315103554315639, 'q': 4.1466850872652365, 'h_rss': 9.110012790812448}

fff = 2.4
print(freq_domain_sin_gaussian(fff, params2))
print(freq_domain_sin_gaussian(fff, params3))
print(log_gaussian_likelihood(freq_domain_sin_gaussian(fff, params2), freq_domain_sin_gaussian(fff, params3), 1))
print(log_gaussian_likelihood(1, 2, 1))

'''
x = np.arange(-5, 5, 0.01)
y = [freq_domain_sin_gaussian(f, params1) for f in x]
normalized_y = [np.sqrt(a.real * a.real + a.imag * a.imag).real for a in y]
plt.plot(x, normalized_y)

x = np.arange(-5, 5, 0.01)
y = [freq_domain_sin_gaussian(f, params2) for f in x]
normalized_y = [np.sqrt(a.real * a.real + a.imag * a.imag).real for a in y]
plt.plot(x, normalized_y)

x = np.arange(-5, 5, 0.01)
y = [freq_domain_sin_gaussian(f, params3) for f in x]
normalized_y = [np.sqrt(a.real * a.real + a.imag * a.imag).real for a in y]
plt.plot(x, normalized_y)

plt.show()
'''

'''
def sin_gaussian(t, params):
    var, freq, amp = params
    return amp * np.sin(t * freq) * np.exp(- t * t / var / 2.0)

def generate_freq_sin_gaussian(t_low, t_high, sample_freq, sg_params):
    times = np.linspace(t_low, t_high, sample_freq * (t_high - t_low))
    t_data = [sin_gaussian(t, sg_params) for t in times]
    f_data = fft(t_data)
    return f_data
'''

class SineGaussianModel(cpnest.model.Model):
    freq_low = -5
    freq_high = 5
    freq_step = 0.01

    names = ['f_0', 'q', 'h_rss']
    bounds = [[1, 3], [4, 6], [9, 11]]

    # What cpnest is trying to guess
    data_params = {
        'alpha': 0,
        'phi_0': 0,
        'f_0': 2,
        'q': 5,
        'h_rss': 10
    }

    # We assume noise for each frequency is gaussian.
    # This function returns the variance of the gaussian for a given frequency.
    def noise(self, freq):
        return 1 #same noise for all distributions

    cnt = 0

    def log_likelihood(self, live_point):
        sample_params = self.data_params.copy()
        for name in live_point.names:
            sample_params[name] = live_point[name]

        tot = 0
        freq = self.freq_low

        while freq <= self.freq_high:
            data_val = freq_domain_sin_gaussian(freq, self.data_params)
            sample_val = freq_domain_sin_gaussian(freq, sample_params)
            noise_var = 1

            tot = tot + log_gaussian_likelihood(data_val,
                                                sample_val,
                                                noise_var)
            freq = freq + self.freq_step

        # To more easily track progress
        #self.cnt = self.cnt + 1
        #if self.cnt % 50 == 0:
        #    print(self.cnt, sample_params, tot)

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
