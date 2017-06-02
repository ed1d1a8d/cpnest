import sys
import os
import numpy as np
from math import log
from collections import deque
from random import random,randrange

from . import parameter
from . import proposal

class Sampler(object):
    """
    Sampler class.
    Initialisation arguments:

    usermodel:
    user defined model to sample

    maxmcmc:
    maximum number of mcmc steps to be used in the sampler

    verbose:
    display debug information on screen
    default: False

    poolsize:
    number of objects for the affine invariant sampling
    default: 1000
    """
    def __init__(self,usermodel,maxmcmc,verbose=False,poolsize=1000):
        self.user = usermodel
        self.maxmcmc = maxmcmc
        self.Nmcmc = maxmcmc
        self.Nmcmc_exact = float(maxmcmc)
        self.proposals = proposal.DefaultProposalCycle()
        self.poolsize = poolsize
        self.evolution_points = deque(maxlen=self.poolsize + 1) # +1 for the point to evolve
        self.verbose=verbose
        self.acceptance=0.0
        self.initialised=False

    def reset(self):
        """
        Initialise the sampler
        """
        for n in range(self.poolsize):
          while True:
            if self.verbose > 2: sys.stderr.write("process {0!s} --> generating pool of {1:d} points for evolution --> {2:.0f} % complete\r".format(os.getpid(), self.poolsize, 100.0*float(n+1)/float(self.poolsize)))
            p = self.user.new_point()
            p.logP = self.user.log_prior(p)
            if np.isfinite(p.logP): break
          p.logL=self.user.log_likelihood(p)
          self.evolution_points.append(p)
        if self.verbose > 2: sys.stderr.write("\n")
        self.proposals.set_ensemble(self.evolution_points)

        processed_points = 0
        for _ in range(len(self.evolution_points)):
          s = self.evolution_points.popleft()
          acceptance,jumps,s = self.metropolis_hastings(s,-np.inf,self.Nmcmc)
          self.evolution_points.append(s)

          processed_points = processed_points + 1
          if self.verbose > 3:
            sys.stderr.write("process {0!s} --> running metropolis hastings on pool of {1:d} points for evolution --> {2:.0f} % complete\n".format(os.getpid(), self.poolsize, 100.0*float(processed_points)/float(self.poolsize)))
        if self.verbose > 3: sys.stderr.write("\n")
        self.proposals.set_ensemble(self.evolution_points)
        self.initialised=True

    def estimate_nmcmc(self, safety=5, tau=None):
        """
        Estimate autocorrelation length of chain using acceptance fraction
        ACL = (2/acc) - 1
        multiplied by a safety margin of 5
        Uses moving average with decay time tau iterations (default: self.poolsize)
        Taken from W. Farr's github.com/farr/Ensemble.jl
        """
        if tau is None: tau = self.poolsize
        else: tau /= float(safety)
        if self.acceptance==0:
            self.Nmcmc_exact = (1.0 + 1.0/tau)*self.Nmcmc_exact
        else:
            self.Nmcmc_exact = (1.0 - 1.0/tau)*self.Nmcmc_exact + (safety/tau)*(2.0/self.acceptance - 1.0)
        if np.isfinite(self.Nmcmc_exact):
            self.Nmcmc = max(safety,min(self.maxmcmc, int(round(self.Nmcmc_exact))))
        else:
            self.Nmcmc = max(safety,self.maxmcmc)
        return self.Nmcmc

    def produce_sample(self, queue, logLmin, seed, ip, port, authkey):
        """
        main loop that generates samples and puts them in the queue for the nested sampler object
        """
        if not self.initialised:
          self.reset()
        # Prevent process from zombification if consumer thread exits
        queue.cancel_join_thread()
        self.seed = seed
        np.random.seed(seed=self.seed)
        self.counter=0
        while(1):
            # Pick a random point from the ensemble to start with
            # Pop it out the stack to prevent cloning
            param = self.evolution_points.popleft()
            if logLmin.value==np.inf:
                break
            acceptance,jumps,outParam = self.metropolis_hastings(param,logLmin.value,self.Nmcmc)
            # Put sample back in the stack
            self.evolution_points.append(outParam.copy())
            # If we bailed out then flag point as unusable
            if acceptance==0.0:
                outParam.logL=-np.inf
            # Push the sample onto the queue
            queue.put((acceptance,jumps,outParam))
            # Update the ensemble every now and again
            if (self.counter%(self.poolsize/10))==0 or self.acceptance<0.001:
                self.proposals.set_ensemble(self.evolution_points)
            self.counter += 1
        sys.stderr.write("Sampler process {0!s}, exiting\n".format(os.getpid()))
        return 0

    def metropolis_hastings(self,inParam,logLmin,nsteps):
        """
        metropolis-hastings loop to generate the new live point taking nmcmc steps
        """
        accepted = 0
        rejected = 0
        jumps = 0
        oldparam=inParam
        logp_old = self.user.log_prior(oldparam)
        while (jumps < nsteps):
            if self.verbose > 3:
                sys.stderr.write("process {0!s} --> running metropolis hastings for single point for {1:d} steps --> {2:.0f} % complete\r".format(os.getpid(), nsteps, 100.0*float(jumps)/float(nsteps)))

            newparam = self.proposals.get_sample(oldparam.copy())
            newparam.logP = self.user.log_prior(newparam)
            if newparam.logP-logp_old + self.proposals.log_J > log(random()):
                newparam.logL = self.user.log_likelihood(newparam)
                if newparam.logL > logLmin:
                  oldparam=newparam
                  logp_old = newparam.logP
                  accepted+=1
                else:
                  rejected+=1
            else:
                rejected+=1

            jumps+=1
            if jumps==10*self.maxmcmc:
              print('Warning, MCMC chain exceeded {0} iterations!'.format(10*self.maxmcmc))
        self.acceptance = float(accepted)/float(jumps)
        self.estimate_nmcmc(tau=nsteps)
        return (float(accepted)/float(rejected+accepted),jumps,oldparam)
