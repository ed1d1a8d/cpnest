from abc import ABCMeta,abstractmethod,abstractproperty
from numpy import inf
from .parameter import LivePoint
from numpy.random import uniform

class Model(object):
  """
  Base class for user's model. User should subclass this
  and implement log_likelihood, names and bounds
  """
  __metaclass__ = ABCMeta
  names=None # Names of parameters, e.g. ['p1','p2']
  bounds=None # Bounds of prior as list of tuples, e.g. [(min1,max1), (min2,max2), ...]
  def in_bounds(self,param):
    """
    Checks whether param lies within the bounds
    """
    return all(self.bounds[i][0] < param.values[i] < self.bounds[i][1] for i in range(param.dimension))
  
  def new_point(self):
    """
    Create a new LivePoint, drawn from within bounds
    """
    p = LivePoint(self.names,[uniform(self.bounds[i][0],self.bounds[i][1]) for i,_ in enumerate(self.names)] )
    return p
  
  @abstractmethod
  def log_likelihood(self,param):
    """
    returns log likelihood of given parameter
    """
    pass
  def log_prior(self,param):
    """
    Returns log of prior.
    Default is flat prior within bounds
    """
    if self.in_bounds(param):
      return 0.0
    else: return -inf
