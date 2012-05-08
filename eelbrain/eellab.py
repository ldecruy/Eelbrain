'''
The eelbrain.eellab module contains a collection of Eelbrain components that 
are useful for 
interactive data analysis, so for interactive usage you could::

    >>> from eelbrain.eellab import *
    # or
    >>> import eelbrain.eellab as EL


**classes and constructors:**

factor
    standard class for categorial data
var
    standard class for univariate scalar data
var_from_dict
    construct a var from a dictionary

...


**Modules:**

load
    functions for loading various types of data

...




Created on Mar 27, 2012

@author: christian
'''

from vessels.data import (factor, 
                          var,
                          var_from_dict,
                          var_from_apply,
                          ndvar,
                          dataset,
                          combine,
                          )

from vessels import load
from vessels import process

#from analyze.glm import anova, ancova
from analyze import (test,
                     testnd,
                     table) 

import plot
import utils.statfuncs as statf
