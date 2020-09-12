# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import matplotlib
import sys

import tqdm


def use_inline_backend():
    "Check whether matplotlib is using an inline backend, e.g. for notebooks"
    # mpl.get_backend() sets backend and imports pyplot; avoid that unless
    # pyplot has already been imported (it is there after a % matplotlib
    # inline call)
    if 'matplotlib.pyplot' in sys.modules:
        backend = matplotlib.get_backend()
        return backend.endswith('inline') or backend == 'nbAgg'


# import inline tqdm
if use_inline_backend():
    try:
        import ipywidgets as _
    except ImportError:
        pass
    else:
        import tqdm.auto as tqdm
