from __future__ import print_function
from glob import glob
import threading
import os
import sys

from sklearn.externals import six
from inspect import signature

import numpy as np
import cffi

'''
    Helper class to execute TSNE in separate thread.
'''


class FuncThread(threading.Thread):
    def __init__(self, target, *args):
        threading.Thread.__init__(self)
        self._target = target
        self._args = args

    def run(self):
        self._target(*self._args)

###############################################################################
# From sklearn/base.py, for the __repr__ in MulticoreTSNE


def _pprint(params, offset=0, printer=repr):
    """Pretty print the dictionary 'params'
    Parameters
    ----------
    params : dict
        The dictionary to pretty print
    offset : int
        The offset in characters to add at the begin of each line.
    printer : callable
        The function to convert entries to strings, typically
        the builtin str or repr
    """
    # Do a multi-line justified repr:
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset // 2) * ' '
    for i, (k, v) in enumerate(sorted(six.iteritems(params))):
        if type(v) is float:
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
        else:
            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if (this_line_length + len(this_repr) >= 75 or '\n' in this_repr):
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = ''.join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines


class MulticoreTSNE:
    """
    Compute t-SNE embedding using Barnes-Hut optimization and
    multiple cores (if avaialble).

    Parameters mostly correspond to parameters of `sklearn.manifold.TSNE`.

    The following parameters are unused:
    * n_iter_without_progress
    * min_grad_norm
    * metric
    * method
    
    When `cheat_metric` is true squared equclidean distance is used to build VPTree. 
    Usually leads to same quality, yet much faster.

    Parameter `init` doesn't support 'pca' initialization, but a precomputed
    array can be passed.
    """
    def __init__(self,
                 n_components=2,
                 perplexity=30.0,
                 early_exaggeration=12,
                 learning_rate=200,
                 n_iter=1000,
                 n_iter_without_progress=30,
                 min_grad_norm=1e-07,
                 metric='euclidean',
                 init='random',
                 verbose=0,
                 random_state=None,
                 method='barnes_hut',
                 angle=0.5,
                 n_jobs=1,
                 cheat_metric=True):
        self.n_components = n_components
        self.angle = angle
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.random_state = -1 if random_state is None else random_state
        self.init = init
        self.embedding_ = None
        self.n_iter_ = None
        self.kl_divergence_ = None
        self.verbose = int(verbose)
        self.cheat_metric = cheat_metric
        assert isinstance(init, np.ndarray) or init == 'random', "init must be 'random' or array"
        if isinstance(init, np.ndarray):
            assert init.ndim == 2, "init array must be 2D"
            assert init.shape[1] == n_components, "init array must be of shape (n_instances, n_components)"
            self.init = np.ascontiguousarray(init, float)

        self.ffi = cffi.FFI()
        self.ffi.cdef(
            """void tsne_run_double(double* X, int N, int D, double* Y,
                                    int no_dims, double perplexity, double theta,
                                    int num_threads, int max_iter, int random_state,
                                    bool init_from_Y, int verbose,
                                    double early_exaggeration, double learning_rate,
                                    double *final_error, int distance);""")

        path = os.path.dirname(os.path.realpath(__file__))
        try:
            sofile = (glob(os.path.join(path, 'libtsne*.so')) +
                      glob(os.path.join(path, '*tsne*.dll')))[0]
            self.C = self.ffi.dlopen(os.path.join(path, sofile))
        except (IndexError, OSError):
            raise RuntimeError('Cannot find/open tsne_multicore shared library')

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(),
                                               offset=len(class_name),),)

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, _y=None):

        assert X.ndim == 2, 'X should be 2D array.'

        # X may be modified, make a copy
        X = np.array(X, dtype=float, order='C', copy=True)

        N, D = X.shape
        init_from_Y = isinstance(self.init, np.ndarray)
        if init_from_Y:
            Y = self.init.copy('C')
            assert X.shape[0] == Y.shape[0], "n_instances in init array and X must match"
        else:
            Y = np.zeros((N, self.n_components))

        cffi_X = self.ffi.cast('double*', X.ctypes.data)
        cffi_Y = self.ffi.cast('double*', Y.ctypes.data)
        final_error = np.array(0, dtype=float)
        cffi_final_error = self.ffi.cast('double*', final_error.ctypes.data)

        t = FuncThread(self.C.tsne_run_double,
                       cffi_X, N, D,
                       cffi_Y, self.n_components,
                       self.perplexity, self.angle, self.n_jobs, self.n_iter, self.random_state,
                       init_from_Y, self.verbose, self.early_exaggeration, self.learning_rate,
                       cffi_final_error, int(self.cheat_metric))
        t.daemon = True
        t.start()

        while t.is_alive():
            t.join(timeout=1.0)
            sys.stdout.flush()

        self.embedding_ = Y
        self.kl_divergence_ = final_error
        self.n_iter_ = self.n_iter

        return Y

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self):
        """Get parameters for this object.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            out[key] = value
        return out
