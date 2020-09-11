"""Boosting as described by David et al. (2007).

Versions
--------
Stored in ``algorithm_version`` attribute

 -1. Prior to storing version
 0. Normalize ``x`` after applying basis

Profiling
---------
ds = datasets._get_continuous()
y = ds['y']
x1 = ds['x1']
x2 = ds['x2']

%prun -s cumulative res = boosting(y, x1, 0, 1)

"""
from dataclasses import dataclass, field
import inspect
from itertools import chain, product, repeat
from multiprocessing.sharedctypes import RawArray
import os
import time
from threading import Event, Thread
from typing import Any, Union, Tuple, Sequence
import warnings

import numpy as np
from tqdm.auto import tqdm

from .._config import CONFIG, mpc
from .._data_obj import Dataset, NDVar, NDVarArg, dataobj_repr
from .._exceptions import OldVersionError
from .._ndvar import convolve_jit
from .._utils import LazyProperty, PickleableDataClass, user_activity
from ._boosting_opt import l1, l2, generate_options, update_error
from .shared import RevCorrData, Split, Splits, merge_segments
from ._fit_metrics import get_evaluators


# process messages
JOB_TERMINATE = -1

# error functions
ERROR_FUNC = {'l2': l2, 'l1': l1}
DELTA_ERROR_FUNC = {'l2': 2, 'l1': 1}


@dataclass(eq=False)
class BoostingResult(PickleableDataClass):
    """Result from boosting a temporal response function

    Attributes
    ----------
    h : NDVar | tuple of NDVar
        The temporal response function. Whether ``h`` is an NDVar or a tuple of
        NDVars depends on whether the ``x`` parameter to :func:`boosting` was
        an NDVar or a sequence of NDVars.
    h_scaled : NDVar | tuple of NDVar
        ``h`` scaled such that it applies to the original input ``y`` and ``x``.
        If boosting was done with ``scale_data=False``, ``h_scaled`` is the same
        as ``h``.
    h_source : NDVar | tuple of NDVar
        If ``h`` was constructed using a basis, ``h_source`` represents the
        source of ``h`` before being convolved with the basis.
    h_time : UTS
        Time dimension of the kernel.
    r : float | NDVar
        Correlation between the measured response and the response predicted
        with ``h``. Type depends on the ``y`` parameter to :func:`boosting`. For
        vector data, measured and predicted responses are normalized, and ``r``
        is computed as the average dot product over time.
    r_rank : float | NDVar
        As ``r``, the Spearman rank correlation.
    t_run : float
        Time it took to run the boosting algorithm (in seconds).
    error : str
        The error evaluation method used.
    residual : float | NDVar
        The residual of the final result

        - ``error='l1'``: the sum of the absolute differences between ``y`` and
          ``h * x``.
        - ``error='l2'``: the sum of the squared differences between ``y`` and
          ``h * x``.

        For vector ``y``, the error is defined based on the distance in space
        for each data point.
    delta : scalar
        Kernel modification step used.
    mindelta : None | scalar
        Mindelta parameter used.
    n_samples : int
        Number of samples in the input data time axis.
    scale_data : bool
        Scale_data parameter used.
    y_mean : NDVar | scalar
        Mean that was subtracted from ``y``.
    y_scale : NDVar | scalar
        Scale by which ``y`` was divided.
    x_mean : NDVar | scalar | tuple
        Mean that was subtracted from ``x``.
    x_scale : NDVar | scalar | tuple
        Scale by which ``x`` was divided.
    splits : Splits
        Data splits used for cross-validation.
    algorithm_version : int
        Version of the algorithm with which the model was estimated; ``-1`` for
        results from before this attribute was added.
    """
    # basic parameters
    y: str
    x: Union[str, Tuple[str]]
    tstart: Union[float, Tuple[float, ...]]
    tstop: Union[float, Tuple[float, ...]]
    scale_data: bool
    delta: float
    mindelta: float
    error: str
    selective_stopping: int
    # data properties
    y_mean: NDVar
    y_scale: NDVar
    x_mean: Union[NDVar, Tuple[NDVar, ...]]
    x_scale: Union[NDVar, Tuple[NDVar, ...]]
    # results
    _h: Union[NDVar, Tuple[NDVar, ...]]
    _isnan: np.ndarray
    t_run: float
    # advanced parameters
    basis: float
    basis_window: str
    splits: Splits
    # advanced data properties
    n_samples: int = None
    _y_info: dict = field(default_factory=dict)
    # fit metrics
    i_test: int = None  # test partition for fit metrics
    residual: NDVar = None
    r: Union[float, NDVar] = None
    r_rank: Union[float, NDVar] = None
    r_l1: NDVar = None
    # store the version of the boosting algorithm with which model was fit
    version: int = 13  # file format (updates when re-saving)
    algorithm_version: int = -1  # does not change when re'saving
    # debug parameters
    y_pred: NDVar = None
    fit: Any = None  # scanpydoc can't handle undocumented 'Boosting'
    # legacy attributes
    prefit: str = None

    def __post_init__(self):
        self.partitions = self.splits.n_partitions

    def __setstate__(self, state: dict):
        # backwards compatibility
        version = state.pop('version')
        if version < self.version:
            if version == 7:
                state['partitions'] = state.pop('n_partitions')
                state['partitions_arg'] = state.pop('n_partitions_arg')
            if version < 9:
                state['residual'] = state.pop('fit_error')
            if version < 11:
                for key in ['partitions_arg', 'h', 'isnan', 'y_info']:
                    state[f'_{key}'] = state.pop(key, None)
                state['r_rank'] = state.pop('spearmanr')
            if version < 12:
                # Vector residuals are averaged
                if state['r_rank'] is None:
                    state['residual'] *= state['n_samples']
            if version < 13:
                state['splits'] = Splits(None, state.pop('_partitions_arg'), state.pop('partitions'), state.pop('validate', 1), state.pop('test', 0), state.pop('model'))
        self.__init__(**state)

    def __repr__(self):
        items = []
        if isinstance(self.tstart, tuple):
            x = ' + '.join(f'{x} ({t1:g} - {t2:g})' for x, t1, t2 in zip(self.x, self.tstart, self.tstop))
        else:
            if self.x is None or isinstance(self.x, str):
                x = self.x
            else:
                x = ' + '.join(map(str, self.x))
            items.append(f'{self.tstart:g} - {self.tstop:g}')
        items.insert(0, f'boosting {self.y} ~ {x}')
        for name, param in inspect.signature(boosting).parameters.items():
            if param.default is inspect.Signature.empty or name == 'ds':
                continue
            elif name == 'debug':
                continue
            elif name == 'partitions':
                value = self.splits.partitions_arg
            elif name == 'model':
                if self.splits.model is None:
                    continue
                value = dataobj_repr(self.splits.model)
            elif name == 'validate':
                value = self.splits.n_validate
            elif name == 'test':
                value = self.splits.n_test
            else:
                value = getattr(self, name)
            if value != param.default and value is not None:
                items.append(f'{name}={value}')
        return f"<{', '.join(items)}>"

    @LazyProperty
    def h(self):
        if not self.basis:
            return self._h
        elif isinstance(self._h, tuple):
            return tuple(h.smooth('time', self.basis, self.basis_window, 'full') for h in self._h)
        else:
            return self._h.smooth('time', self.basis, self.basis_window, 'full')

    @LazyProperty
    def h_scaled(self):
        if self.y_scale is None:
            return self.h
        elif isinstance(self.h, NDVar):
            out = self.h * (self.y_scale / self.x_scale)
            out.info = self._y_info.copy()
            return out
        else:
            out = []
            for h, sx in zip(self.h, self.x_scale):
                h = h * (self.y_scale / sx)
                h.info = self._y_info.copy()
                out.append(h)
            return tuple(out)

    @LazyProperty
    def h_source(self):
        return self._h

    @LazyProperty
    def h_time(self):
        if isinstance(self.h, NDVar):
            return self.h.time
        else:
            return self.h[0].time

    @LazyProperty
    def _variability(self):
        # variability in the data
        if self.y_scale is None:
            raise NotImplementedError("Not implemented for scale_data=False")
        elif self.n_samples is None:
            raise OldVersionError("This is an older result object which did not store some necessary information; refit the model to use this attribute")
        else:
            # Due to the normalization:
            return self.n_samples

    @LazyProperty
    def proportion_explained(self):
        return 1 - (self.residual / self._variability)

    def _set_parc(self, parc):
        """Change the parcellation of source-space result
         
        Notes
        -----
        No warning for missing sources!
        """
        from .._ndvar import set_parc

        if not self.r.has_dim('source'):
            raise RuntimeError('BoostingResult does not have source-space data')

        def sub_func(obj):
            if obj is None:
                return None
            elif isinstance(obj, tuple):
                return tuple(sub_func(obj_) for obj_ in obj)
            obj_new = set_parc(obj, parc)
            index = np.invert(obj_new.source.parc.startswith('unknown-'))
            return obj_new.sub(source=index)

        for attr in ('h', 'r', 'r_rank', 'residual', 'y_mean', 'y_scale'):
            setattr(self, attr, sub_func(getattr(self, attr)))


class SplitResult:
    __slots__ = ('split', 'h', 'h_failed')

    def __init__(self, split: Split, n_y: int, n_x: int, n_times_h: int):
        self.split = split
        self.h = np.empty((n_y, n_x, n_times_h), np.float64)
        self.h_failed = np.zeros(n_y, bool)

    def add_h(self, i_y: int, h: Union[np.ndarray, None]):
        if h is None:
            self.h_failed[i_y] = True
            self.h[i_y] = 0
        else:
            self.h[i_y] = h

    def h_with_nan(self):
        "Set failed TRFs to NaN"
        if np.any(self.h_failed):
            h = self.h.copy()
            h[self.h_failed] = np.nan
            return h
        return self.h


class Boosting:
    """Object-oriented API for boosting

    Examples
    --------
    Standard usage of the object-oriented API for a model with
    cross-validation, comparable to the :func:`boosting` function::

        data = RevCorrData(y, x, ds)
        data.apply_basis(0.05, 'hamming'')
        data.normalize('l1')
        data.initialize_cross_validation(5, test=1)
        model = Boosting(data)
        model.fit(0, 0.500, selective_stopping=1, error='l1')
        result = model.evaluate_fit()
    """
    # fit parameters
    tstart = None
    tstart_h = None
    tstop = None
    selective_stopping = None
    error = None
    delta = None
    mindelta = None
    # timing
    t_fit_start = None
    t_fit_done = None
    # fit result
    _i_start = None
    split_results = None
    n_skip = 0
    # eval result
    y_pred = None

    def __init__(
            self,
            data: RevCorrData,
    ):
        self.data = data

    def fit(
            self,
            tstart: Union[float, Sequence[float]],
            tstop: Union[float, Sequence[float]],
            selective_stopping: int = 1,
            error: str = 'l1',
            delta: float = 0.005,  # coordinate search step
            mindelta: float = None,  # narrow search by reducing delta until reaching mindelta
    ):
        self.data._check_data()
        assert error in ERROR_FUNC
        mindelta_ = delta if mindelta is None else mindelta
        self.selective_stopping = selective_stopping
        self.error = error
        self.delta = delta
        self.mindelta = mindelta
        n_y = len(self.data.y)
        n_x = len(self.data.x)
        # find TRF start/stop for each x
        if isinstance(tstart, (tuple, list, np.ndarray)):
            if len(tstart) != len(self.data.x_name):
                raise ValueError(f'tstart={tstart!r}: len(tstart) ({len(tstart)}) is different from len(x) ({len(self.data.x_name)}')
            elif len(tstart) != len(tstop):
                raise ValueError(f'tstop={tstop!r}: mismatched len(tstart) = {len(tstart)}, len(tstop) = {len(tstop)}')
            self.tstart = tuple(tstart)
            self.tstart_h = min(self.tstart)
            self.tstop = tuple(tstop)
            n_xs = [1 if dim is None else len(dim) for _, dim, _ in self.data._x_meta]
            tstart = [t for t, n in zip(tstart, n_xs) for _ in range(n)]
            tstop = [t for t, n in zip(tstop, n_xs) for _ in range(n)]
        else:
            self.tstart = self.tstart_h = tstart
            self.tstop = tstop
            tstart = [tstart] * n_x
            tstop = [tstop] * n_x

        # TRF extent in indices
        tstep = self.data.time.tstep
        i_start_by_x = np.asarray([int(round(t / tstep)) for t in tstart], np.int64)
        i_stop_by_x = np.asarray([int(round(t / tstep)) for t in tstop], np.int64)
        self._i_start = i_start = np.min(i_start_by_x)
        i_stop = np.max(i_stop_by_x)
        h_n_times = i_stop - i_start

        if len(self.data.segments) == 1:
            self.n_skip = h_n_times - 1

        # progress bar
        n_splits = len(self.data.splits.splits)
        pbar = tqdm(desc=f"Fitting models", total=n_y * n_splits, disable=CONFIG['tqdm'])
        self.t_fit_start = time.time()

        # boosting
        split_results = [SplitResult(split, n_y, n_x, h_n_times) for split in self.data.splits.splits]
        if CONFIG['n_workers']:
            # Make sure cross-validations are added in the same order, otherwise
            # slight numerical differences can occur
            job_queue, result_queue = setup_workers(self.data, i_start_by_x, i_stop_by_x, delta, mindelta_, error, selective_stopping)
            stop_jobs = Event()
            thread = Thread(target=put_jobs, args=(job_queue, n_y, n_splits, stop_jobs))
            thread.start()
            # collect results
            try:
                for _ in range(n_y * n_splits):
                    i_y, i_split, h = result_queue.get()
                    split_results[i_split].add_h(i_y, h)
                    pbar.update()
            except KeyboardInterrupt:
                stop_jobs.set()
                raise
        else:
            for i_y, y_i in enumerate(self.data.y):
                for split in split_results:
                    h = boost(y_i, self.data.x, self.data.x_pads, split.split, i_start_by_x, i_stop_by_x, delta, mindelta_, error, selective_stopping)
                    split.add_h(i_y, h)
                    pbar.update()
        self.split_results = split_results
        pbar.close()
        self.t_fit_done = time.time()

    def _get_i_tests(self):
        assert self.data.splits.n_test
        return sorted({split.split.i_test for split in self.split_results})

    def _get_h(
            self,
            skip_failed: bool = False,
            i_test: int = None,
    ):
        if i_test is None:
            split_results = self.split_results
            segments = self.data.segments
        else:
            split_results = [split for split in self.split_results if split.split.i_test == i_test]
            segments = split_results[0].split.test

        if skip_failed:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)
                h = np.nanmean([split.h_with_nan() for split in split_results], 0)
            is_nan = np.isnan(h[:, :, 0])
            h[is_nan] = 0
        else:
            h = np.mean([split.h for split in split_results])
        return h, segments

    def _get_h_failed(self, i_test: int = None) -> np.ndarray:
        if i_test is None:
            split_results = self.split_results
        else:
            split_results = [split for split in self.split_results if split.split.i_test == i_test]
        out = np.all([split.h_failed for split in split_results], 0)
        if self.data.vector_dim:
            out = np.all(out.reshape((len(self.data.vector_dim), -1)), 0)
        return out

    def evaluate_fit(
            self,
            i_test: int = None,
            metrics: Sequence[str] = None,
            cross_fit: bool = None,
            debug: bool = False,
    ):
        """Compute average TRF and fit metrics

        Parameters
        ----------
        i_test
            Test partition index (only applies for models fit with
            cross-validation). By default, ``y`` in each test segment is
            predicted from the averaged model from the corresponding training
            sets, and all test segments are pooled for computing fit metrics.
            Set ``i_test`` with an integer to compute fit metrics for a single
            test partition.
        metrics
            Which model fit metrics to compute (default depends on the data).
            If no metrics are needed, ``metrics=()`` is faster.
        cross_fit
            Compute fit metrics from cross-validation (only applies to model
            with cross-validation; default ``True``).
        debug
            Add additional attributes to the returned result.
        """
        if cross_fit is None:
            cross_fit = bool(self.data.splits.n_test)
        elif cross_fit and not self.data.splits.n_test:
            raise ValueError(f"cross_fit={cross_fit!r} for model without cross-validation")

        # fit evaluation
        if metrics is None:
            if self.data.vector_dim:
                metrics = [f'vec-{self.error}', f'vec-corr']
                if self.error == 'l1':
                    metrics.append('vec-corr-l1')
            else:
                metrics = [self.error, 'r', 'r_rank']

        # test sets to use
        if cross_fit:
            if i_test is None:
                i_tests = self._get_i_tests()
            else:
                i_tests = [i_test]
        elif i_test is not None:
            raise ValueError(f"i_test={i_test!r} without cross_fit")
        else:
            i_tests = None

        # hs: [(h, segments), ...]
        if cross_fit:
            hs = [self._get_h(True, i) for i in i_tests]
            all_segments = np.sort(np.vstack([segments for _, segments in hs]), 0)
            eval_segments = merge_segments(all_segments, True)
        else:
            hs = [self._get_h(True)]
            eval_segments = merge_segments(self.data.segments, True)
            if self.n_skip:
                eval_segments[:, 0] += self.n_skip
                # check for invalid segments (negative duration)
                valid = eval_segments[:, 1] - eval_segments[:, 0] > 0
                if not np.all(valid):
                    eval_segments = eval_segments[valid]

        if metrics:
            # y dimensions
            n_y = len(self.data.y)
            if self.data.vector_dim:
                n_vec = len(self.data.vector_dim)
                n_vecs = n_y // n_vec
            else:
                n_vec = n_vecs = 0

            # predicted y
            if debug:
                y_pred_iter = y_pred = np.empty(self.data.y.shape)
            elif n_vecs:
                y_pred = np.empty((n_vec, *self.data.y.shape[1:]))
                y_pred_iter = chain.from_iterable(repeat(tuple(y_pred), n_vecs))
            else:
                y_pred = np.empty(self.data.y.shape[1:])
                y_pred_iter = repeat(y_pred, n_y)

            evaluators, evaluators_s, evaluators_v = get_evaluators(metrics, self.data)

            # fit and evaluate each y
            for i_y, y_pred_i in enumerate(y_pred_iter):
                # for cross-validation, different segments are predicted by different h:
                for h, segments in hs:
                    convolve(h[i_y], self.data.x, self.data.x_pads, self._i_start, segments, y_pred_i)

                if evaluators_s:
                    for e in evaluators_s:
                        e.add_y(i_y, self.data.y[i_y], y_pred_i, eval_segments)

                if evaluators_v and i_y % n_vec == n_vec - 1:
                    i_vec = i_y // n_vec
                    i_y_vec = slice(i_y-n_vec+1, i_y+1)
                    y_pred_i_vec = y_pred[i_y_vec] if debug else y_pred
                    for e in evaluators_v:
                        e.add_y(i_vec, self.data.y[i_y_vec], y_pred_i_vec, eval_segments)

            # Package evaluators
            evaluations = {e.attr: self.data.package_value(e.x, e.name, meas=e.meas) for e in evaluators}
            if debug:
                evaluations['y_pred'] = self.data.package_y_like(y_pred, 'y-pred')
        else:
            evaluations = {}

        # package h
        h_xs = [h for h, _ in hs]
        h_x = h_xs[0] if len(h_xs) == 1 else np.mean(h_xs, 0)
        h = self.data.package_kernel(h_x, self.tstart_h)
        # package model parameters
        y_mean, y_scale, x_mean, x_scale = self.data.data_scale_ndvars()
        if debug:
            evaluations['fit'] = self
        t_run = self.t_fit_done - self.t_fit_start
        return BoostingResult(
            # basic parameters
            self.data.y_name, self.data.x_name, self.tstart, self.tstop, bool(self.data.scale_data), self.delta, self.mindelta, self.error, self.selective_stopping,
            # data properties
            y_mean, y_scale, x_mean, x_scale,
            # results
            h, self._get_h_failed(), t_run,
            # advanced parameters
            self.data.basis, self.data.basis_window, self.data.splits,
            # advanced data properties
            self.data.y.shape[1], self.data.y_info,
            algorithm_version=0,
            i_test=i_test, **evaluations)


@user_activity
def boosting(
        y: NDVarArg,
        x: Union[NDVarArg, Sequence[NDVarArg]],
        tstart: Union[float, Sequence[float]],
        tstop: Union[float, Sequence[float]],
        scale_data: Union[bool, str] = True,  # normalize y and x; can be 'inplace'
        delta: float = 0.005,  # coordinate search step
        mindelta: float = None,  # narrow search by reducing delta until reaching mindelta
        error: str = 'l2',  # 'l1' | 'l2', for scaling data
        basis: float = 0,
        basis_window: str = 'hamming',
        partitions: int = None,  # Number of partitionings for cross-validation
        model: str = None,
        validate: int = 1,  # Number of segments in validation set
        test: int = 0,  # Number of segments in test set
        ds: Dataset = None,
        selective_stopping: int = 0,
        debug: bool = False,
):
    """Estimate a linear filter with coordinate descent

    Parameters
    ----------
    y : NDVar
        Signal to predict.
    x : NDVar | sequence of NDVar
        Signal to use to predict ``y``. Can be sequence of NDVars to include
        multiple predictors. Time dimension must correspond to ``y``.
    tstart : scalar | sequence of scalar
        Start of the TRF in seconds. A list can be used to specify different
        values for each item in ``x``.
    tstop : scalar | sequence of scalar
        Stop of the TRF in seconds. Format must match ``tstart``.
    scale_data : bool | 'inplace'
        Scale ``y`` and ``x`` before boosting: subtract the mean and divide by
        the standard deviation (when ``error='l2'``) or the mean absolute
        value (when ``error='l1'``). Use ``'inplace'`` to save memory by scaling
        the original objects specified as ``y`` and ``x`` instead of making a 
        copy.
    delta : scalar
        Step for changes in the kernel.
    mindelta : scalar
        If the error for the training data can't be reduced, divide ``delta``
        in half until ``delta < mindelta``. The default is ``mindelta = delta``,
        i.e. ``delta`` is constant.
    error : 'l2' | 'l1'
        Error function to use (default is ``l2``).

        - ``error='l1'``: the sum of the absolute differences between ``y`` and
          ``h * x``.
        - ``error='l2'``: the sum of the squared differences between ``y`` and
          ``h * x``.

        For vector ``y``, the error is defined based on the distance in space
        for each data point.
    basis : scalar
        Use a basis of windows with this length for the kernel (by default,
        impulses are used).
    basis_window : str | scalar | tuple
        Basis window (see :func:`scipy.signal.get_window` for options; default
        is ``'hamming'``).
    partitions : int
        Divide the data into this many ``partitions`` for cross-validation-based
        early stopping. In each partition, ``n - 1`` segments are used for
        training, and the remaining segment is used for validation.
        If data is continuous, data are divided into contiguous segments of
        equal length (default 10).
        If data has cases, cases are divided with ``[::partitions]`` slices
        (default ``min(n_cases, 10)``; if ``model`` is specified, ``n_cases``
        is the lowest number of cases in any cell of the model).
    model : Categorial
        If data has cases, divide cases into different categories (division
        for crossvalidation is done separately for each cell).
    ds : Dataset
        If provided, other parameters can be specified as string for items in
        ``ds``.
    validate : int
        Number of segments in validation dataset (currently has to be 1).
    test : int
        By default (``test=0``), the boosting algorithm uses all available data
        to estimate the kernel. Set ``test=1`` to perform *k*-fold cross-
        validation instead (with *k* = ``partitions``):
        Each partition is used as test dataset in turn, while the remaining
        ``k-1`` partitions are used to estimate the kernel. The resulting model
        fit metrics reflect the re-combination of all partitions, each one
        predicted from the corresponding, independent training set.
    selective_stopping : int
        By default, the boosting algorithm stops when the testing error stops
        decreasing. With ``selective_stopping=True``, boosting continues but
        excludes the predictor (one time-series in ``x``) that caused the
        increase in testing error, and continues until all predictors are
        stopped. The integer value of ``selective_stopping`` determines after
        how many steps with error increases each predictor is excluded.
    debug : bool
        Add additional attributes to the returned result.

    Returns
    -------
    result : BoostingResult
        Results (see :class:`BoostingResult`).

    Notes
    -----
    In order to predict data, use the :func:`convolve` function::

    >>> ds = datasets.get_uts()
    >>> ds['a1'] = epoch_impulse_predictor('uts', 'A=="a1"', ds=ds)
    >>> ds['a0'] = epoch_impulse_predictor('uts', 'A=="a0"', ds=ds)
    >>> res = boosting('uts', ['a0', 'a1'], 0, 0.5, partitions=10, model='A', ds=ds)
    >>> y_pred = convolve(res.h_scaled, ['a0', 'a1'], ds=ds)

    The boosting algorithm is described in [1]_.

    References
    ----------
    .. [1] David, S. V., Mesgarani, N., & Shamma, S. A. (2007). Estimating
        sparse spectro-temporal receptive fields with natural stimuli. Network:
        Computation in Neural Systems, 18(3), 191-212.
        `10.1080/09548980701609235 <https://doi.org/10.1080/09548980701609235>`_.
    """
    # scale_data
    if isinstance(scale_data, bool):
        scale_in_place = False
    elif isinstance(scale_data, str):
        if scale_data == 'inplace':
            scale_in_place = True
        else:
            raise ValueError(f"scale_data={scale_data!r}")
    else:
        raise TypeError(f"scale_data={scale_data!r}, need bool or str")
    # selective_stopping
    selective_stopping = int(selective_stopping)
    if selective_stopping < 0:
        raise ValueError(f"selective_stopping={selective_stopping}")

    data = RevCorrData(y, x, ds, scale_in_place)
    data.apply_basis(basis, basis_window)
    if scale_data:
        data.normalize(error)
    data.initialize_cross_validation(partitions, model, ds, validate, test)

    fit = Boosting(data)
    fit.fit(tstart, tstop, selective_stopping, error, delta, mindelta)
    return fit.evaluate_fit(debug=debug)


class BoostingStep:
    __slots__ = ('i_stim', 'i_time', 'delta', 'e_train', 'e_test')

    def __init__(self, i_stim, i_time, delta_signed, e_test, e_train):
        self.i_stim = i_stim
        self.i_time = i_time
        self.delta = delta_signed
        self.e_train = e_train
        self.e_test = e_test


def boost(y, x, x_pads, split, i_start_by_x, i_stop_by_x, delta, mindelta, error, selective_stopping=0, return_history=False):
    """Estimate one filter with boosting

    Parameters
    ----------
    y : array (n_times,)
        Dependent signal, time series to predict.
    x : array (n_stims, n_times)
        Stimulus.
    x_pads : array (n_stims,)
        Padding for x.
    split : Split
        Training/validation data split.
    i_start_by_x : ndarray
        Array of i_start for trfs.
    i_stop_by_x : ndarray
        Array of i_stop for TRF.
    delta : scalar
        Step of the adjustment.
    mindelta : scalar
        Smallest delta to use. If no improvement can be found in an iteration,
        the first step is to divide delta in half, but stop if delta becomes
        smaller than ``mindelta``.
    error : str
        Error function to use.
    selective_stopping : int
        Selective stopping.
    return_history : bool
        Return error history as second return value.

    Returns
    -------
    history[best_iter] : None | array
        Winning kernel, or None if 0 is the best kernel.
    test_sse_history : list (only if ``return_history==True``)
        SSE for test data at each iteration.
    """
    delta_error_func = DELTA_ERROR_FUNC[error]
    error = ERROR_FUNC[error]
    n_stims, n_times = x.shape
    assert y.shape == (n_times,)
    i_start = np.min(i_start_by_x)
    n_times_trf = np.max(i_stop_by_x) - i_start
    h = np.zeros((n_stims, n_times_trf))
    # buffers
    y_error = y.copy()
    new_error = np.empty(h.shape)
    new_error.fill(np.inf)  # ignore values outside TRF
    new_sign = np.empty(h.shape, np.int8)
    x_active = np.ones(n_stims, dtype=np.int8)

    # history
    best_test_error = np.inf
    history = []
    i_stim = i_time = delta_signed = None
    best_iteration = 0
    # pre-assign iterators
    for i_boost in range(999999):
        # evaluate current h
        e_train = error(y_error, split.train)
        e_test = error(y_error, split.validate)
        step = BoostingStep(i_stim, i_time, delta_signed, e_test, e_train)
        history.append(step)

        # evaluate stopping conditions
        if e_test < best_test_error:
            best_test_error = e_test
            best_iteration = i_boost
        elif i_boost >= 2 and e_test > history[-2].e_test:
            if selective_stopping:
                if selective_stopping > 1:
                    n_bad = selective_stopping - 1
                    # only stop if the predictor overfits twice without intermittent improvement
                    undo = 0
                    for i in range(-2, -len(history), -1):
                        step = history[i]
                        if step.e_test > e_test:
                            break  # the error improved
                        elif step.i_stim == i_stim:
                            if step.e_test > history[i - 1].e_test:
                                # the same stimulus caused an error increase
                                if n_bad == 1:
                                    undo = i
                                    break
                                n_bad -= 1
                            else:
                                break
                else:
                    undo = -1

                if undo:
                    # revert changes
                    for i in range(-undo):
                        step = history.pop(-1)
                        h[step.i_stim, step.i_time] -= step.delta
                        update_error(y_error, x[step.i_stim], x_pads[step.i_stim], split.train_and_validate, -step.delta, step.i_time + i_start)
                    step = history[-1]
                    # disable predictor
                    x_active[i_stim] = False
                    if not np.any(x_active):
                        break
                    new_error[i_stim, :] = np.inf
            # Basic
            # -----
            # stop the iteration if all the following requirements are met
            # 1. more than 10 iterations are done
            # 2. The testing error in the latest iteration is higher than that in
            #    the previous two iterations
            elif i_boost > 10 and e_test > history[-3].e_test:
                # print("error(test) not improving in 2 steps")
                break

        # generate possible movements -> training error
        generate_options(y_error, x, x_pads, x_active, split.train, i_start, i_start_by_x, i_stop_by_x, delta_error_func, delta, new_error, new_sign)
        i_stim, i_time = np.unravel_index(np.argmin(new_error), h.shape)
        new_train_error = new_error[i_stim, i_time]
        delta_signed = new_sign[i_stim, i_time] * delta

        # If no improvements can be found reduce delta
        if new_train_error > step.e_train:
            delta *= 0.5
            if delta >= mindelta:
                i_stim = i_time = delta_signed = None
                # print("new delta: %s" % delta)
                continue
            else:
                # print("No improvement possible for training data")
                break

        # abort if we're moving in circles
        if step.delta and i_stim == step.i_stim and i_time == step.i_time and delta_signed == -step.delta:
            break

        # update h with best movement
        h[i_stim, i_time] += delta_signed
        update_error(y_error, x[i_stim], x_pads[i_stim], split.train_and_validate, delta_signed, i_time + i_start)
    else:
        raise RuntimeError("Maximum number of iterations exceeded")
    # print('  (%i iterations)' % (i_boost + 1))

    # reverse changes after best iteration
    if best_iteration:
        for step in history[-1: best_iteration: -1]:
            if step.delta:
                h[step.i_stim, step.i_time] -= step.delta
    else:
        h = None

    if return_history:
        return h, [step.e_test for step in history]
    else:
        return h


def setup_workers(data, i_start, i_stop, delta, mindelta, error, selective_stopping):
    n_y, n_times = data.y.shape
    n_x, _ = data.x.shape

    y_buffer = RawArray('d', n_y * n_times)
    y_buffer[:] = data.y.ravel()
    x_buffer = RawArray('d', n_x * n_times)
    x_buffer[:] = data.x.ravel()
    x_pads_buffer = RawArray('d', n_x)
    x_pads_buffer[:] = data.x_pads

    job_queue = mpc.Queue(200)
    result_queue = mpc.Queue(200)

    args = (y_buffer, x_buffer, x_pads_buffer, n_y, n_times, n_x, data.splits.splits, i_start, i_stop, delta, mindelta, error, selective_stopping, job_queue, result_queue)
    for _ in range(CONFIG['n_workers']):
        process = mpc.Process(target=boosting_worker, args=args)
        process.start()

    return job_queue, result_queue


def boosting_worker(y_buffer, x_buffer, x_pads_buffer, n_y, n_times, n_x, splits, i_start, i_stop, delta, mindelta, error, selective_stopping, job_queue, result_queue):
    if CONFIG['nice']:
        os.nice(CONFIG['nice'])

    y = np.frombuffer(y_buffer, np.float64, n_y * n_times).reshape((n_y, n_times))
    x = np.frombuffer(x_buffer, np.float64, n_x * n_times).reshape((n_x, n_times))
    x_pads = np.frombuffer(x_pads_buffer, np.float64, n_x)

    while True:
        i_y, i_split = job_queue.get()
        if i_y == JOB_TERMINATE:
            return
        h = boost(y[i_y], x, x_pads, splits[i_split], i_start, i_stop, delta, mindelta, error, selective_stopping)
        result_queue.put((i_y, i_split, h))


def put_jobs(queue, n_y, n_splits, stop):
    "Feed boosting jobs into a Queue"
    for job in product(range(n_y), range(n_splits)):
        queue.put(job)
        if stop.isSet():
            while not queue.empty():
                queue.get()
            break
    for _ in range(CONFIG['n_workers']):
        queue.put((JOB_TERMINATE, None))


def convolve(
        h: np.ndarray,
        x: np.ndarray,
        x_pads: np.ndarray,
        h_i_start: int,
        segments: np.ndarray = None,
        out: np.ndarray = None,
):
    """h * x with time axis matching x

    Parameters
    ----------
    h : array, (n_stims, h_n_samples)
        H.
    x : array, (n_stims, n_samples)
        X.
    x_pads : array (n_stims,)
        Padding for x.
    h_i_start : int
        Time shift of the first sample of ``h``.
    segments : array (n_segments, 2)
        Data segments.
    out : array
        Buffer for predicted ``y``.
    """
    n_x, n_times = x.shape
    h_n_times = h.shape[1]
    if segments is None:
        segments = np.array(((0, n_times),), np.int64)
    if out is None:
        out = np.zeros(n_times)
    else:
        for a, b in segments:
            out[a:b] = 0
    h_i_stop = h_i_start + h_n_times

    # padding
    h_pad = np.sum(h * x_pads.reshape((-1, 1)), 0)
    # padding for pre-
    pad_head_n_times = max(0, h_n_times + h_i_start)
    if pad_head_n_times:
        pad_head = np.zeros(pad_head_n_times)
        for i in range(min(pad_head_n_times, h_n_times)):
            pad_head[:pad_head_n_times - i] += h_pad[- i - 1]
    # padding for post-
    pad_tail_n_times = -min(0, h_i_start)
    if pad_tail_n_times:
        pad_tail = np.zeros(pad_tail_n_times)
        for i in range(pad_tail_n_times):
            pad_tail[i:] += h_pad[i]

    for start, stop in segments:
        if pad_head_n_times:
            out[start: start + pad_head_n_times] += pad_head
        if pad_tail_n_times:
            out[stop - pad_tail_n_times: stop] += pad_tail
        convolve_jit(h, x[:, start:stop], out[start:stop], h_i_start, h_i_stop)
    return out
