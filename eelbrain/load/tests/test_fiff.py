# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

import os
from warnings import catch_warnings, filterwarnings

from nose.tools import eq_
from numpy.testing import assert_array_equal, assert_array_almost_equal

import mne
from mne import pick_types
from mne.io.kit.tests import data_dir as kit_test_data_dir

from eelbrain import load

from ...tests.test_data import assert_dataobj_equal
from eelbrain._utils.testing import requires_mne_sample_data


FILTER_WARNING = ('The measurement information indicates a low-pass frequency '
                  'of 40 Hz.')


def test_load_fiff_sensor():
    umd_sqd_path = os.path.join(kit_test_data_dir, 'test_umd-raw.sqd')
    raw = mne.io.read_raw_kit(umd_sqd_path)

    sensor = load.fiff.sensor_dim(raw)
    eq_(sensor.sysname, 'KIT-UMD-3')


@requires_mne_sample_data
def test_load_fiff_from_raw():
    "Test loading data from a fiff raw file"
    data_path = mne.datasets.sample.data_path()
    meg_path = os.path.join(data_path, 'MEG', 'sample')
    raw_path = os.path.join(meg_path, 'sample_audvis_filt-0-40_raw.fif')
    evt_path = os.path.join(meg_path, 'sample_audvis_filt-0-40_raw-eve.fif')

    # load events
    ds = load.fiff.events(raw_path)
    eq_(ds['i_start'].x.dtype.kind, 'i')

    # test separate events
    ds_evt = load.fiff.events(events=evt_path)
    ds_evt.name = ds.name
    assert_dataobj_equal(ds_evt, ds)

    # add epochs as ndvar
    ds = ds.sub('trigger == 32')
    with catch_warnings():
        filterwarnings('ignore', message=FILTER_WARNING)
        ds_ndvar = load.fiff.add_epochs(ds, -0.1, 0.3, decim=10, data='mag',
                                        proj=False, reject=2e-12)
    meg = ds_ndvar['meg']
    eq_(meg.ndim, 3)
    data = meg.get_data(('case', 'sensor', 'time'))

    # compare with mne epochs
    with catch_warnings():
        filterwarnings('ignore', message=FILTER_WARNING)
        ds_mne = load.fiff.add_mne_epochs(ds, -0.1, 0.3, decim=10, proj=False,
                                          reject={'mag': 2e-12})
    epochs = ds_mne['epochs']
    picks = pick_types(epochs.info, meg='mag')
    mne_data = epochs.get_data()[:, picks]
    assert_array_equal(meg.sensor.names, [epochs.info['ch_names'][i] for i in picks])
    assert_array_equal(data, mne_data)
    assert_array_almost_equal(meg.time.x, epochs.times)

    # with proj
    with catch_warnings():
        filterwarnings('ignore', message=FILTER_WARNING)
        meg = load.fiff.epochs(ds, -0.1, 0.3, decim=10, data='mag', proj=True,
                               reject=2e-12)
        epochs = load.fiff.mne_epochs(ds, -0.1, 0.3, decim=10, proj=True,
                                      reject={'mag': 2e-12})
    picks = pick_types(epochs.info, meg='mag')
    mne_data = epochs.get_data()[:, picks]
    assert_array_almost_equal(meg.x, mne_data, 10)
