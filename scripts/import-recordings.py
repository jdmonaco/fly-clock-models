#!/usr/bin/env python3

import sys

import numpy as np
import tables as tb

from pouty import ConsolePrinter
from roto.data import new_table, new_array

from flyback import RAW_DATA_ROOT, store
from flyback.lib.data import recordings_description, recording_path
from flyback.lib.datasets import find_data_paths, timecode_to_phase


def main(log, root):
    f = store.get(False)
    rectable = new_table(f, '/', 'recordings', recordings_description)
    datafiles = find_data_paths(root)
    log('Found {} data files.', len(datafiles))

    i = 0
    rec = rectable.row
    for i, path in enumerate(sorted(datafiles.keys())):
        info = datafiles[path]
        info.update(phase=timecode_to_phase(info['timecode']))

        rec['id'] = i
        for key in ('area', 'timecode', 'label', 'phase'):
            rec[key] = info[key]

        # Set paths to file on disk and to data group in store
        group_path = recording_path(info)
        rec['file'] = path
        rec['path'] = group_path

        # Now actually import the voltage trace data
        T, V = np.loadtxt(path, skiprows=3, unpack=True)
        log('Imported: {}', path)

        # And save the time series to the data group
        Tarr = new_array(f, group_path, 'T', T, createparents=True)
        Varr = new_array(f, group_path, 'V', V)
        log('Saved: {}', Tarr._v_pathname)
        log('Saved: {}', Varr._v_pathname)

        rec.append()
        i += 1

    rectable.flush()
    store.close()

    log('Finished importing recording data.')

    return 0


if __name__ == "__main__":
    log = ConsolePrinter(prefix='FlybackImport', prefix_color='lightgreen')
    sys.exit(main(log, RAW_DATA_ROOT))
