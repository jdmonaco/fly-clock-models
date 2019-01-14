"""
Find paths to data files and parse them.
"""

import os
import re


dir_pattern = re.compile('(\w+) (ZT[\w-]+)')
file_pattern = re.compile('(\d+).atf')
ZT_pattern = re.compile('ZT(\d+)-\d+')


def timecode_to_phase(tcode):
    """Convert a timecode to a string label for the wake/sleep condition."""
    match = re.match(ZT_pattern, tcode)
    if match is None:
        raise ValueError('bad timecode: {}'.format(tcode))

    ZT = int(match.groups()[0])
    if ZT < 12:
        return 'day'
    return 'night'


def find_data_paths(root):
    """Given the root data directory, find and label all the data."""

    data = {}

    for dirpath, dirnames, filenames in os.walk(root):
        stem, base = os.path.split(dirpath)
        match = re.match(dir_pattern, base)
        if match is None:
            continue

        area, condn = match.groups()

        for fn in filenames:
            match = re.match(file_pattern, fn)
            if match is None:
                continue

            label = match.groups()[0]
            path = os.path.join(dirpath, fn)

            data[path] = dict(area=area, timecode=condn, label=label)

    return data





