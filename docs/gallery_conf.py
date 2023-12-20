import os
import sys
import re

from mkdocs_gallery.sorting import FileNameSortKey

min_reported_time = 0
if 'SOURCE_DATE_EPOCH' in os.environ:
    min_reported_time = sys.maxint if sys.version_info[0] == 2 else sys.maxsize

# To be used as the "base" config,
# mkdocs-gallery is a port of sphinx-gallery. For a detailed list
# of configuration options see https://sphinx-gallery.github.io/stable/configuration.html
conf = {
    # report runtime if larger than this value
    'min_reported_time': min_reported_time,
    # order your section in file name alphabetical order
    'within_subsection_order': FileNameSortKey,
    # run every script that matches pattern
    # (here we match every file that ends in .py)
    'filename_pattern': re.escape(os.sep) + r".+\.py$"
}
