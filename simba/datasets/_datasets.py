"""Built-in datasets for the minimal scATAC workflow."""

import os
import urllib.request

from tqdm import tqdm

from .._settings import settings
from ..readwrite import read_h5ad


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path, desc=None):
    if desc is None:
        desc = url.split('/')[-1]
    with DownloadProgressBar(
        unit='B',
        unit_scale=True,
        miniters=1,
        desc=desc,
    ) as pbar:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=pbar.update_to)


def atac_buenrostro2018():
    """Single-cell ATAC-seq human blood data used in the SIMBA tutorial."""
    url = 'https://www.dropbox.com/s/7hxjqgdxtbna1tm/atac_seq.h5ad?dl=1'
    filename = 'atac_buenrostro2018.h5ad'
    data_dir = os.path.join(settings.workdir, 'data')
    fullpath = os.path.join(data_dir, filename)

    if not os.path.exists(fullpath):
        print('Downloading data ...')
        os.makedirs(data_dir, exist_ok=True)
        download_url(url, fullpath, desc=filename)
        print(f'Downloaded to {data_dir}.')

    return read_h5ad(fullpath)
