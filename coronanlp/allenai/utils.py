
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Union

import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from wasabi import msg

from ..utils import get_store_dir
from .config import DEFAULT_CACHE_DIR, HIST_REALEASES


class tqdm_stream(tqdm):
    def update_stream(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def rename_file_and_path(content: Dict[str, Union[str, Path]],
                         newname: str, suffix='') -> None:
    """Helper method to rename the key and file of a path in a dict object.

    NOTE: Since this method renames the existing file `in-place`; updating
    the key (name) value (path) is also done in-place.

    :params newname: The new name most match a chuck of the sequence
        or key. For example, if `key: cord_19_embeddings_4_17` and its
        `path: path/to/cord_19_embeddings_4_17.csv` and we want the
        name to be only `"embeddings"` - this method will work since
        both (key,path) meet this condition (have a chuck of sequence
        of the `newname`). The expected result for `key: embeddings`
        and `path: path/to/embeddings.csv`. The file does not need
        have a `".csv"` - any suffix will work as long as the argument
        `suffix=".my-suffix"` or leave as `""` if it doesn't have one.
    """
    trg_name = [k for k in content.keys() if newname in k]
    if trg_name and len(trg_name) == 1 \
            and trg_name[0] != newname:
        old = content.pop(trg_name[0])
        is_posix_type = False
        if isinstance(old, str) \
                and not isinstance(old, Path):
            old = Path(old)
            is_posix_type = True
        if old.exists() and old.is_file() \
                and old.suffix == suffix:
            new = old.parent.joinpath(f'{newname}{suffix}')
            old.rename(new)
            # Key and new filename for path renamed successfully.
            content[newname] = new.as_posix() \
                if is_posix_type else new
            return
        # Pop key and previous path back in (file is not renamed).
        content[trg_name[0]] = old.as_posix() \
            if is_posix_type else old
        return


def get_cache_home_dir(subdir: str = None,
                       cache_dir: Optional[Union[str, Path]] = None) -> Path:
    """Obtain the path to the cache directory or w/a sub-directory.

    - NOTE: Any non-existing directories is will be created. Including
        its non-existing parents by default.

    Usage:
    ```python
    d = get_cache_home_dir(None, None)  # fallback to all defaults.
    # PosixPath('/home/usr/.cache/coronanlp/semanticscholar')

    d = get_cache_home_dir('hr') # default cache with +sub-directory
    # PosixPath('/home/usr/.cache/coronanlp/semanticscholar/hr')

    mycache = 'other/cache_dir/path'  # custom cache and sub directories.
    d = get_cache_home_dir('hr', mycache)
    # PosixPath('other/cache_dir/path/semanticscholar/hr')
    ```
    """
    cache_home = DEFAULT_CACHE_DIR
    if subdir is not None and isinstance(subdir, str):
        subdir = subdir.strip()
        if subdir.find('/', 0) == 0:
            subdir = subdir[1:]  # remove first '/' occurrance if
        cache_home = f'{cache_home}/{subdir}'
    if cache_dir is None:
        cache = get_store_dir()
        cache_dir = cache.parent.joinpath(cache_home)
    elif isinstance(cache_dir, str) and not isinstance(cache_dir, Path):
        cache_dir = Path(cache_dir)
        cache_dir = cache_dir.joinpath(cache_home)
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)
    return cache_dir


def load_release_content(release, ignore_suffixes=['.gz', '.tar']):
    if not isinstance(release, Path):
        release = Path(release)
    dirname = release.parts[-1]  # Check if last dir is in date format.
    isdate = all(map(str.isnumeric, dirname.split('-')))
    assert isdate, ('Expected last directory in date (ISO) '
                    f'`00-00-00` format, but instead got: {dirname}')
    return {p.name.replace(p.suffix, ''): p for p in release.iterdir()
            if p.suffix not in ignore_suffixes}


def _download_hist_releases() -> Dict[str, Dict[str, str]]:
    """Download all available historical releases from Semantic-Scholar.

    Usage:
    ```python
    releases = download_hist_releases()
    releases["2020-10-10"]  # changelog & zip urls per release date.
    ...
    {'log':'https://ai2../hist_releases/2020-10-10/changelog',
     'zip':'https://ai2../hist_releases/cord-19_2020-10-10.tar.gz'}
    ```
    """
    req = requests.get(HIST_REALEASES)
    soup = BeautifulSoup(req.text, features='html.parser')
    table = soup.find('table')
    links = table.find_all('a')
    hist_releases = {}
    for i in range(0, len(links), 2):  # 2: changelog+tarfile
        link = links[i: i + 2]
        if len(link) != 2:
            continue
        logfile, zipfile = (link[0].attrs['href'],
                            link[1].attrs['href'])
        date = link[1].getText()[len('cord-19_'):]
        date = date.replace('.tar.gz', '')
        hist_releases.update({
            date: {'log': logfile, 'zip': zipfile}})

    return hist_releases


def _download_cord19(date, hist_releases, outdir=None, suffix='.tar.gz'):

    def unzip(file, outdir):
        try:
            tar = tarfile.open(file, 'r:gz')
            tar.extractall(outdir)
        except Exception as err:
            tar = tarfile.open(file, 'r:bz2')
            tar.extractall(outdir)

    # Get the default cache directory, and make if do not exist.
    if outdir is None:
        outdir = get_cache_home_dir(subdir='hr')
    # Otherwise if outdir is not None, use the custom destination.
    elif not isinstance(outdir, Path):
        outdir = Path(outdir)
    # Create the directory with the date as the name, files
    # will be downloaded and extracted here per unique date.
    ds_dir = outdir.joinpath(date)
    if not ds_dir.exists():
        ds_dir.mkdir(parents=True)
    # download: Keep track of the downloaded files (for cleaning later)
    # ready: Keep track of the files and directories that can be accessed
    # by the user.
    metadata = {'ready': {}, 'download': {}}
    release = hist_releases[date]
    split = urllib.request.urlsplit(release['zip'])
    temp_fp = Path(split.path)
    temp_fp = outdir.joinpath(temp_fp.name)

    kwargs = {'unit': 'B',
              'unit_scale': True,
              'unit_divisor': 1024,
              'miniters': 1,
              'desc': f'Downloading ⚕ '}

    with tqdm_stream(**kwargs) as stream:
        try:
            link = split.geturl()
            result = urllib.request.urlretrieve(
                link, filename=temp_fp, reporthook=stream.update_stream)
        except Exception as e:
            raise ValueError(
                f"Link {split.geturl()} raised an exception: {e}")

    metadata['download'].update({'zip': result[0]})
    metadata['ready'].update({'log': ds_dir.joinpath('changelog')})

    msg.info(f'extracting compressed file {result[0].name} ⌛ ...')
    unzip(result[0], outdir=ds_dir.parent)
    subtars = {
        st.name.replace(suffix, ''): st
        for st in ds_dir.glob(f'*{suffix}')
        if st != result[0]
    }
    if subtars:
        msg.info('extracting content: `{}` ⌛ ...'.format(
            ', '.join(list(subtars.keys())))
        )
        for name, subtar in subtars.items():
            sub_fp = ds_dir.joinpath(name)
            unzip(subtar, sub_fp.parent)
            metadata['ready'].update({name: sub_fp})

        metadata['download'].update(subtars)
    return release, metadata
