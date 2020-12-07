import json
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, NamedTuple, Optional, Union

from .utils import (_download_cord19, _download_hist_releases,
                    get_cache_home_dir, load_release_content,
                    rename_file_and_path, source_directories)


class Info(NamedTuple):
    has_cached_files: bool = False
    download_date: str = ''

    def asdict(self):
        return dict(self._asdict())


class Links(NamedTuple):
    log: str
    zip: str

    def asdict(self):
        return dict(self._asdict())


class ArchiveBase(NamedTuple):
    date: str
    info: Info
    links: Links
    content: Dict
    source: Optional[Any] = None

    @property
    def log_url(self) -> str:
        return self.links.log

    @property
    def zip_url(self) -> str:
        return self.links.zip

    @property
    def has_cached_files(self) -> bool:
        return self.info.has_cached_files

    @property
    def download_date(self) -> str:
        return self.info.download_date

    def asdict(self, date_as_root=True):
        dict_obj = {'info': self.info.asdict(),
                    'links': self.links.asdict(),
                    'content': self.content}
        if date_as_root:
            dict_obj = {self.date: dict_obj}
        return dict_obj


class ArchiveConfig(ArchiveBase):

    @staticmethod
    def from_dict(cfg: Dict[str, Dict[str, Dict]]) -> 'ArchiveConfig':
        date = list(cfg.keys())[0]
        arch = cfg[date]
        info = Info(**arch['info'])
        links = Links(**arch['links'])
        dirs = source_directories(cfg)[0]
        Base = namedtuple('Base', tuple(dirs.keys()))

        class Source(Base):
            @property
            def paths(self) -> Union[Path, List[Path]]:
                p = list(map(self.__getattribute__, self._fields))
                if len(p) == 1:
                    return p[0]
                return p
            def __repr__(self):
                output = list(self._asdict().items())
                return pformat(output, compact=True)

        source = Source(*dirs.values())
        return ArchiveConfig(date, info, links, arch['content'], source)

    def __repr__(self):
        return pformat(list(self.asdict().items()), compact=True)


class DownloadManager:
    release_file = 'realeases.json'
    archive_file = 'archive.json'

    def __init__(self, custom_hist_dir=None, custom_cachedir=None):
        self.custom_hist_dir = custom_hist_dir
        self.custom_cachedir = custom_cachedir
        self._hist_dir: Optional[Path] = None
        self._cachedir: Optional[Path] = None
        self._archive: Optional[Dict[str, Dict[str, str]]] = None
        self.has_releases: Optional[bool] = None
        self.has_archive: Optional[bool] = None
        self.is_custom_cachedir: Optional[bool] = None
        self.is_custom_hist_dir: Optional[bool] = None
        self._init_verify()

    def _init_verify(self):
        # Configure one or both custom directories (if any)
        # otherwise the defaults will be used for any is None.
        custom_cachedir = self.custom_cachedir
        custom_hist_dir = self.custom_hist_dir
        if custom_cachedir is not None:
            if not isinstance(custom_cachedir, Path):
                custom_cachedir = Path(custom_cachedir)
            self.custom_cachedir = custom_cachedir
            self.is_custom_cachedir = True

        if custom_hist_dir is not None:
            if not isinstance(custom_hist_dir, Path):
                custom_hist_dir = Path(custom_hist_dir)
            self.custom_hist_dir = custom_hist_dir
            self.is_custom_hist_dir = True

        if not self.is_custom_cachedir or not self.is_custom_hist_dir:
            self._hist_dir = get_cache_home_dir(subdir='hr')
            self.is_custom_hist_dir = False
            if not self.is_custom_cachedir:
                self._cachedir = self._hist_dir.parent
                self.is_custom_cachedir = False

        # Check if the user has downloaded the historical releases
        # within the default cache directory.
        if not self.is_custom_cachedir:
            if self.release_fp.exists() and self.release_fp.is_file():
                self.has_releases = True
            else:
                self.has_releases = False
            # Make sure the file can be loaded; otherwise the archive
            # should not be considered valid even if the file exist.
            if self.archive_fp.exists() and self.archive_fp.is_file():
                try:
                    self.load_archive()
                except Exception:
                    self.has_archive = False
                else:
                    self.has_archive = True
            else:
                self.has_archive = False

    @ property
    def cachedir(self):
        return self._cachedir if self.custom_cachedir is None \
            else self.custom_cachedir

    @ property
    def hist_dir(self):
        return self._hist_dir if self.custom_hist_dir is None \
            else self.custom_hist_dir

    @ property
    def release_fp(self) -> Path:
        return self.cachedir.joinpath(self.release_file)

    @ property
    def archive_fp(self) -> Path:
        return self.cachedir.joinpath(self.archive_file)

    def all_archive_dates(self) -> List[str]:
        current_archive = self.load_archive()
        dates = list(current_archive.keys())
        return dates

    def all_release_dates(self) -> List[str]:
        hist_releases = self.load_releases()
        dates = list(hist_releases.keys())
        return dates

    def load_releases(self, download=False):
        data:  Dict[str, Dict[str, str]]
        if not self.has_releases:
            if download or not self.is_custom_cachedir:
                self.download_releases(self.release_fp)
            else:
                raise ValueError(
                    'Release data does not exist. Set download = True or '
                    'download to a custom/default location by using the '
                    'self.download_releases("path/to/realeases.json") method. '
                    'The filename must match `realeases.json` name otherwise '
                    'overwrite the self.release_file="custom.json" attribute.'
                )
        with self.release_fp.open('r') as hr_file:
            data = json.load(hr_file)
            self.has_releases = True
        return data

    def download_releases(self, fp: Union[str, Path]) -> None:
        if not isinstance(fp, Path):
            fp = Path(fp)
        with fp.open('w') as hr_file:
            hr = _download_hist_releases()
            json.dump(hr, hr_file, indent=2)
            self.has_releases = True

    def load_archive(self) -> Dict[str, Dict[str, str]]:
        archive_fp = self.archive_fp
        if not archive_fp.exists():
            raise ValueError(f"Archive file does not exist {archive_fp}")
        if self._archive is not None:
            return self._archive
        with archive_fp.open('r') as fp:
            archive_data = json.load(fp)
            self._archive = archive_data
        return archive_data

    def download(
        self, date: str, rm_cached: bool = True, fix_csv_name: bool = True,
    ) -> Dict[str, Dict[str, str]]:
        info = {
            'has_cached_files': False if rm_cached else True,
            'download_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        hist_releases = self.load_releases(download=True)
        release, results = _download_cord19(date, hist_releases, self.hist_dir)
        if rm_cached:
            for path in results['download'].values():
                path.unlink()

        content = load_release_content(results['ready']['log'].parent)
        if fix_csv_name:
            rename_file_and_path(
                content, newname='embeddings', suffix='.csv')

        content = {k: v.as_posix() for k, v in content.items()}
        archive_data = {date: {'links': release,
                               'content': content, 'info': info}}
        tmp_archive = {}
        if self.archive_fp.exists() and self.archive_fp.is_file():
            tmp_archive.update(self.load_archive())

        tmp_archive.update(archive_data)
        with self.archive_fp.open('w') as writer:
            json.dump(tmp_archive, writer, indent=4)
        return archive_data
