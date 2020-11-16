import json
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Dict, List, NamedTuple, Union

from .utils import (_download_cord19, _download_hist_releases,
                    get_cache_home_dir, load_release_content,
                    rename_file_and_path)


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


class Content(NamedTuple):
    metadata: Path
    changelog: Path
    embeddings: Path
    document_parses: Path

    @property
    def pdf_json(self):
        return self.document_parses.joinpath('pdf_json')

    @property
    def pmc_json(self):
        return self.document_parses.joinpath('pmc_json')

    def asdict(self, detach_posix=True):
        dict_obj = {}
        for key, val in self._asdict().items():
            dict_obj[key] = val.as_posix() if detach_posix else val
        dict_obj.update({'pdf_json': self.pdf_json.as_posix(),
                         'pmc_json': self.pmc_json.as_posix()})
        return dict_obj


class ArchiveBase(NamedTuple):
    date: str
    info: Info
    links: Links
    content: Content

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

    @property
    def metadata(self) -> Path:
        return self.content.metadata

    @property
    def changelog(self) -> Path:
        return self.content.changelog

    @property
    def embeddings(self) -> Path:
        return self.content.embeddings

    @property
    def document_parses(self) -> Path:
        return self.content.document_parses

    def asdict(self, date_as_root=True):
        dict_obj = {'info': self.info.asdict(),
                    'links': self.links.asdict(),
                    'content': self.content.asdict()}
        if date_as_root:
            dict_obj = {self.date: dict_obj}
        return dict_obj


class ArchiveConfig(ArchiveBase):
    @staticmethod
    def from_dict(date: str, cfg: Dict[str, Dict[str, str]]):
        return ArchiveConfig(
            date=date,
            info=Info(**cfg['info']),
            links=Links(**cfg['links']),
            content=Content(**{
                k: Path(v) for k, v in cfg['content'].items()}
            )
        )

    def __repr__(self):
        return pformat(list(self.asdict().items()), compact=True)


class DownloadManager:
    release_file = 'realeases.json'
    archive_file = 'archive.json'

    def __init__(self, custom_hist_dir=None, custom_cachedir=None):
        self.custom_hist_dir = custom_hist_dir
        self.custom_cachedir = custom_cachedir
        self._hist_dir = get_cache_home_dir(subdir='hr')
        self._cachedir = self._hist_dir.parent
        self._archive = None
        self.has_releases = False
        if custom_cachedir is not None \
                and not isinstance(custom_cachedir, Path):
            self.custom_cachedir = Path(custom_cachedir)
        if custom_hist_dir is not None \
                and not isinstance(custom_hist_dir, Path):
            self.custom_hist_dir = Path(custom_hist_dir)

    @property
    def cachedir(self) -> Path:
        return self.custom_cachedir if not None else self._cachedir

    @property
    def hist_dir(self) -> Path:
        return self.custom_hist_dir if not None else self._hist_dir

    @property
    def release_fp(self) -> Path:
        return self.cachedir.joinpath(self.release_file)

    @property
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

    def load_releases(self, download=False) -> Dict[str, Dict[str, str]]:
        if not self.release_fp.exists():
            if not download:
                raise ValueError(
                    'Release data does not exist. Set download=True or'
                    ' download to a custom/default location by using'
                    ' the self.download("save/dump/here.json") method')
            self.download_releases(self.release_fp)
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

    def download(self, date: str, rm_cached=True, fix_csv_name=True) -> Dict[str, Dict[str, str]]:
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
        archive_data = {date: {
            'links': release, 'content': content, 'info': info}
        }
        with self.archive_fp.open('w') as writer:
            if not self.archive_fp.exists():
                json.dump(archive_data, writer, indent=4)
            else:
                hist_archive = self.load_archive()
                hist_archive.update(archive_data)
                json.dump(hist_archive, writer, indent=4)

        return archive_data
