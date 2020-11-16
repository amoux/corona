
DEFAULT_CACHE_DIR = 'semanticscholar'
HIST_REALEASES = 'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases.html'
IDENTIFIER_KEYS = ['S2PaperID', 'DOI', 'ArXivID',
                   'MAGID', 'ACLID', 'PubMedID', 'CorpusID']
API_ENDPOINTS = {
    's2_paper': {
        'id': 'S2PaperID',
        'url': 'https://api.semanticscholar.org/v1/paper/'
    },
    'doi': {
        'id': 'DOI',
        'url': 'https://api.semanticscholar.org/v1/paper/'
    },
    'arxiv': {
        'id': 'ArXivID',
        'url': 'https://api.semanticscholar.org/v1/paper/arXiv:'
    },
    'mag': {
        'id': 'MAGID',
        'url': 'https://api.semanticscholar.org/v1/paper/MAG:'
    },
    'acl': {
        'id': 'ACLID',
        'url': 'https://api.semanticscholar.org/v1/paper/ACL:'
    },
    'pubmed': {
        'id': 'PubMedID',
        'url': 'https://api.semanticscholar.org/v1/paper/'
    },
    'corpus': {
        'id': 'CorpusID',
        'url': 'https://api.semanticscholar.org/v1/paper/CorpusID:'
    }
}
