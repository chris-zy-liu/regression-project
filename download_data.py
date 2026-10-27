"""Download an unmodified Census workbook and record its provenance."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import requests

URL = 'https://www.census.gov/hfp/btos/downloads/Sector%20by%20Employment%20Size%20Class.xlsx'


def download(destination, url=URL):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=(15, 180))
    response.raise_for_status()
    if not response.content.startswith(b'PK'):
        raise ValueError('Census did not return an XLSX workbook; existing data were not changed.')
    import pandas as pd
    from io import BytesIO
    book = pd.ExcelFile(BytesIO(response.content))
    required = {'Response Estimates', 'Response Standard Errors', 'Collection and Reference Dates'}
    if not required.issubset(book.sheet_names):
        raise ValueError('Unexpected workbook schema; existing data were not changed.')
    destination.write_bytes(response.content)
    metadata = {'url': url, 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                'sha256': hashlib.sha256(response.content).hexdigest(), 'bytes': len(response.content),
                'note': 'Downloaded vintage; the original October 3 snapshot was not supplied.'}
    destination.with_suffix('.provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/sector_size.xlsx')
    parser.add_argument('--url', default=URL)
    args = parser.parse_args()
    download(args.output, args.url)
