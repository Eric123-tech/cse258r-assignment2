"""Download pinned Last.fm parquet files atomically and record checksums."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = 'https://huggingface.co/datasets/matthewfranglen/lastfm-1k'

def main():
    raw = ROOT / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = raw / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    revision = manifest.get('revision')
    if not revision:
        with urllib.request.urlopen('https://huggingface.co/api/datasets/matthewfranglen/lastfm-1k', timeout=60) as r:
            revision = json.load(r)['sha']
    expected = {r['file']: r['sha256'] for r in manifest.get('files', [])}
    records = []
    for name in ['README.md', 'train.gz.parquet', 'valid.gz.parquet', 'test.gz.parquet']:
        target = raw / name
        url = f'{REPO}/resolve/{revision}/{name}'
        if not target.exists():
            print(f'Downloading {name}', flush=True)
            tmp = target.with_suffix(target.suffix + '.partial')
            with urllib.request.urlopen(url, timeout=180) as response, tmp.open('wb') as output:
                while chunk := response.read(8 * 1024 * 1024):
                    output.write(chunk)
            tmp.replace(target)
        digest = hashlib.sha256()
        with target.open('rb') as f:
            while chunk := f.read(8 * 1024 * 1024):
                digest.update(chunk)
        if name in expected and digest.hexdigest() != expected[name]:
            raise ValueError(f'Checksum mismatch for {name}; inspect/remove this file before retrying')
        records.append(dict(file=name, bytes=target.stat().st_size, sha256=digest.hexdigest(), url=url))
        print(f'{name}: {target.stat().st_size:,} bytes', flush=True)
    manifest_path.write_text(json.dumps(dict(repository=REPO, revision=revision, files=records), indent=2))

if __name__ == '__main__':
    main()
