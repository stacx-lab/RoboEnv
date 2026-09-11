"""Freeze only reusable code/configuration and verify every copied byte."""

import hashlib
import json
import shutil
from pathlib import Path

CONTENT = ('.kinex/tool', '.kinex/skill', '.kinex/lib', '.kinex/python', '.kinex/kinex.json', 'AGENTS.md')


def inventory(root, paths):
    root = Path(root)
    result = {}
    for relative in paths:
        base = root / relative
        if not base.exists():
            continue
        files = [base] if base.is_file() else sorted(base.rglob('*'))
        for file in files:
            if file.is_symlink():
                raise ValueError('Frozen content cannot contain symlinks')
            if not file.is_file() or '__pycache__' in file.parts or file.suffix == '.pyc':
                continue
            result[file.relative_to(root).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
    return result


def devices(workspace):
    result = inventory(workspace, ['.kinex/devices', '.kinex/workspace.yaml'])
    return {key: value for key, value in result.items() if not key.endswith('/connection.json')}


def freeze(workspace, destination, *, identity, tool, arguments):
    source, destination = Path(workspace), Path(destination)
    if destination.exists():
        raise ValueError('Frozen artifact already exists')
    manifest = inventory(source, CONTENT)
    for required in ('.kinex/python/pyproject.toml', '.kinex/python/uv.lock'):
        if required not in manifest:
            raise ValueError('Python environment must be locked before freezing')
    if f'.kinex/tool/{tool}/tool.yaml' not in manifest:
        raise ValueError('Selected controller must be a reusable workspace tool')
    destination.mkdir(parents=True)
    for relative in manifest:
        output = destination / 'content' / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, output)
    record = {'schema':'kinex/controller-artifact/1','identity':identity,'files':manifest,
              'devices':devices(source),'tool':tool,'args':arguments}
    (destination/'manifest.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
    return record


def restore(artifact, workspace, *, identity, development=False):
    artifact, workspace = Path(artifact), Path(workspace)
    record = json.loads((artifact/'manifest.json').read_text())
    if ((record['identity'] != identity and not development)
            or inventory(artifact/'content',CONTENT) != record['files']):
        raise ValueError('Frozen artifact or dependency identity changed')
    if devices(workspace) != record['devices']:
        raise ValueError('Validation device profiles/drivers differ from the frozen artifact')
    for relative in record['files']:
        target = workspace / relative
        if development and relative == 'AGENTS.md' and target.exists():
            continue  # Keep the newly prepared development workspace's operating instructions.
        if target.exists() and target.read_bytes() != (artifact/'content'/relative).read_bytes():
            raise ValueError(f'Fresh workspace contains conflicting content: {relative}')
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(artifact/'content'/relative,target)
    return record


def unchanged(workspace, artifact):
    record=json.loads((Path(artifact)/'manifest.json').read_text())
    return inventory(workspace,CONTENT)==record['files'] and devices(workspace)==record['devices']
