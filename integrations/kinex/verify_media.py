"""CLI bridge into the same media validator used by owner campaigns."""
import json
import sys
from pathlib import Path
from recording import verify

if len(sys.argv)!=4:raise SystemExit('verify_media.py WORKSPACE RUNNER_JSON OUTPUT_JSON')
output=Path(sys.argv[3])
if output.exists():raise ValueError('Preserve existing media receipts')
report=verify(sys.argv[1],json.loads(Path(sys.argv[2]).read_text())['recording'])
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
