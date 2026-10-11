#!/bin/bash
set -euo pipefail
# Run inside the existing 4rc environment after installing upstream 4RC.
python -m pip install 'websockets>=14' 'msgpack>=1.0.8' 'msgpack-numpy>=0.4.8' 'pydantic>=2.5' pyyaml scipy 'viser>=1.1,<2'
