# Environments

`requirements-analysis.txt` and `analysis-freeze.txt` describe the Python 3.13
CPU environment used for implementation-level tests and packaging checks.

`requirements-gpu-historical.txt` is the server's historical research
dependency list. It contains unpinned packages and a Git dependency, so it is
not a lockfile and is not presented as a clean installation recipe. A complete
GPU environment must be validated after the final model/checkpoint release
scope is decided.
