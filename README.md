# Robustness benchmark of Machine Translation methods

## Workspace settings

Code tested with

- gcc 7.3.0
- CUDA 9.2
- marian-dev (commit [467b15e](https://github.com/marian-nmt/marian-dev/commit/467b15e2b94b7c7b25ceaee764f790d8faaeabf2))
- Python 3.6 (langid 1.1.6)

### Environment variables

Please set `MARIAN`, `TOOLS` and `DATA` environment variables to the correct locations.
For instance if everything is in `~/` and you are working on the MTNT dataset

```bash
export MARIAN=$HOME/marian/build
export TOOLS=$HOME/marian-dev/examples/tools
export DATA=$HOME/data/MTNT
```
