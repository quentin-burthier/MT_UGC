# Robustness benchmark of Machine Translation methods

## Workspace settings

Code tested with

- gcc 7.3.0
- CUDA 9.2
- marian-dev (commit [467b15e](https://github.com/marian-nmt/marian-dev/commit/467b15e2b94b7c7b25ceaee764f790d8faaeabf2))
- Python 3.8 with packages
  - sacrebleu 1.14.4
  - sentencepiece 0.1.91
  - langid 1.1.6

### Environment variables

Please set `MARIAN`, `TOOLS` and `DATA` environment variables to the correct locations.
For instance if everything is in `~/` and you are working on the MTNT dataset

```bash
export MARIAN=$HOME/marian-dev/build
export TOOLS=$HOME/marian-dev/examples/tools
export DATA=$HOME/robust_bench/data
```

or in a conda environement (see
[conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#setting-environment-variables))

```bash
conda activate nmt
conda env config vars set MARIAN=$HOME/marian-dev/build
conda env config vars set TOOLS=$HOME/marian-dev/examples/tools
conda env config vars set DATA=$HOME/robust_bench/data
conda activate nmt
```
