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

```bash
export MARIAN=$HOME/marian-dev/build
export MOSES_SCRIPTS=$HOME/marian-dev/examples/tools/moses-scripts/scripts
export TOOLS=$HOME/robust_bench/tools
export DATA=/data/almanach/user/$(whoami)
```

or in a conda environement (see
[conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#setting-environment-variables))

```bash
conda activate nmt
conda env config vars set MARIAN=$HOME/marian-dev/build
conda env config vars set MOSES_SCRIPTS=$HOME/marian-dev/examples/tools/moses-scripts/scripts
conda env config vars set TOOLS=$HOME/robust_bench/tools
conda env config vars set DATA=/data/almanach/user/$(whoami)

conda activate nmt
```

### NEF settings

Fairseq modules

```bash
#!/bin/bash
module load conda/5.0.1-python3.6 cuda/11.0 cudnn/8.0-cuda-11.0 cmake/3.10.1
export MKL_THREADING_LAYER=GNU
```

Marian modules

```bash
#!/bin/bash
module load gcc/7.3.0 cuda/9.2 conda/5.0.1-python3.6
```