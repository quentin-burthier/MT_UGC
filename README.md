# Robustness benchmark of Machine Translation methods

## General settings

Python 3.6 with packages
  - sacrebleu 1.14.4
  - sentencepiece 0.1.91
  - fastText
  - nltk 3.5
  - pandas 1.1.5 (only required by `tools/compare_lexicon.py`)


```bash
export MOSES_SCRIPTS=$HOME/marian-dev/examples/tools/moses-scripts/scripts
export TOOLS=$HOME/robust_bench/tools
export DATA=/data/almanach/user/$(whoami)
```

or in a conda environement (see
[documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#setting-environment-variables))

```bash
conda activate nmt
conda env config vars set MOSES_SCRIPTS=$HOME/marian-dev/examples/tools/moses-scripts/scripts
conda env config vars set TOOLS=$HOME/robust_bench/tools
conda env config vars set DATA=/data/almanach/user/$(whoami)

conda activate nmt
```
## Marian

marian-dev (commit [467b15e](https://github.com/marian-nmt/marian-dev/commit/467b15e2b94b7c7b25ceaee764f790d8faaeabf2))

```bash
export MARIAN=$HOME/marian-dev/build
```

- gcc 7.3.0
- CUDA 9.2

## Fairseq

c8a0659

- Pytorch 1.5.0
- fairseq (commit [c8a0659](https://github.com/pytorch/fairseq/commit/c8a0659))
- CUDA 11.0
- CUDNN 8.0
- cmake 3.10.1

```bash
export MKL_THREADING_LAYER=GNU
```