#!/bin/bash
# Adapted from https://medium.com/@Drew_Stokes/bash-argument-parsing-54f3b81a6a8f

function parse_cli() {
shuffle=true
ratio="1.0"
while (( "$#" )); do
  case "$1" in
    -s|--source)
    if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        src=$2
        shift 2
    else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
    fi
    ;;
    -t|--target)
    if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        tgt=$2
        shift 2
    else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
    fi
    ;;
    -v|--voc_sz)
    if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        voc_sz=$2
        shift 2
    else
        voc_sz=""
        shift
    fi
    ;;
    -d|--dataset)
    if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        dataset=$2
        shift 2
    else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
    fi
    ;;
    --no-shuffle)
        shuffle=false
        shift
    ;;
    -r|--ratio)
    if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        ratio=$2
        shift 2
    else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
    fi
    ;;
    -m|--model)
    if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        model_dir=$2
        shift 2
    else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
    fi
    ;;
    -btm|--back-translation-model)
    if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        bt_model=$2
        back_translate=true
        bt="bt."  # hack for input file naming
        shift 2
    else
        back_translate=false
        shift 1
    fi
    ;;
    -o|--output-dir)
    if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        output_dir=$2
        shift 2
    else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
    fi
    ;;
    -vo|--val-output-dir)
    if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        val_output_dir=$2
        shift 2
    else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
    fi
    ;;
    --gpus)
    if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        while [ -n "$2" ] && [ ${2:0:1} != "-" ]; do
            gpus="$gpus $2"
            shift
        done
    fi
    shift
    ;;
    -*|--*=) # unsupported flags
        echo "Error: Unsupported flag $1" >&2
        exit 1
    ;;
    *) # preserve positional arguments
        echo "Warning: Flag $1 ignored."
        shift
    ;;
  esac
done
}

function set_dataset_args() {
case $dataset in
    MTNT)
        if $shuffle; then
            mtnt=$DATA/MTNT_reshuffled
        else
            mtnt=$DATA/MTNT
        fi
        dir=$mtnt/$src-$tgt.$ratio
        mono_dir=$mtnt/monolingual
        bt_dir=$mtnt/back-translated
        preprocess_args="$src $tgt $dir $mtnt $ratio"
    ;;
    NCE)
        dir=$DATA/europarl_nc.$src-$tgt
        preprocess_args="$src $tgt $dir"
    ;;
    Europarl|News-Commentary|OpenSubtitles)
        dir=$DATA/$dataset.$src-$tgt
        preprocess_args="$src $tgt"
    ;;
    Crapbank)
        dir=$DATA/Crapbank
        preprocess_args="$DATA/OpenSubtitles.en-fr/truecaser/model"
    ;;
esac
formated_date=$(date +"%d.%m.%Y_%T")
if [ ! "$output_dir" ]; then output_dir=$dir/output_$formated_date; fi
if [ ! "$val_output_dir" ]; then val_output_dir=$dir/val_output_$formated_date; fi
if [ ! "$bt_dir" ]; then bt_dir=$dir/bt_$formated_date; fi
}
