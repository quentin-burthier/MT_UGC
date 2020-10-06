#!/bin/bash

function parse_cli () {
PARAMS=""
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
        shift 1
    fi
    ;;
    -r|--ratio)
    if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        ratio=.$2
        shift 2
    else
        ratio=""
        shift 1
    fi
    ;;
    --gpus)
        if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
            while [ -n "$2" ] && [ ${2:0:1} != "-" ]; do
                gpus="$gpus $2"
                shift 1
            done
        else
        # no gpus
        shift 1
    fi
      ;;
    -*|--*=) # unsupported flags
        echo "Error: Unsupported flag $1" >&2
        exit 1
        ;;
    *) # preserve positional arguments
        PARAMS="$PARAMS $1"
        shift
        ;;
  esac
done
# set positional arguments in their proper place
eval set -- "$PARAMS"
}
