#!/bin/bash

function to_latex() {
    echo "\begin{tabular}{lrrrr}"
    echo "\toprule \\\\"
    echo "Dataset & Lines & Mean line length & Mean token length & Unique tokens \\\\"
    echo "\midrule \\\\"
    for dataset in $@
    do
        compute_stats $dataset
    done
    echo "\bottomrule"
    echo "\end{tabular}"
}

function compute_stats() {
    local file=$1

    local lines tokens characters filename
    read lines tokens characters filename <<< $( wc $file )

    local avg_token_size=$(( $characters / $tokens ))
    local avg_line_size=$(( $tokens / $lines ))

    local n_unique_tokens=$(count_unique_tokens $file)
    local unique_tokens_proportion=$(( 100 * $n_unique_tokens / $tokens ))

    echo "$file & $lines & $avg_line_size & $avg_token_size & $unique_tokens_proportion \\\\"
}

function count_unique_tokens() {
    # https://stackoverflow.com/a/34377954
    local file=$1
    
    tr ':;,?!\"' ' ' < $file | tr -s ' ' '\n' \
    | awk '!a[$0]++{c++} END{print c}'
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    to_latex "$@"
fi
