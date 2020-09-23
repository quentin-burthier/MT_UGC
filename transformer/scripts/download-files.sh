#!/bin/bash -v

# MTNT data
wget -nc -O $DATA/MTNT.1.1.tar.gz \
    https://github.com/pmichel31415/mtnt/releases/download/v1.1/MTNT.1.1.tar.gz

# extract data
tar -C $DATA -xf $DATA/MTNT.1.1.tar.gz

# clean
rm -r $DATA/MTNT.1.1.tar.gz
