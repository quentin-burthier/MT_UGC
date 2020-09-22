#!/bin/bash -v

mkdir -p data
cd data

# MTNT data
wget -nc https://github.com/pmichel31415/mtnt/releases/download/v1.1/MTNT.1.1.tar.gz

# extract data
tar -xf MTNT.1.1.tar.gz

# clean
rm -r .tgz 

cd ..
