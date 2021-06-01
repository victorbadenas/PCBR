#!/bin/sh

# conda environment creation
conda create -y --name pcbr python=3.6 && \
conda activate pcbr && \
python -m pip install --upgrade pip && \
pip install -r requirements.txt

status=$?
[ $status -eq 0 ] && echo "conda environment creation succeeded" || return $status

# download bert model
rm -ri models/
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
mv uncased_L-12_H-768_A-12 models
rm uncased_L-12_H-768_A-12.zip
