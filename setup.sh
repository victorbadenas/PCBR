#!/bin/sh

# conda environment creation
conda create -y --name pcbr python=3.6 && \
conda activate pcbr && \
python -m pip install --upgrade pip && \
pip install -r requirements.txt
