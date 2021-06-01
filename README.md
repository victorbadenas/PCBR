# PCBR

W3 MAI-SEL final project. Implementation of a CBR system for PC Specs recommendation.

## Installation

For automated instalation using conda on linux use:

```bash
source setup.sh
```

For the time being we only support python 3.7. Top install the environment:

```bash
conda create --name pcbr python=3.7
conda activate pcbr
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The bert as a service

```bash
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
mv uncased_L-12_H-768_A-12 models
rm uncased_L-12_H-768_A-12.zip
```

## Run

### Bert Server

In a separate shell run: ```bert-serving-start -model_dir models/ -num_worker=4 -p 8000```. A screen will also be very helpful to run the bert server. For more information, refer to [bert-as-a-service](https://github.com/hanxiao/bert-as-service)
