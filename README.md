# PCBR: PC Parts Recommender

<p align="center">
  <img src="logo.png" alt="PCBR logo" width="250"/>
</p>

W3 MAI-SEL final project. Implementation of a CBR system for PC Specs recommendation.

## Installation

For automated instalation using conda on linux use:

```bash
source setup.sh
```

The application was developed in python3.6 using Linux and MacOS. Those OS are strongly recommended. The application has been spot-checked in Windows 10, but some cross-platoform incompatibilities in PyQt bindings cause the GUI to freeze at times. We strongly recommend MacOS os Linux for this reason. For the time being we only support python 3.6. To install the environment:

```bash
conda create --name pcbr python=3.6
conda activate pcbr
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run with GUI

The following command will run the pcbr software using the graphical user interface.

```bash
python interface/welcomePage.py
```

additionally, a simple help command will display the arguments available through the command line:

```text
$ python interface/welcomePage.py --help
usage: welcomePage.py [-h] [-c CBL_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -c CBL_PATH, --cbl_path CBL_PATH
                        path to the case base library
```

## Run with CLI

The following command will run the pcbr software using the command line interface.

```bash
cd src/ && python pcbr.py
```

additionally, a simple help command will display the arguments available through the command line:

```text
$ python pcbr.py --help
usage: pcbr.py [-h] [-g] [-c CBL_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -g, --generator       run random case generator
  -c CBL_PATH, --cbl_path CBL_PATH
                        path to the case base library
```

## Directory structure

The directory has the structure of the following tree:

```text
.
├── data
│   ├── cpu_table.csv
│   ├── feature_relevance.csv
│   ├── feature_scalers.json
│   ├── gpu_table.csv
│   ├── hdd_table.csv
│   ├── mock_requests.tsv
│   ├── optical_drive_table.csv
│   ├── pc_specs.csv
│   ├── ram_table.csv
│   └── ssd_table.csv
├── doc
│   ├── PCBR Final Presentation.pdf
│   ├── PCBR_User_Manual.pdf
│   └── SEL_PW3-Team4Report.pdf
├── interface
│   ├── mainWindow_comp.py
│   ├── tableWindow.py
│   ├── multichoice.py
│   └── welcomePage.py
├── LICENSE
├── README.md
├── requirements.txt
├── scripts
│   └── compute_prices_for_cases.py
├── setup.sh
├── src
│   ├── adapt_pc.py
│   ├── constraints.py
│   ├── data
│   │   ├── mapper.py
│   │   └── preprocessor.py
│   ├── neighbors
│   │   ├── knn.py
│   │   ├── metrics.py
│   │   ├── nn.py
│   │   └── utils.py
│   ├── pcbr.py
│   ├── user_request.py
│   └── utils
│       ├── __init__.py
│       ├── io.py
│       └── typing.py
└── test
    ├── __init__.py
    ├── test_nearest_neighbors.py
    └── test_user_input.py
```

And the contents of the folders are the following:

- **data**: *csv, json, tsv* and anything data related. Contains the initial cbl in `data/pc_specs.csv`, the tables with component data in `data/*_table.csv`, the feature scalers info and normalization parameters per feature in json format in `data/feature_scalers.json`, the feature relevance matrix in csv format in `data/feature_relevance.csv` and finally the mock requests file in `data/mock_requests.tsv`.
- **doc**: delivery documents in pdf format.
- **interface**: GUI PyQt5 files. It contains the welcome page and starting executable file `interface/welcomePage.py` as well as other window files.
- **scripts**: only contains scripts used during the project. The only file is a script that given a cbl and the component's tables, computes the price for each instance.
- **src**: main code folder for the project.
  - **data**: objects and functions to preprocess and map values from the data. `preprocessor.py` contains the functions to load and preprocess the initial cbl and the component tables as well as the feature scalers' parameters to preprocess the cbl. `mapper.py` contains a utility object that allows the caller to map from one column to another of the csv returning the closest match to the input.
  - **neighbors**: contains the objects mainly used for supervised and unsupervised nearest neighbors algorithms. `knn.py` contains the supervised k-Nearest Neighbors implementation used for the retrieve step. `nn.py` contains the unsupervised nearest neighbors algorithm used to construct similarity graphs.
  - **utils**: collection of io and typing utilities.
  - `adapt_pc.py`: contains the class used to perform the Reuse function and adapt a case to the input request
  - `constraints.py`: contains the class used to apply the constraints to the system's data.
  - `pcbr.py`: main pcbr object. Running this file runs the cli interface for the project.
  - `user_request.py`: contains the object for storing and processing user requests into the format that the system expects.
- **test**: unittest folder.

## Unittests

Unittest are located in the `./test/` folder. Each unittest must follow the following guidelines:

```python
import unittest

class TestCase(unittest.TestCase)):
    def test_whatever(self):
        a = 1
        # do something with a
        self.assertEqual(a, 1) # replace 1 with expected value

if __name__ == "__main__":
    unittest.main()
```

To run all unittests: `python -m unittest discover -v`
