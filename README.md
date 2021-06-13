# PCBR

W3 MAI-SEL final project. Implementation of a CBR system for PC Specs recommendation.

## Installation

For automated instalation using conda on linux use:

```bash
source setup.sh
```

For the time being we only support python 3.7. Top install the environment:

```bash
conda create --name pcbr python=3.6
conda activate pcbr
python -m pip install --upgrade pip
pip install -r requirements.txt
```

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

## Run

```bash
python interface/welcomePage.py
```
