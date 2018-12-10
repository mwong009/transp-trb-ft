## Transportation Forecasting
For TRBAM 2019 TRANSFOR19 Forecasting competition

### Setup and library installation

Requirements: 
- Python >= 3.5 (3.7 recommended)
- BLAS libraries (OpenBLAS, ATLAS, Intel MKL etc.) 
- gcc >= 4.9
- Git
- Python pip

example install on Debian Linux:

`apt install python3.7 python3.7-dev python3-pip g++ libblas-dev git`

to install basic required Python libraries run:

`pip3 install --user -r requirements.txt`

### Notes

- total data size = `(17856 rows × 1024 columns)`

- available date range = `2016-10-01 to 2016-12-01`

- speed forecasting range = `2016-12-01 00:00:00+08:00 to 2016-12-01 23:55:00+08:00` `6am - 10:55am, 4pm to 8:55pm`

- prediction boundaries: 
  * `108.94615156073496 < longitude < 108.94765628015638`
  * `34.2324012260476 < Latitude < 34.23940650580562`

- Dec 01, 2016 is a Thursday

### This is a 3 part series:

Use the following to navigate

- [Part 1: data processing](processing.ipynb)
- [Part 2: data preparation](preparation.ipynb)
- [Part 3: model training](training.ipynb)
