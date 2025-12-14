# Contextual Bandits for Internet Banner Selection

This repository contains the Python implementations of the algorithms described in the final report, as well as the code used to generate the figures and plots presented in the report.


## Repository Contents

- Implementations of contextual bandit algorithms (LinUCB, LinTS, baselines)
- Simulation code for synthetic environments
- Evaluation code for real-world logged bandit data
- Plotting code used to generate the figures included in the report


## Dataset Setup (Required)

To run the experiments on real data, you must download the **Open Bandit Dataset**.

### Steps:
1. Download the dataset from:  
   https://research.zozo.com/data.html

2. Unzip the downloaded files.

3. Place the unzipped dataset directory in the project working directory (i.e., the root directory of this repository).

The code assumes the dataset is available locally.


## Installation

It is recommended to use a virtual environment.

Install the required Python dependencies with:

```bash
pip install -r requirements.txt
```

The code was tested with Python 3.9.


## How to Run

To start the simulations:

```bash
python run.py
```

This will perform the simulations and produce the plots used in the report. The resulting figures will be saved in a `plots/` directory, which is created automatically if it does not already exist.

