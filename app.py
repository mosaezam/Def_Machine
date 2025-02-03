# Working with real data

# Look at the big picture.  (1-> Localisation, 2-> Number of rooms and so on...)

# Get the data.
   #Load data

import pandas as pd
import os
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data_(housing_path='D:/Machine leraningTraining/End-to-End Machine Learning Project'):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
def load_housing_data():
    tarball_path = Path("D:/Machine leraningTraining/End-to-End Machine Learning Project/datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("D:/Machine leraningTraining/End-to-End Machine Learning Project/datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="D:/Machine leraningTraining/End-to-End Machine Learning Project/datasets")
    return pd.read_csv(Path("D:/Machine leraningTraining/End-to-End Machine Learning Project/datasets/housing/housing.csv"))
housing = load_housing_data()

# Explore and visualize the data to gain insights.

print(housing.head())

# Prepare the data for machine learning algorithms.

# Select a model and train it.

# Fine-tune your model.

# Present your solution.

# Launch, monitor, and maintain your system.

