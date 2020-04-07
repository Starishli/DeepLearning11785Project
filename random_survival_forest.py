import os
import pandas as pd

from source import DATA_DIR


a = pd.read_csv(os.path.join(DATA_DIR, "support8873.csv"))
print(a.head())