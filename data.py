import pandas as pd

class read_data:
    def __init__(self, file):
        # Load data
        raw_data = pd.read_csv(file)
        # Drop row contain Blank or Null or NaN value
        raw_data = raw_data.dropna()
        # Reset index in dataset
        raw_data = raw_data.reset_index()
        # Select main Columns
        self.dataset = raw_data.loc[:,
                       ['nearc4', 'educ', 'age', 'black', 'wage', 'IQ', 'married', 'exper', 'lwage', 'expersq']].copy()

    def load_data(self):
        return self.dataset
