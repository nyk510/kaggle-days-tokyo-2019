import json
import os

import pandas as pd
from vivid.core import AbstractFeature


class SimpleFillNaFeature(AbstractFeature):
    def __init__(self, agg='mean', *args, **kwargs):
        super(SimpleFillNaFeature, self).__init__(*args, **kwargs)
        self.fill_values = None
        self.agg = agg

    @property
    def fill_value_path(self):
        if self.is_recording:
            return os.path.join(self.output_dir, 'fill_value.json')

    def call(self, df_source: pd.DataFrame, y=None, test=False):
        if not test:

            self.fill_values = df_source.agg(self.agg).to_dict()
            with open(self.fill_value_path, 'w') as f:
                json.dump(self.fill_values, f, indent=4)

        else:
            with open(self.fill_value_path) as f:
                self.fill_values = json.load(f)

        return df_source.fillna(self.fill_values)
