from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
from vivid.featureset.atoms import AbstractAtom


class TargetEncodingAtom(AbstractAtom):
    n_fold = 10

    def __init__(self, use_columns: List[str]):
        super(TargetEncodingAtom, self).__init__()

        self.mapping_df_ = None
        self.use_columns = use_columns

    def create_mapping(self, input_df, y):
        self.mapping_df_ = {}
        fold = GroupKFold(self.n_fold)
        out_df = pd.DataFrame()

        for col_name in tqdm(self.use_columns, total=len(self.use_columns)):
            keys = input_df[col_name].unique()
            target = input_df['age']
            values = input_df[col_name]

            oof = np.zeros_like(values, dtype=np.float)
            mapping_df = None

            for idx_train, idx_valid in fold.split(input_df, None, groups=input_df['user_id'].values):
                _df = target[idx_train].groupby(values[idx_train]).mean()
                _df = _df.reindex(keys)
                _df = _df.fillna(_df.median())
                oof[idx_valid] = input_df[col_name][idx_valid].map(_df.to_dict())

                if mapping_df is None:
                    mapping_df = _df
                else:
                    mapping_df = mapping_df + _df

            out_df[col_name] = oof

            self.mapping_df_[col_name] = mapping_df / self.n_fold

        return self.mapping_df_, out_df

    def call(self, input_df, y=None):
        if y is None:
            out_df = self._predict(input_df)
        else:
            mapping, out_df = self.create_mapping(input_df, y)
            self.mapping_df_ = mapping

        return out_df.add_prefix('TE_')

    def _predict(self, input_df):
        if self.mapping_df_ is None:
            raise ValueError('Must Learn before predict')

        out_df = pd.DataFrame()

        for c in self.use_columns:
            out_df[c] = input_df[c].map(self.mapping_df_[c])
        return out_df
