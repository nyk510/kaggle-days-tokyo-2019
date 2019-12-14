"""時間に関係するもの"""
import numpy as np
import pandas as pd
from vivid.featureset import AbstractAtom


class DateDiffAtom(AbstractAtom):
    def call(self, df_input, y=None):
        _df = df_input[['user_id', 'ts']]
        _df['ts'] = pd.to_datetime(_df['ts'])
        _df = _df.groupby('user_id').diff()
        seconds = _df['ts'].dt.seconds
        out_df = pd.DataFrame()
        out_df['ts_diff'] = seconds / 60 ** 2
        out_df['ts_under_5min'] = seconds < 60 * 5
        return out_df


class UedaAtom(AbstractAtom):
    def call(self, input_df, y=None):
        date = pd.to_datetime(input_df['ts'])

        hours = date.dt.hour
        bins = np.linspace(0, 24, 9)
        _df = pd.get_dummies(pd.cut(hours, bins=bins))
        _df = pd.DataFrame(_df.values, dtype=np.int)
        _df = _df.astype('float')
        _df['user_id'] = input_df['user_id'].values
        _df = _df.groupby('user_id').sum()
        X = _df.values / _df.sum(axis=1).values.reshape(-1, 1)

        out_df = pd.DataFrame(X, columns=['normalized_by_user_ts_{}'.format(b) for b in bins[1:]])
        out_df['user_id'] = _df.index
        return out_df

    def _post_generate(self, df_input, df_out):
        pass
