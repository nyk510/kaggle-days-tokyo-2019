import numpy as np
import pandas as pd
from vivid.featureset.atoms import StringContainsAtom, AbstractAtom
from vivid.featureset.encodings import CountEncodingAtom

from kaggle_days.dataset import read_data
from .dates import UedaAtom


class CopyAtom(AbstractAtom):
    def call(self, df_input, y=None):
        return df_input.copy()


class DirectAtom(AbstractAtom):
    """入力変数そのまま使う
    """
    use_columns = []

    def call(self, df_input, y=None):
        out_df = df_input[self.use_columns].copy().fillna(0)
        return out_df


class DateAtom(AbstractAtom):
    use_columns = ['date']

    def call(self, df_input, y=None):
        out_df = pd.DataFrame()
        date = pd.to_datetime(df_input['date'])
        dt = date.dt
        out_df['day_of_week'] = dt.dayofweek
        out_df['day_of_month'] = dt.day
        return out_df


class DevManuAtom(StringContainsAtom):
    queryset = {
        'er_dev_manufacture': ['Apple'],
        'er_dev_device_type': ['Mobile']
    }


class OneHotEncodingAtom(AbstractAtom):
    """use_columns に対して One Hot Encoding を実行する"""

    def __init__(self):
        super(OneHotEncodingAtom, self).__init__()
        self.master_df = None

    @property
    def is_fitted(self):
        return self.master_df is not None

    def call(self, df_input, y=None):
        out_df = pd.DataFrame()

        if not y is None:
            self.master_df = df_input.copy()

        for c in self.use_columns:
            x = df_input[c]
            vc = self.master_df[c].value_counts()
            cats = vc[vc > len(self.master_df) * .05].index
            cat = pd.Categorical(x, categories=sorted(cats))
            df_i = pd.get_dummies(cat, prefix=f'{c}_')
            df_i.columns = list(df_i.columns)
            out_df = pd.concat([out_df, df_i], axis=1)

        return out_df


class VersionsAtom(DirectAtom):
    use_columns = [
        'user_id',
        'er_dev_os_version',
        'er_geo_bc_flag',
        'er_dev_browser_version'
    ]


class NumericAtom(DirectAtom):
    use_columns = [
        'ig_ctx_red_elapsed_since_page_load',
        'ig_ctx_red_viewed_percent',
    ]


class BasicOneHotEncoding(OneHotEncodingAtom):
    use_columns = [
        'er_dev_device_name',
        'er_dev_browser_family',
        'er_dev_device_type',
        'er_dev_os_family',
        'er_geo_city_j_name',
        'er_geo_country_code',
        'er_geo_pref_j_name',
        'ig_ctx_product',
        'ig_usr_connection'
    ]


class BasicCountEncoding(CountEncodingAtom):
    use_columns = [
        'kiji_id', 'user_id', 'ig_ctx_red_viewed_percent',
        'ig_ctx_red_elapsed_since_page_load', 'er_geo_bc_flag',
        'ig_ctx_product', 'er_geo_pref_j_name', 'er_geo_city_j_name',
        'er_geo_country_code', 'er_dev_browser_family',
        'er_dev_browser_version', 'er_dev_device_name', 'er_dev_device_type',
        'er_dev_manufacture', 'er_dev_os_family', 'er_dev_os_version',
        'er_rfs_reffered_visit', 'er_rfs_service_name', 'er_rfs_service_type',
        'er_rfc_kiji_id_raw', 'ig_usr_connection'
    ]


class DateTimeAtom(AbstractAtom):
    use_columns = ['ts']

    def call(self, df_input, y=None):
        x = df_input['ts']
        x = pd.to_datetime(x)
        hours = x.dt.hour
        minutes = x.dt.minute
        time_val = hours * minutes / 60
        out_df = pd.DataFrame()
        out_df['ts_sin_hour'] = np.sin(time_val.values / 24. * 2 * np.pi)
        out_df['ts_cos_hour'] = np.cos(time_val.values / 24 * 2 * np.pi)
        return out_df


class TargetEncodingAtom(AbstractAtom):
    use_columns = ['kiji_id', 'company']

    def __init__(self):
        self.encodings_mapping = {}
        super(TargetEncodingAtom, self).__init__()

    def make_encoding_map(self, input_df: pd.DataFrame, y):
        _temp_df = input_df.copy()
        _temp_df['target'] = y

        for c in self.use_columns:
            _df = _temp_df.groupby(c).agg(['mean', 'std'])
            _df.columns = ['_'.join(x) for x in _df.columns.to_flat_index()]
            self.encodings_mapping[c] = _df

    def call(self, df_input, y=None):
        if y is not None:
            self.make_encoding_map(input_df=df_input, y=y)

        out_df = pd.DataFrame()
        for key, map_df in self.encodings_mapping.items():
            for c in map_df.columns:
                out_df[f'{key}_{c}'] = df_input[key].map(map_df[c])

        return out_df


class GroupByUserAtom(AbstractAtom):
    """この時点では user_id は残ってる"""

    def call(self, df_input, y=None):
        df = df_input.groupby('user_id').agg(['mean', 'sum', 'max', 'min', 'std', 'nunique']).sort_values('user_id')
        df.columns = ['_'.join(x) for x in df.columns.to_flat_index()]
        df = df.reset_index()
        additional_atoms = [UedaAtom()]
        for atom in additional_atoms:
            if y is None:
                input_df = read_data(test=True)
            else:
                input_df, _ = read_data()
            df = pd.merge(df, atom.generate(input_df, y=None), on='user_id', how='left')
        return df.reset_index(drop=True)

    def _post_generate(self, *args, **kwargs):
        pass
