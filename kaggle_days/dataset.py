import os

import numpy as np
import pandas as pd

INPUT_DIR = '/analysis/data/raw'


def read_data(test=False):
    filename = 'train.csv' if not test else 'test.csv'
    df = pd.read_csv(os.path.join(INPUT_DIR, filename))
    if test:
        return df

    _df = df[['age', 'user_id']].groupby('user_id').mean()
    y = _df.sort_values('user_id')['age'].values
    return df, y


def read_kiji():
    return pd.read_csv(os.path.join(INPUT_DIR, 'kiji_metadata.csv'))


def read_sample_submit():
    return pd.read_csv(os.path.join(INPUT_DIR, 'sample_submission.csv'))


def generate_next_step_dataset(y_pred=None) -> [pd.DataFrame, np.ndarray]:
    """
    Generate New Psedo Labeling Dataset On Next Step

    :param y_pred: predict values. **MUST** sorted by user_id.
        By default in sample submission, user_id is sorted.
    :return:
        new Dataset.
    """
    if y_pred is None:
        return read_data()

    sub_df = read_sample_submit()

    train_df, y = read_data()
    test_df = read_data(test=True)

    train_user_ids = np.sort(train_df['user_id'].unique())
    assert len(train_user_ids) == len(y)

    # user と予測値 (学習データ) を作成
    train_pred = pd.DataFrame([train_user_ids, y]).T

    # テストデータでの user_id と予測値のデータを作成
    test_pred = pd.DataFrame([sub_df['user_id'].values, y_pred]).T
    test_pred.columns = [0, 1]

    # concat して user_id で sort して次のデータセットでのラベルとする
    # データ自体も結合して返す
    all_pred_df = pd.concat([train_pred, test_pred], ignore_index=True)
    y = all_pred_df.sort_values(0)[1].values
    df = pd.concat([train_df, test_df], ignore_index=True, sort=True)
    df = df.sort_values('user_id').reset_index(drop=True)
    return df, y
