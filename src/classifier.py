"""train as classification test"""
import os

import numpy as np
import pandas as pd
from vivid.featureset.molecules import MoleculeFeature, find_molecule

from kaggle_days.dataset import read_data
from kaggle_days.dataset import read_sample_submit
from kaggle_days.models.classifiers import LGBMCls
from kaggle_days.molecules import user_merge_molecule

if __name__ == '__main__':
    m = find_molecule('benchmark')[0]
    raw_feature = MoleculeFeature(m, root_dir='/analysis/data/checkpoint/classification')
    entry_feature = MoleculeFeature(user_merge_molecule, parent=raw_feature)

    train_df, y = read_data()
    test_df = read_data(test=True)
    origin = np.sort(np.unique(y))
    k_labels = np.arange(len(origin))
    y2k = dict(zip(origin, k_labels))
    y_labels = pd.Series(y).map(y2k).values

    clf = LGBMCls(name='lgbm_cls', parent=entry_feature)
    oof_df = clf.fit(train_df, y_labels)
    prob_predict = clf.predict(test_df).values
    predict = np.sum(prob_predict * origin, axis=1)

    sub_df = read_sample_submit()
    sub_df['age'] = predict
    sub_df.to_csv(os.path.join(clf.output_dir, 'predict.csv'), index=False)
