"""Tuning LightGBM Model by Optuna"""

import os

from vivid.featureset.molecules import MoleculeFeature
from vivid.featureset.molecules import find_molecule

from kaggle_days.dataset import read_data, read_sample_submit
from kaggle_days.models.regressors import LGBMRegOptuna
from kaggle_days.molecules import user_merge_molecule

if __name__ == '__main__':
    train_df, y = read_data()
    test_df = read_data(test=True)

    m = find_molecule('benchmark')[0]

    feat = MoleculeFeature(molecule=m, root_dir=f'/analysis/data/optuna_lgbm_{m.name}')
    feat_group = MoleculeFeature(molecule=user_merge_molecule, parent=feat)
    model = LGBMRegOptuna(name='lgbm_optuna',
                          n_trials=200,
                          parent=feat_group)
    model.fit(train_df, y)
    pred_df = model.predict(test_df)

    sub_df = read_sample_submit()
    sub_df['age'] = pred_df.values[:, 0]
    sub_df.to_csv(os.path.join(model.output_dir, 'predict.csv'), index=False)
