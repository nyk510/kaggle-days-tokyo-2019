import os
from typing import List

import pandas as pd
import seaborn as sns
from vivid.core import AbstractFeature
from vivid.core import EnsembleFeature, MergeFeature
from vivid.featureset.molecules import MoleculeFeature, Molecule
from vivid.metrics import regression_metrics
from vivid.out_of_fold.boosting.block import create_boosting_seed_blocks

from .dataset import read_sample_submit
from .feature_base import KaggleDaysFeature
from .models import regressors
from .molecules import user_merge_molecule
from .utils import SimpleFillNaFeature


class TrainComposer:
    def __init__(self, molecule: Molecule, suffix=None, simple=False):
        self.single_models = []

        root_dir = os.path.join('/analysis/data/checkpoints', molecule.name)
        if suffix:
            root_dir = root_dir + '_' + suffix

        if simple:
            root_dir = root_dir + '_simple'

        self.root_dir = root_dir
        self.raw_feature = KaggleDaysFeature(molecule, root_dir=root_dir)
        self.entry_feature = KaggleDaysFeature(user_merge_molecule, parent=self.raw_feature)
        self.fillna_entry = SimpleFillNaFeature(agg='mean', parent=self.entry_feature, name='fillna')

        if simple:
            self.single_models = [regressors.LGBMReg(name='lgbm_simple', parent=self.entry_feature),
                                  regressors.LGBMReg(name='lgbm_poisson', parent=self.entry_feature,
                                                     add_init_param={'objective': 'poisson'}),
                                  regressors.LGBMReg(name='lgbm_gamma', parent=self.entry_feature,
                                                     add_init_param={'objective': 'gamma'}), ]
            self.stacking_models = []
            return

        self.single_models += [
            regressors.LGBMReg(name='lgbm', parent=self.entry_feature),
            regressors.LGBMReg(name='lgbm_simple', parent=self.entry_feature),
            regressors.LGBMReg(name='lgbm_poisson', parent=self.entry_feature,
                               add_init_param={'objective': 'poisson'}),
            regressors.LGBMReg(name='lgbm_gamma', parent=self.entry_feature,
                               add_init_param={'objective': 'gamma'}),

            regressors.RidgeReg(name='logistic', parent=self.fillna_entry),
            regressors.KerasReg(name='keras', parent=self.fillna_entry),
            *create_boosting_seed_blocks(parent=self.entry_feature, prefix='lgbm', feature_class=regressors.LGBMReg),
            *create_boosting_seed_blocks(parent=self.entry_feature, prefix='lgbm_poisson',
                                         feature_class=regressors.LGBMReg,
                                         add_init_params={'objective': 'poisson'}),
            regressors.XGBReg(name='xgb', parent=self.entry_feature),
            regressors.ExtraReg(name='extra_tree', parent=self.fillna_entry),
            regressors.RFReg(name='rf', parent=self.fillna_entry),
        ]

        self.merge_feature = MergeFeature([*self.single_models, self.fillna_entry],
                                          name='merged',
                                          root_dir=root_dir)

        self.stacking_models = [
            regressors.RidgeReg(parent=self.merge_feature, name='stacked_ridge',
                                add_init_param={'input_scaling': 'standard'}),
            regressors.XGBReg(name='stacked_xgb', parent=self.merge_feature)
        ]
        ens = EnsembleFeature(self.stacking_models[:], name='ensemble_of_stackings',
                              root_dir=self.merge_feature.root_dir)
        self.stacking_models.append(ens)

    @property
    def models(self) -> List[AbstractFeature]:
        return [
            *self.single_models,
            *self.stacking_models
        ]

    def fit(self, train_df, y, test_df) -> [pd.DataFrame, dict]:
        """
        学習の実行

        Args:
            train_df:
            y:
            test_df:

        Returns:
            metric dataframe and predicts dict to test set.

            metric_df:
                index: model_name
                columns: metric_name (in regression metrics)

            predicts:
                key: model_name
                value: np.ndarray. shape = (n_test,)
        """
        metric_df = None
        oof_all_df = pd.DataFrame()

        predict = {}

        for model in self.models:
            oof_df = model.fit(train_df, y)
            pred_df = model.predict(test_df)
            if model.is_recording:
                sub_df = read_sample_submit()
                sub_df['age'] = pred_df.values[:, 0]
                sub_df.to_csv(os.path.join(model.output_dir, 'predict.csv'), index=False)

            predict[model.name] = pred_df.values[:, 0]

            oof_all_df = pd.concat([oof_all_df, oof_df], axis=1)

            metric_i = regression_metrics(y, oof_df.values[:, 0]).rename(columns={'score': model.name})
            if metric_df is None:
                metric_df = metric_i
            else:
                metric_df = pd.concat([metric_df, metric_i], axis=1)

        metric_df = metric_df.T.sort_values('rmse')
        metric_df.to_csv(os.path.join(self.root_dir, 'metrics.csv'))
        oof_all_df.to_csv(os.path.join(self.root_dir, 'out_of_fold.csv'), index=False)

        self.out_of_fold_df_ = oof_all_df
        self.metric_df_ = metric_df

        try:
            g = sns.clustermap(self.out_of_fold_df_.corr(), cmap='viridis')
            g.fig.tight_layout()
            g.fig.savefig(os.path.join(self.root_dir, 'out-of-fold-cluster.png'), dpi=120)
        except Exception as e:
            print(e)

        return metric_df, predict
