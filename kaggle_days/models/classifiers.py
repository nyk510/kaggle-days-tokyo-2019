from copy import deepcopy

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from optuna.trial import Trial
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegressionCV
from vivid.out_of_fold.base import BaseOutOfFoldFeature
from vivid.out_of_fold.boosting import LGBMClassifierOutOfFold, XGBoostClassifierOutOfFold
from vivid.out_of_fold.boosting.mixins import get_boosting_parameter_suggestions
from vivid.out_of_fold.ensumble import RFClassifierFeatureOutOfFold
from vivid.utils import timer

from .base import KaggleDaysMixin, BoostingOptunaSet
from .keras import CustomKerasClassifier


class LGBMCls(KaggleDaysMixin, LGBMClassifierOutOfFold):
    fit_verbose = 20

    initial_params = {
        'learning_rate': .05,
        'reg_lambda': .05,
        'n_estimators': 1000,
        'min_child_samples': 5,
        'colsample_bytree': .6,
        'subsample': .9,
        'num_leaves': 31,
        'importance_type': 'gain',
    }

    def call(self, df_source, y=None, test=False):
        if not test:
            self.user_id_group = df_source['user_id'].values

        if 'user_id' in df_source.columns:
            del df_source['user_id']
        if test:
            return self._predict_trained_models(df_source)

        x_train, y_train = df_source.values, y

        # Note: float32 にしないと dtype が int になり, 予測確率を代入しても 0 のままになるので注意
        pred_train = None
        params = self.get_best_model_parameters(x_train, y_train)

        for i, ((x_i, y_i), (x_valid, y_valid), (_, idx_valid)) in enumerate(
                self.get_folds(x_train, y_train, groups=None)):  # [NOTE] Group KFold is not Supported yet
            self.logger.info('start k-fold: {}/{}'.format(i + 1, self.num_cv))

            with timer(self.logger, format_str='Fold: {}/{}'.format(i + 1, self.num_cv) + ' {:.1f}[s]'):
                clf = self.fit_model(x_i, y_i, params, x_valid=x_valid, y_valid=y_valid, cv=i)

                if self.is_regression_model:
                    pred_i = clf.predict(x_valid).reshape(-1)
                else:
                    pred_i = clf.predict(x_valid, prob=True)

            if pred_train is None:
                pred_train = np.zeros(shape=(len(y), pred_i.shape[1]), dtype=np.float)
            pred_train[idx_valid] = pred_i
            self.fitted_models.append(clf)

        self.finish_fit = True
        # self.after_kfold_fitting(df_source, y, pred_train)
        df_train = pd.DataFrame(pred_train, columns=[str(self)])
        return df_train

    def _predict_trained_models(self, df_test):
        if not self.finish_fit:
            models = self.load_best_models()
        else:
            models = self.fitted_models

        kfold_predicts = [model.predict(df_test.values, prob=True) for model in models]
        preds = np.asarray(kfold_predicts).mean(axis=0)
        df = pd.DataFrame(preds)
        return df


class LGBMClsOptuna(BoostingOptunaSet):
    model_class = LGBMClassifier
    eval_metric = 'logloss'
    initial_params = deepcopy(LGBMCls.initial_params)

    def parameter_postprocess(self, params):
        params = deepcopy(params)
        params['num_leaves'] = int(2 ** params['max_depth'] * .7)
        return params

    def generate_model_class_try_params(self, trial: Trial):
        params = get_boosting_parameter_suggestions(trial)
        params = self.parameter_postprocess(params)
        params['n_jobs'] = 1
        return params

    def get_best_model_parameters(self, X, y):
        params = super(LGBMClsOptuna, self).get_best_model_parameters(X, y)
        params = self.parameter_postprocess(params)
        return params


class XGBCls(KaggleDaysMixin, XGBoostClassifierOutOfFold):
    initial_params = {
        'n_jobs': -1,
        'learning_rate': .05,
        'reg_lambda': .1,
        'n_estimators': 1000,
        'colsample_bytree': .7,
        'subsample': .7,
        'max_depth': 6
    }


class RFCls(KaggleDaysMixin, RFClassifierFeatureOutOfFold):
    initial_params = {'n_estimators': 125, 'max_features': 0.2, 'max_depth': 25, 'min_samples_leaf': 4, 'n_jobs': -1}


class ExtraTreeCls(KaggleDaysMixin, BaseOutOfFoldFeature):
    model_class = ExtraTreesClassifier
    initial_params = {'n_estimators': 100, 'max_features': 0.5, 'max_depth': 18, 'min_samples_leaf': 4, 'n_jobs': -1}


class LogisticCls(KaggleDaysMixin, BaseOutOfFoldFeature):
    model_class = LogisticRegressionCV
    initial_params = {
        'input_scaling': 'standard'
    }


class KerasCls(KaggleDaysMixin, BaseOutOfFoldFeature):
    model_class = CustomKerasClassifier
    initial_params = {
        'epochs': 100,
        'batch_size': 32,
        'input_scaling': 'standard'
    }
