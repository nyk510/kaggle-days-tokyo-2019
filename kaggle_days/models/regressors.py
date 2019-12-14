from copy import deepcopy

from lightgbm import LGBMRegressor
from optuna.trial import Trial
from sklearn.ensemble import ExtraTreesRegressor
from vivid.out_of_fold.base import BaseOutOfFoldFeature
from vivid.out_of_fold.boosting.lgbm import LGBMRegressorOutOfFold
from vivid.out_of_fold.boosting.mixins import get_boosting_parameter_suggestions
from vivid.out_of_fold.boosting.xgboost import XGBoostRegressorOutOfFold, OptunaXGBRegressionOutOfFold
from vivid.out_of_fold.ensumble import RFRegressorFeatureOutOfFold
from vivid.out_of_fold.kneighbor import KNeighborRegressorOutOfFold, OptunaKNeighborRegressorOutOfFold
from vivid.out_of_fold.linear import RidgeOutOfFold

from .base import KaggleDaysMixin, BoostingOptunaSet
from .keras import CustomKerasRegressor


# LightGBM
class LGBMReg(KaggleDaysMixin, LGBMRegressorOutOfFold):
    initial_params = {
        'learning_rate': .05,
        'reg_lambda': 1e-1,
        'n_estimators': 1000,
        'min_child_samples': 5,
        'colsample_bytree': .6,
        'subsample': .7,
        'metric': 'rmse',
        'num_leaves': 31,
    }


class LGBMRegOptuna(BoostingOptunaSet):
    model_class = LGBMRegressor
    eval_metric = 'rmse'
    initial_params = deepcopy(LGBMReg.initial_params)

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
        params = super(LGBMRegOptuna, self).get_best_model_parameters(X, y)
        params = self.parameter_postprocess(params)
        return params


# XGBOOST

class XGBReg(KaggleDaysMixin, XGBoostRegressorOutOfFold):
    pass


class XGBRegOptuna(KaggleDaysMixin, OptunaXGBRegressionOutOfFold):
    pass


# linear model
class RidgeReg(KaggleDaysMixin, RidgeOutOfFold):
    initial_params = {
        'input_scaling': 'standard',
        'target_scaling': 'standard',
        'target_logscale': True
    }

    def generate_model_class_try_params(self, trial):
        return {
            'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e2),
            # 'input_logscale': trial.suggest_categorical('input_logscale', [True, False])
        }


# rf
class RFReg(KaggleDaysMixin, RFRegressorFeatureOutOfFold):
    pass


# k-neighbor
class KNeighborReg(KaggleDaysMixin, KNeighborRegressorOutOfFold):
    pass


# optuna tuning kneighbor
class KneighberRegOptuna(KaggleDaysMixin, OptunaKNeighborRegressorOutOfFold):
    pass


class ExtraReg(KaggleDaysMixin, BaseOutOfFoldFeature):
    model_class = ExtraTreesRegressor
    initial_params = {'n_estimators': 100, 'max_features': 0.5, 'max_depth': 18, 'min_samples_leaf': 4, 'n_jobs': -1}


class KerasReg(KaggleDaysMixin, BaseOutOfFoldFeature):
    model_class = CustomKerasRegressor
    initial_params = {
        'epochs': 100,
        'batch_size': 128,
        'input_scaling': 'standard',
        'target_scaling': 'standard',
        'target_logscale': True
    }
