from sklearn.model_selection import GroupKFold
from vivid.out_of_fold.base import BaseOptunaOutOfFoldFeature
from vivid.out_of_fold.boosting.mixins import FeatureImportanceMixin, BoostingEarlyStoppingMixin
from vivid.sklearn_extend import PrePostProcessModel

from kaggle_days.feature_base import CacheFeatureMixin


class KaggleDaysMixin(CacheFeatureMixin):
    num_cv = 10
    user_id_group = None

    def load_best_models(self):
        models = []

        for cv in range(self.num_cv):
            model = PrePostProcessModel(model_class=self.model_class,
                                        output_dir=self.output_dir,
                                        prepend_name=str(cv))
            model.load_trained_model()
            models.append(model)
        return models

    def get_fold_splitting(self, X, y, groups=None):
        return GroupKFold(n_splits=self.num_cv).split(X, y, self.user_id_group)
        # return StratifiedKFold(n_splits=self.num_cv, shuffle=True, random_state=41).split(X, y, groups)

    # def save_best_models(self, best_models):
    #     pass
    def call(self, df_source, y=None, test=False):
        if not test:
            self.user_id_group = df_source['user_id'].values

        if 'user_id' in df_source.columns:
            del df_source['user_id']
        return super(KaggleDaysMixin, self).call(df_source, y, test)

    def after_kfold_fitting(self, df_source, y, predict):
        try:
            return super(KaggleDaysMixin, self).after_kfold_fitting(df_source, y, predict)
        except Exception as e:
            self.logger.warn(str(e))

        self.show_metrics(y, predict)


class BoostingOptunaSet(FeatureImportanceMixin, BoostingEarlyStoppingMixin, BaseOptunaOutOfFoldFeature):
    pass
