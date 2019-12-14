from vivid.featureset.molecules import MoleculeFeature


class CacheFeatureMixin:
    """予測値に関しても cache file を使う mixin"""
    predict_df_ = None

    def predict(self, input_df):
        if self.predict_df_ is not None:
            print('load from cache {}'.format(str(self)))
            return self.predict_df_

        pred_df = super(CacheFeatureMixin, self).predict(input_df)
        self.predict_df_ = pred_df
        return pred_df


class KaggleDaysFeature(CacheFeatureMixin, MoleculeFeature):
    """Kaggle Days 用の基底クラス"""

    def call(self, df_source, y=None, test=False):
        out_df = super(KaggleDaysFeature, self).call(df_source, y, test)

        self.logger.info('n_features: {}'.format(len(out_df.columns)))
        for c in out_df.columns:
            self.logger.info(c)

        if 'index' in out_df.columns:
            raise ValueError(f'Invalid Column Name: INDEX')
        return out_df
