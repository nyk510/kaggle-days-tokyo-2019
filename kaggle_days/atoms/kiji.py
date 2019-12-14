import os

import joblib
import numpy as np
import pandas as pd
from anemone.embedding import SWEM
from anemone.preprocess import normalize_neologd
from anemone.preprocess.wakati import DocumentParser
from gensim.models import FastText
from sklearn.decomposition import PCA
from vivid.featureset import AbstractMergeAtom
from vivid.utils import timer, get_logger

from kaggle_days.dataset import read_data
from kaggle_days.dataset import read_kiji
from kaggle_days.env import CACHE_DIR

logger = get_logger('kaggle-days.kiji', log_level='INFO')


def safe_normalize(d):
    try:
        return normalize_neologd(d)
    except:
        return None


class NikkeiFastText:
    path_to_model = os.path.join(CACHE_DIR, 'fasttext.model')
    logger = get_logger('kaggle-days.fasttext', log_level='INFO')
    cache_model = None

    @classmethod
    def load_model(cls):
        if cls.cache_model is not None:
            cls.logger.info('return from cache')
            return cls.cache_model

        if os.path.exists(cls.path_to_model):
            cls.logger.info('model already created. load from disk.')
            model = FastText.load(cls.path_to_model)
        else:
            with timer(cls.logger, format_str='create fasttext: ' + '{:.3f}'):
                model = cls.create_fast_model()
            model.save(cls.path_to_model)

        cls.cache_model = model
        return model

    @classmethod
    def create_fast_model(cls):
        kiji_df = read_kiji()

        main_text = kiji_df['title'].fillna('') + kiji_df['title2'].fillna('') + kiji_df['title3'].fillna('') + kiji_df[
            'body'].fillna('')
        main_text = main_text.values
        main_text = [normalize_neologd(d) for d in main_text]
        parser = DocumentParser()
        parsed_docs = [parser.call(d) for d in main_text]
        model = FastText(parsed_docs, size=128, workers=6, iter=10)
        return model


class KijiMixin:
    merge_key = 'kiji_id'

    def read_outer_dataframe(self):
        return read_kiji()


class KijiCountAtom(KijiMixin, AbstractMergeAtom):
    def generate_outer_feature(self):
        df = self.df_outer
        columns = [
            'service_category', 'title', 'title2',
            'title3', 'genres', 'belong_topic_info', 'keywords', 'body'
        ]
        out_df = df[['kiji_id_raw']].rename(columns=dict(kiji_id_raw=self.merge_key))
        out_df['moji_count'] = df['moji_count'].values

        for c in columns:
            out_df[f'{c}_count'] = df[c].str.len()
        return out_df


def keyword_text_to_list(s):
    try:
        words = s.replace('"', '').replace('[', '').replace(']', '').replace("'", '').replace(' ', '').split(
            ',')
        words = [normalize_neologd(w) for w in words]
        return words
    except Exception as e:
        logger.warn(f'{str(e)}@{str(s)}')
        return []


class KijiKeyowrdAtom(KijiMixin, AbstractMergeAtom):

    def generate_outer_feature(self):
        data = self.df_outer['keywords']

        keywords = data.map(keyword_text_to_list)
        words = [w for d in keywords.values for w in d]
        vc = pd.Series(words).value_counts()

        def to_count(doc):
            return [vc[d] for d in doc if len(d) > 0]

        keyword_counts = [to_count(doc) for doc in keywords]
        keyword_counts = np.array(keyword_counts)

        def parse_to(row, agg=np.mean):
            if len(row) == 0:
                return 0
            return agg(row)

        data = {
            'mean': [parse_to(d) for d in keyword_counts],
            'sum': [parse_to(d, np.sum) for d in keyword_counts],
            'count': [len(d) for d in keyword_counts]
        }

        key_df = pd.DataFrame(data)
        key_df[self.merge_key] = self.df_outer[['kiji_id_raw']].rename(columns=dict(kiji_id_raw=self.merge_key))
        return key_df


class KijiGenreAtom(KijiMixin, AbstractMergeAtom):

    def generate_outer_feature(self):
        import json

        def load_genres(w):
            try:
                return json.loads(w.replace("'", '"'))
            except Exception:
                return []

        datas = self.df_outer['genres'].map(load_genres)
        datas = [[d.get('name') for d in row] for row in datas]
        vc = pd.Series([w for d in datas for w in d]).value_counts()

        x = [[c in row for row in datas] for c in vc.index]
        x = np.array(x)
        out_df = pd.DataFrame(x.T, columns=vc.index)
        out_df[self.merge_key] = self.df_outer['kiji_id_raw']
        return out_df


class TextSWEMAtom(KijiMixin, AbstractMergeAtom):

    def __init__(self, n_components=32, agg='max', text_column='body'):
        self.n_components = n_components
        self.agg = agg
        self.text_column = text_column
        self.cache_path = os.path.join(CACHE_DIR, f'kiji_text_{text_column}.joblib')

        super(TextSWEMAtom, self).__init__()

    def __str__(self):
        s = super(TextSWEMAtom, self).__str__()
        s = s + f' {self.agg}@{self.text_column}_n_components={self.n_components}'
        return s

    def call(self, df_input, y=None):
        self.train = y is not None
        return super(TextSWEMAtom, self).call(df_input, y)

    def load_parsed_docs(self):

        if os.path.exists(self.cache_path):
            return joblib.load(self.cache_path)

        df = self.df_outer
        text_data = df[self.text_column]

        if self.text_column == 'keywords':
            text_data = text_data.map(keyword_text_to_list)
            text_data = [' '.join(d) for d in text_data]

        with timer(logger=logger, format_str=self.text_column + ' parse context {:.3f}[s]'):
            title_docs = [safe_normalize(d) for d in text_data]
            title_docs = np.array(title_docs)
            idx_none = title_docs == None
            title_docs = title_docs[~idx_none]
            parser = DocumentParser()
            parsed = [parser.call(s) for s in title_docs]

            swem = SWEM(NikkeiFastText.load_model(), aggregation=self.agg)
            x = swem.transform(parsed)

        joblib.dump([x, idx_none], self.cache_path)
        return x, idx_none

    def generate_outer_feature(self):
        with timer(logger, format_str=self.text_column + ' load {:.3f}[s]'):
            x, idx_none = self.load_parsed_docs()

        if self.train:
            clf_pca = PCA(n_components=self.n_components)
            clf_pca.fit(x)
            self.clf_pca_ = clf_pca

        transformed = self.clf_pca_.transform(x)
        retval = np.zeros(shape=(len(self.df_outer), self.n_components))
        retval[~idx_none] = transformed
        out_df = pd.DataFrame(retval, columns=[f'swem_{self.agg}_{self.text_column}_' + str(i) for i in
                                               range(self.n_components)])
        out_df[self.merge_key] = self.df_outer[['kiji_id_raw']].rename(columns=dict(kiji_id_raw=self.merge_key))
        return out_df


class KijiDocVectorAtom(KijiMixin, AbstractMergeAtom):

    def call(self, df_input, y=None):
        self.train = y is not None
        return super(KijiDocVectorAtom, self).call(df_input, y)

    @property
    def fast_model_path(self):
        return os.path.join(CACHE_DIR, 'kiji_vec.model')

    def generate_outer_feature(self):
        train_df, _ = read_data()
        test_df = read_data(test=True)
        all_df = pd.concat([train_df, test_df], ignore_index=True)

        users = all_df['user_id'].unique()

        docs = []
        for u in users:
            docs.append(all_df[all_df['user_id'] == u]['kiji_id'].values)

        vc = all_df['kiji_id'].value_counts()
        to_none_ids = vc[vc < 5].index

        def to_word(d):
            if d in to_none_ids:
                return 'None'
            return d

        if os.path.exists(self.fast_model_path):
            model = FastText.load(self.fast_model_path)
        else:
            docs = [[to_word(w) for w in doc] for doc in docs]
            with timer(logger, format_str='create kiji_id fast_model' + ' {:.3f}[s]'):
                model = FastText(docs, workers=6, size=64)
            model.save(self.fast_model_path)

        z = self.df_outer['kiji_id_raw'].map(to_word).map(lambda x: model.wv[x])
        df = pd.DataFrame(np.array(z.values.tolist())).add_prefix('kiji_wv_')
        df[self.merge_key] = self.df_outer['kiji_id_raw']
        return df


class KijiPublishTimeAtom(KijiMixin, AbstractMergeAtom):
    def generate_outer_feature(self):
        x = pd.to_datetime(self.df_outer['display_time'])
        out_df = pd.DataFrame()
        t = x.dt.hour * 60 + x.dt.minute

        theta = t / 60 * 24 * 2 * np.pi
        out_df['cos'] = np.cos(theta)
        out_df['sin'] = np.sin(theta)
        out_df['minute'] = t
        out_df[self.merge_key] = self.df_outer['kiji_id_raw']
        return out_df
