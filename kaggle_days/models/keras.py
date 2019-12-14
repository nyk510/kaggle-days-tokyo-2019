from keras.layers import Dropout, Dense, Input
from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.base import ClassifierMixin, RegressorMixin


class BaseKerasNNClassifiter(KerasClassifier):

    def fit(self, x, y, sample_weight=None, **kwargs):
        self.sk_params['n_input'] = x.shape[1]
        super().fit(x, y, sample_weight=sample_weight, **kwargs)
        return self

    def __call__(self, n_input: int) -> Model:
        """
        define keras model class and compile

        Args:
            n_input:

        Returns:

        """

        raise NotImplementedError()


class CustomKerasClassifier(ClassifierMixin, BaseKerasNNClassifiter):
    def build_model(self, input_layer):
        x = Dense(128, activation='relu')(input_layer)
        x = Dropout(rate=.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(rate=.2)(x)
        x = Dense(1, activation='sigmoid')(x)
        return x

    def __call__(self, n_input):
        input = Input(shape=(n_input,))
        output = self.build_model(input)
        model = Model(input, outputs=[output])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


class CustomKerasRegressor(RegressorMixin, BaseKerasNNClassifiter):
    def build_model(self, input_layer):
        layers = [256, 256, 128, 64]

        for l in layers:
            x = Dense(l, activation='relu')(input_layer)
            x = Dropout(rate=.5)(x)

        x = Dense(64, activation='relu')(x)
        x = Dropout(rate=.3)(x)
        x = Dense(1)(x)
        return x

    def __call__(self, n_input):
        input = Input(shape=(n_input,))
        output = self.build_model(input)
        model = Model(input, outputs=[output])
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
