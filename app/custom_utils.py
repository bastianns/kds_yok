import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, TransformerMixin

class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        if self.columns_to_drop:
            cols_to_drop_existing = [col for col in self.columns_to_drop if col in X_transformed.columns]
            X_transformed = X_transformed.drop(columns=cols_to_drop_existing)
        return X_transformed


class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler_type='robust'):
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("scaler_type must be 'standard', 'minmax', or 'robust'")
        self.scaler_type = scaler_type
        self.columns_ = None

    def fit(self, X, y=None):
        self.columns_ = X.columns
        self.index_ = X.index
        self.scaler.fit(X)
        return self

    def transform(self, X):
        if self.columns_ is None:
            raise RuntimeError("Transformer has not been fitted yet!")
        X_scaled_np = self.scaler.transform(X)
        X_transformed = pd.DataFrame(X_scaled_np, columns=self.columns_, index=X.index)
        return X_transformed

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = LabelEncoder()
        self.classes_ = None

    def fit(self, y):
        # ? y --> is 1D
        y_series = pd.Series(y).astype(str)
        self.encoder.fit(y_series)
        self.classes_ = self.encoder.classes_
        return self

    def transform(self, y):
        if self.classes_ is None:
            raise RuntimeError("CustomLabelEncoder has not been fitted yet!")
        y_series = pd.Series(y).astype(str)
        y_transformed = self.encoder.transform(y_series)
        return y_transformed

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y_transformed):
        if self.classes_ is None:
            raise RuntimeError("CustomLabelEncoder has not been fitted yet!")
        y_original = self.encoder.inverse_transform(y_transformed)
        return y_original

    def get_class_mapping(self):
        if self.classes_ is not None:
            return {label: index for index, label in enumerate(self.encoder.classes_)}
        return None

class ProteinRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers, bidirectional, dropout_p, padding_idx=0):
        super(ProteinRNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim,
                              hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              dropout=dropout_p if n_layers > 1 else 0,
                              batch_first=True)

        self.dropout = nn.Dropout(dropout_p)

        linear_input_features = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(linear_input_features, output_dim)

    def forward(self, text_sequences):
        embedded = self.embedding(text_sequences)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        if self.bidirectional:
            hidden_last_layer = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden_last_layer = hidden[-1,:,:]
        dropped_output = self.dropout(hidden_last_layer)
        final_output = self.fc(dropped_output)

        return final_output
