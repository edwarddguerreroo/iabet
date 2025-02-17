from typing import List, Optional

class TFTConfig:
    def __init__(self, **kwargs):
        self.raw_time_features_dim: int = kwargs.get('raw_time_features_dim', 0)
        self.raw_static_features_dim: int = kwargs.get('raw_static_features_dim', 0)
        self.time_varying_categorical_features_cardinalities: List[int] = kwargs.get('time_varying_categorical_features_cardinalities', [])
        self.static_categorical_features_cardinalities: List[int] = kwargs.get('static_categorical_features_cardinalities', [])
        self.num_quantiles: int = kwargs.get('num_quantiles', 3)
        self.hidden_size: int = kwargs.get('hidden_size', 64)
        self.lstm_layers: int = kwargs.get('lstm_layers', 2)
        self.attention_heads: int = kwargs.get('attention_heads', 4)
        self.dropout_rate: float = kwargs.get('dropout_rate', 0.1)
        self.use_positional_encoding: bool = kwargs.get('use_positional_encoding', True)
        self.use_dropconnect: bool = kwargs.get('use_dropconnect', False)
        self.use_scheduled_drop_path: bool = kwargs.get('use_scheduled_drop_path', False)
        self.drop_path_rate: float = kwargs.get('drop_path_rate', 0.1)
        self.kernel_initializer: str = kwargs.get('kernel_initializer', 'glorot_uniform')
        self.use_glu_in_grn: bool = kwargs.get('use_glu_in_grn', True)
        self.use_layer_norm_in_grn: bool = kwargs.get('use_layer_norm_in_grn', True)
        self.use_multi_query_attention: bool = kwargs.get('use_multi_query_attention', False)
        self.use_indrnn: bool = kwargs.get('use_indrnn', False)
        self.use_logsparse_attention: bool = kwargs.get('use_logsparse_attention', False)
        self.sparsity_factor: int = kwargs.get('sparsity_factor', 4)
        self.use_evidential_regression: bool = kwargs.get('use_evidential_regression', False)
        self.use_mdn: bool = kwargs.get('use_mdn', False)
        self.num_mixtures: int = kwargs.get('num_mixtures', 5)
        self.use_time2vec: bool = kwargs.get('use_time2vec', False)
        self.time2vec_dim: int = kwargs.get('time2vec_dim', 32)
        self.time2vec_activation: str = kwargs.get('time2vec_activation', 'sin')  # O 'sin'
        self.use_fourier_features: bool = kwargs.get('use_fourier_features', False)
        self.num_fourier_features: int = kwargs.get('num_fourier_features', 10)
        self.use_reformer_attention: bool = kwargs.get('use_reformer_attention', False)
        self.num_buckets: int = kwargs.get('num_buckets', 8)
        self.use_sparsemax: bool = kwargs.get('use_sparsemax', False)
        self.l1_reg: float = kwargs.get('l1_reg', 0.0)
        self.l2_reg: float = kwargs.get('l2_reg', 0.0)
        self.use_gnn: bool = kwargs.get('use_gnn', False)
        self.gnn_embedding_dim: Optional[int] = kwargs.get('gnn_embedding_dim', None)
        self.use_transformer: bool = kwargs.get('use_transformer', False)
        self.transformer_embedding_dim: Optional[int] = kwargs.get('transformer_embedding_dim', None)
        self.seq_len: int = kwargs.get('seq_len', 20)  # Añadido seq_len


    def copy(self, update=None):
        if update is None:
            update = {}
        return TFTConfig(**{**self.__dict__, **update})

    def dict(self):
        return self.__dict__