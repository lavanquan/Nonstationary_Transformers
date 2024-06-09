
class NS_TransformerConfig():
    def __init__(self) -> None:
        self.is_training=1
        self.model_id='ECL_96_192'
        self.model='ns_Transformer'

        # data loader
        self.data='ETTh2'
        self.root_path='./dataset/electricity/'
        self.data_path='electricity.csv'
        self.features='M'
        self.target='OT'
        self.freq='h'
        self.checkpoints='./checkpoints/'

        # forecasting task
        self.seq_len=96
        self.label_len=48
        self.pred_len=48

        # model define
        self.enc_in=37
        self.dec_in=37
        self.c_out=37
        self.d_model=64
        self.n_heads=2
        self.e_layers=6
        self.d_layers=4
        self.d_ff=32
        self.moving_avg=25
        self.factor=3
        self.distil=True
        self.dropout=0.1
        self.embed='timeF'
        self.activation='gelu'
        self.output_attention='store_true'
        self.do_predict='store_true'

        # optimization
        self.num_workers=1
        self.itr=2
        self.train_epochs=10
        self.batch_size=32
        self.patience=3
        self.learning_rate=0.001
        self.des='test'
        self.loss='mse'
        self.lradj='type1'
        self.use_amp=False

        # GPU
        self.use_gpu=True
        self.gpu=0
        self.use_multi_gpu=False
        self.devices='0,1,2,3'
        self.seed=2021

        # de-stationary projector params
        self.p_hidden_dims=[64, 64]
        self.p_hidden_layers=2
