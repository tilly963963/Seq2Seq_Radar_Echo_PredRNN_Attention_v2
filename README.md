# Seq2Seq_Radar_Echo_PredRNN_Attention_v2

## Radar echo picture sequence prediction 

![radar](https://ppt.cc/frzzMx@.png)

![radar](https://ppt.cc/fhHykx@.png)

![radar](https://ppt.cc/fIvOcx@.png)

![radar](https://ppt.cc/fDsuYx@.png)

## Get Started Train the PredRNN_Attention(IDA-LSTM)

1. Install env_v2(env for 120)

2. Set parameters 

check InterDST_train.py is in the 'save_path' folder

check load_radar_echo_df_path is in the 'save_path' folder

check model name
```
parser.add_argument('--model_name', type=str, default='InterDST_LSTMCell_checkpoint')
```

You can set pretrained model and check pretrained_model is in the 'save_path' folder

You can set parameters, ex: weighted loss function

```
core/models/model_factory_LayerNormpy.py /

self.weight = [1,2,5,10,30,40]

self.custom_criterion = MyMSELoss(self.weight)
```

3. Train the model


```
python -m InterDST_train.py
```

## Test the PredRNN_Attention(IDA-LSTM) model

check InterDST_test.py is in the 'save_path' folder

check model name
```
parser.add_argument('--model_name', type=str, default='InterDST_LSTMCell_checkpoint')
```
check model name, ex:

```
model_name = 'model_itr20_test_cost18.401666419122662.pkl'
```

2. Test and evaluation

```
python -m InterDST_test.py
```
