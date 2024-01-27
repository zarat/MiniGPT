# MiniGPT

A toy GPT. Just put a bunch of text files in a folder.

Train a model (just 1 cycle for demonstration).

```
C:\Users\manuel> python train.py --data-dir .\test --word-tokens --num-epochs 1
Loading data...
corpus length: 33517
vocab size: 3775
x.shape: (1312, 50)
y.shape: (1312, 50, 1)
Data load time 0.18010282516479492
Building model...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 embedding (Embedding)       (32, None, 64)            241600

 lstm (LSTM)                 (32, None, 128)           98816

 dropout (Dropout)           (32, None, 128)           0

 time_distributed (TimeDist  (32, None, 3775)          486975
 ributed)

=================================================================
Total params: 827391 (3.16 MB)
Trainable params: 827391 (3.16 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
Model build time 5.000430107116699
Training...
41/41 [==============================] - 12s 175ms/step - loss: 8.2343 - accuracy: 0.0264
Training time 14.842327356338501
Model saved to test\model.pkl test\model.h5
```

Generate text.

```
C:\Users\manuel>python sample.py --data-dir .\test --length 50
Using seed: sie wieder an dem StÃ¼ckchen in der rechten
--------------------------------------------------
sie wieder an dem StÃ¼ckchen in der rechten vielen nachgerade traben fuãÿboden famoser siehst franzã¶sischen schatten einen sollen neue schatten vorhielt fertig sperrten schmeckt gã¤nzlichem gerade woran erklã¤re derselben hã¤nde einfache zwã¶lf variationen schriftstã¼ck zuwarf herauslassen klã¤rten guckte krachen ablã¤ugnen nachfã¼hlen heftigsten feinste miene einer concert kã¼rzlich tiefes quieken zitternder hummerballet fehle gestemmt jene wirklichen schwierig schlich entrã¼stet
´´´
