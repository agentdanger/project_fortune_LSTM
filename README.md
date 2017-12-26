# Project Fortune - an LSTM generative neural network
Copyright 2017 Courtney Perigo, please attribute if you use this.

Please see my blog for more information on this code.

https://www.courtneyperigo.com/single-post/2017/12/25/A-Holiday-Gift-Sprout-the-Motivational-Bot

Please install the following prerequisites in order to use this code...
- Python 3.5.4
- Numpy 1.13.3
- Keras 2.0.8 (Tensorflow Backend)
- Tensorflow 1.3.0 

Finally, you will need a text data set.  Any text will do, but should be saved as a .txt file.

Notes about settings:

SEQ_LENGTH = 100 #This is used to determine how long you want your training sequences to be.  100 is okay, but depending on your data, this could be bigger or smaller.
length = 250 #This is used to set the amount of generative text you'd like the software to generate after each EPOCH is complete.
HIDDEN_DIM = 1000 # This can be tweaked but my model worked with 1000 hidden dimensions.  You may tweak this to make your model smarter.
BATCH_SIZE = 50 # This is set to 50 because it was the fastest that my machine could handle.  You may need to decrease this number if your data is huge.
LAYER_NUM = 3 # My POC used 3 layers.  Tweak as needed.
EPOCHS = 30 # I didn't get useful results until at least 30 EPOCHS were completed.

Copyright 2017 Courtney Perigo
