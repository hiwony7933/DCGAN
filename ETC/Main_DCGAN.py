import numpy as np 
from keras.models import * 
from keras.layers import * 
from keras.optimizers import * 
from keras.datasets import mnist 
import keras.backend as K 
import matplotlib.pyplot as plt 
from pandas import Series, DataFrame 
from DCGAN import *
import time

K.set_image_data_format('channels_last') 

#x_train, x_test, y_train, y_test = np.load('binary_image_data.npy')
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0 
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)

# Init network 
gan = Gan(x_train) 

# Some parameters. 
epochs = 30 
sample_size = 10 
batch_size = 100
train_per_epoch = x_train.shape[0] // batch_size
history=[]

for epoch in range(0, epochs): 
    start_time=time.time()
    total_d_loss = 0.0 
    total_a_loss = 0.0 
    
    for batch in range(0, train_per_epoch): 
        d_loss, a_loss, imgs = gan.train(batch_size) 
        total_d_loss += d_loss 
        total_a_loss += a_loss 
        
    total_d_loss /= train_per_epoch 
    total_a_loss /= train_per_epoch
    
    recode =(epoch, total_d_loss, total_a_loss)
    history.append(recode)

    print("Epoch : {}, elapsed : {}, D Loss: {}, AM Loss: {}".format(epoch + 1,time.time() - start_time, total_d_loss, total_a_loss)) 
    
    fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1)) 
    for i in range(0, sample_size): 
        ax[i].set_axis_off() 
        ax[i].imshow(imgs[i].reshape((gan.img_rows, gan.img_cols)), interpolation='nearest')
    plt.show() 
    plt.close(fig)

    gan.save()


df = DataFrame(history, columns=['epoch', 'total_d_loss', 'total_a_loss'])
df.plot(y=['total_d_loss', 'total_a_loss'])
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()