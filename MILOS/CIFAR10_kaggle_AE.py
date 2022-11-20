from keras.datasets import cifar10
from keras.layers import Input, Dense,Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization
from keras.models import Model,Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

(X_train, _), (X_test, _) = cifar10.load_data()
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255
X_train = X_train.reshape(len(X_train),X_train.shape[1],X_train.shape[2],3)
X_test = X_test.reshape(len(X_test), X_test.shape[1],X_test.shape[2],3)
print(X_train.shape)    # (50 000, 32, 32, 3)
print(X_test.shape)     # (10 000, 32, 32, 3)


### Good for Minist Dataset ###

input_img = Input(shape=(32,32,3))

#Encoder
x = Conv2D(16,(3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)

x = Conv2D(8,(3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)

x = Conv2D(8,(3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same', name='encoder')(x)

#Decoder
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(16, (3, 3), activation='relu',padding='same')(x)
x = UpSampling2D((2, 2))(x)

decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')


### Good for coloured Image ###

model = Sequential()

model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(BatchNormalization())     # 32x32x32
model.add(Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'))      # 16x16x32
model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))      # 16x16x32
model.add(BatchNormalization())     # 16x16x32
model.add(UpSampling2D())
model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))      # 32x32x32
model.add(BatchNormalization())
model.add(Conv2D(3,  kernel_size=1, strides=1, padding='same', activation='sigmoid'))   # 32x32x3

model.compile(optimizer='adam', metrics=['accuracy'], loss='mean_squared_error')
#print(model.summary())

# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d_7 (Conv2D)           (None, 32, 32, 32)        896       
                                                                 
#  batch_normalization (BatchN  (None, 32, 32, 32)       128       
#  ormalization)                                                   
                                                                 
#  conv2d_8 (Conv2D)           (None, 16, 16, 32)        9248      
                                                                 
#  conv2d_9 (Conv2D)           (None, 16, 16, 32)        9248      
                                                                 
#  batch_normalization_1 (Batc  (None, 16, 16, 32)       128       
#  hNormalization)                                                 
                                                                 
#  up_sampling2d_3 (UpSampling  (None, 32, 32, 32)       0         
#  2D)                                                             
                                                                 
#  conv2d_10 (Conv2D)          (None, 32, 32, 32)        9248      
                                                                 
#  batch_normalization_2 (Batc  (None, 32, 32, 32)       128       
#  hNormalization)                                                 
                                                                 
#  conv2d_11 (Conv2D)          (None, 32, 32, 3)         99        
                                                                 
# =================================================================
# Total params: 29,123
# Trainable params: 28,931
# Non-trainable params: 192
# _________________________________________________________________
# None


autoencoder=Model(input_img, decoded)

#print(autoencoder.summary())

# Model: "model_1"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input_1 (InputLayer)        [(None, 32, 32, 3)]       0         
                                                                 
#  conv2d (Conv2D)             (None, 32, 32, 16)        448       
                                                                 
#  max_pooling2d (MaxPooling2D  (None, 16, 16, 16)       0         
#  )                                                               
                                                                 
#  conv2d_1 (Conv2D)           (None, 16, 16, 8)         1160      
                                                                 
#  max_pooling2d_1 (MaxPooling  (None, 8, 8, 8)          0         
#  2D)                                                             
                                                                 
#  conv2d_2 (Conv2D)           (None, 8, 8, 8)           584       
                                                                 
#  encoder (MaxPooling2D)      (None, 4, 4, 8)           0         
                                                                 
#  conv2d_3 (Conv2D)           (None, 4, 4, 8)           584       
                                                                 
#  up_sampling2d (UpSampling2D  (None, 8, 8, 8)          0         
#  )                                                               
                                                                 
#  conv2d_4 (Conv2D)           (None, 8, 8, 8)           584       
                                                                 
#  up_sampling2d_1 (UpSampling  (None, 16, 16, 8)        0         
#  2D)                                                             
                                                                 
#  conv2d_5 (Conv2D)           (None, 16, 16, 16)        1168      
                                                                 
#  up_sampling2d_2 (UpSampling  (None, 32, 32, 16)       0         
#  2D)                                                             
                                                                 
#  conv2d_6 (Conv2D)           (None, 32, 32, 3)         435       
                                                                 
# =================================================================
# Total params: 4,963
# Trainable params: 4,963
# Non-trainable params: 0
# _________________________________________________________________
# None

encoder = Model(input_img, encoded)

#print(encoder.summary())

# Model: "model_2"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input_1 (InputLayer)        [(None, 32, 32, 3)]       0         
                                                                 
#  conv2d (Conv2D)             (None, 32, 32, 16)        448       
                                                                 
#  max_pooling2d (MaxPooling2D  (None, 16, 16, 16)       0         
#  )                                                               
                                                                 
#  conv2d_1 (Conv2D)           (None, 16, 16, 8)         1160      
                                                                 
#  max_pooling2d_1 (MaxPooling  (None, 8, 8, 8)          0         
#  2D)                                                             
                                                                 
#  conv2d_2 (Conv2D)           (None, 8, 8, 8)           584       
                                                                 
#  encoder (MaxPooling2D)      (None, 4, 4, 8)           0         
                                                                 
# =================================================================
# Total params: 2,192
# Trainable params: 2,192
# Non-trainable params: 0
# _________________________________________________________________
# None

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, X_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))

#encoded_imgs = model.predict(X_test)
predicted = model.predict(X_test) # predicted.shape = (10000, 32, 32, 3)

plt.figure(figsize=(40,4))
for i in range(10):
    # display original images
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(X_test[i].reshape(32, 32,3))
    
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    

    
    # display reconstructed images
    ax = plt.subplot(3, 20, 2*20 +i+ 1)
    plt.imshow(predicted[i].reshape(32, 32,3))
    #plt.savefig(f'./reconstructed_images_{i}.png')
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  
    
plt.savefig('/home/novakovm/iris/MILOS/reconstructed_images.png')
#plt.show()


avg_l2_recon_test_errors = []
for idx_test_img_sample, test_img_sample in enumerate(predicted):
    x = X_test[idx_test_img_sample]
    x_hat = test_img_sample
    # avg_l2_recon_error= (np.mean((x[:, : , 0] - x_hat[:, : , 0])**2)+\
    #                     np.mean((x[:, : , 1] - x_hat[:, : , 1])**2)+\
    #                     np.mean((x[:, : , 2] - x_hat[:, : , 1])**2))/3
    avg_l2_recon_error = ((x - x_hat)**2).mean()
    avg_l2_recon_test_errors.append(avg_l2_recon_error)
avg_l2_recon_test_errors = np.array(avg_l2_recon_test_errors)
print(avg_l2_recon_test_errors.min(),avg_l2_recon_test_errors.max(), avg_l2_recon_test_errors.mean())
print(predicted.min(), predicted.max(), predicted.mean())
print(X_test.min(), X_test.max(), X_test.mean())