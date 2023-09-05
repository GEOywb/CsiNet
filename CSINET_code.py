import h5py
import keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils
from keras.layers import Activation,Input,Dense,Lambda,LSTM,Concatenate,BatchNormalization
iterations=100000
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
tf.config.experimental_run_functions_eagerly(True)
data=h5py.File('houston2013.mat')
X_spatial_all=data['X_spatial'][()].transpose(3,2,1,0)
LiDAR_all=data['LiDAR'][()].transpose(2,1,0)
LiDAR_all=np.expand_dims(LiDAR_all,-1)
act_Y_train_all=data['act_Y_train'][()].transpose(1,0)
indexi_all=data['indexi'][()].transpose(1,0)
indexj_all=data['indexj'][()].transpose(1,0)
X_spatial_all=X_spatial_all.astype('float32')
LiDAR_all=LiDAR_all.astype('float32')
act_Y_train_all=act_Y_train_all.astype('int')
indexi_all=indexi_all.astype('float32')
indexj_all=indexj_all.astype('float32')
act_Y_train_all[act_Y_train_all==-1]=0
slide_size=5
if slide_size==3:
    X_spatial_all=X_spatial_all[:,4:7,4:7,:]
    LiDAR_all=LiDAR_all[:,4:7,4:7,:]
if slide_size==5:
    X_spatial_all=X_spatial_all[:,3:8,3:8,:]
    LiDAR_all=LiDAR_all[:,3:8,3:8,:]
if slide_size==7:
    X_spatial_all=X_spatial_all[:,2:9,2:9,:]
    LiDAR_all=LiDAR_all[:,2:9,2:9,:]
if slide_size==9:
    X_spatial_all=X_spatial_all[:,1:10,1:10,:]
    LiDAR_all=LiDAR_all[:,1:10,1:10,:]
if slide_size==11:
    X_spatial_all=X_spatial_all[:,:,:,:]
    LiDAR_all=LiDAR_all[:,:,:,:]
scaler=MinMaxScaler(feature_range=(-1,1))
X_spatial_all_=X_spatial_all.reshape([X_spatial_all.shape[0],X_spatial_all.shape[1]*X_spatial_all.shape[2]*X_spatial_all.shape[3]])
LiDAR_all_=LiDAR_all.reshape([LiDAR_all.shape[0],LiDAR_all.shape[1]*LiDAR_all.shape[2]*LiDAR_all.shape[3]])
X_spatial_all_=scaler.fit_transform(X_spatial_all_)
LiDAR_all_=scaler.fit_transform(LiDAR_all_)
X_spatial_all=X_spatial_all_.reshape([X_spatial_all.shape[0],X_spatial_all.shape[1],X_spatial_all.shape[2],X_spatial_all.shape[3]])
LiDAR_all=LiDAR_all_.reshape([LiDAR_all.shape[0],LiDAR_all.shape[1],LiDAR_all.shape[2],LiDAR_all.shape[3]])
X_spatial_all_=X_spatial_all.reshape([X_spatial_all.shape[0],X_spatial_all.shape[1]*X_spatial_all.shape[2],X_spatial_all.shape[3]])
LiDAR_all_=LiDAR_all.reshape([LiDAR_all.shape[0],LiDAR_all.shape[1]*LiDAR_all.shape[2],LiDAR_all.shape[3]])
mid=int((slide_size-1)/2)
for num in range(X_spatial_all.shape[0]):
    print(num)
    weight_s=np.zeros([slide_size,slide_size])
    weight_x=np.zeros([slide_size,slide_size])
    weight_y=np.zeros([slide_size,slide_size])
    weight_h=np.zeros([slide_size,slide_size])
    spectral=X_spatial_all[num,:,:,:]
    elevation=LiDAR_all[num,:,:,0]
    for ii in range(slide_size):
        for jj in range(slide_size):
            weight_s[ii,jj]=1-np.dot(spectral[ii,jj,:],spectral[mid,mid,:])/(np.linalg.norm(spectral[ii,jj,:])*np.linalg.norm(spectral[mid,mid,:]))
            weight_x[ii,jj]=np.abs(ii-mid)
            weight_y[ii,jj]=np.abs(jj-mid)
            weight_h[ii,jj]=np.abs(elevation[ii,jj]-elevation[mid,mid])
    weight_x=(weight_x-weight_x.min())/(weight_x.max()-weight_x.min())
    weight_y=(weight_y-weight_y.min())/(weight_y.max()-weight_y.min())
    weight_h=(weight_h-weight_h.min())/(weight_h.max()-weight_h.min())
    weight_s=(weight_s-weight_s.min())/(weight_s.max()-weight_s.min())
    weight=np.zeros([slide_size,slide_size])
    for i in range(slide_size):
        for j in range(slide_size):
            weight[i,j]=np.sqrt(np.square(weight_x[ii,jj])+np.square(weight_y[ii,jj])+np.square(weight_h[ii,jj]))*weight_s[i,j]
    weight=weight.reshape([slide_size*slide_size])
    randpaixv=weight.argsort()
    X_spatial_all_[num,:,:]=X_spatial_all_[num,randpaixv,:]
    LiDAR_all_[num,:,:]=LiDAR_all_[num,randpaixv,:]
X_spatial_all=X_spatial_all_
LiDAR_all=LiDAR_all_
act_Y_train_all=np.reshape(act_Y_train_all,act_Y_train_all.shape[0])
indexi_all=np.reshape(indexi_all,indexi_all.shape[0])
indexj_all=np.reshape(indexj_all,indexj_all.shape[0])
act_Y_train=act_Y_train_all
randpaixv=act_Y_train_all.argsort()
X_spatial_all=X_spatial_all[randpaixv]
LiDAR_all=LiDAR_all[randpaixv]
indexi_all=indexi_all[randpaixv]
indexj_all=indexj_all[randpaixv]
act_Y_train_all=act_Y_train_all[randpaixv]
X_spatial_all=X_spatial_all[act_Y_train_all>0]
LiDAR_all=LiDAR_all[act_Y_train_all>0]
indexi_all=indexi_all[act_Y_train_all>0]
indexj_all=indexj_all[act_Y_train_all>0]
act_Y_train_all=act_Y_train_all[act_Y_train_all>0]
indices=np.arange(X_spatial_all.shape[0])
indices_train,indices_test,act_Y_train_train,act_Y_train_test=train_test_split(indices,act_Y_train_all,test_size=0.99,stratify=act_Y_train_all)
X_spatial_train=X_spatial_all[indices_train,:,:]
LiDAR_train=LiDAR_all[indices_train,:,:]
act_Y_train_train=act_Y_train_all[indices_train]
indexi_train=indexi_all[indices_train]
indexj_train=indexj_all[indices_train]
X_spatial_test=X_spatial_all[indices_test,:,:]
LiDAR_test=LiDAR_all[indices_test,:,:]
act_Y_train_test=act_Y_train_all[indices_test]
indexi_test=indexi_all[indices_test]
indexj_test=indexj_all[indices_test]
act_Y_train_train=np_utils.to_categorical(act_Y_train_train-1)
act_Y_train_test=np_utils.to_categorical(act_Y_train_test-1)
kernelsize=3
activation='tanh'
kernel_regularizer=tf.keras.regularizers.l2(0.01)
lr=0.001
H_input=Input(shape=(X_spatial_train.shape[1],X_spatial_train.shape[2]))
L_input=Input(shape=(LiDAR_train.shape[1],1))
H_single=H_input
L_single=L_input
H=LSTM(16,return_sequences=True,kernel_regularizer=kernel_regularizer)(H_single)
H=BatchNormalization()(H)
H=Activation(activation)(H)
H=LSTM(32,return_sequences=True,kernel_regularizer=kernel_regularizer)(H)
H=BatchNormalization()(H)
H=Activation(activation)(H)
H=LSTM(64,return_sequences=True,kernel_regularizer=kernel_regularizer)(H)
H=BatchNormalization()(H)
H=Activation(activation)(H)
H=LSTM(128,return_sequences=True,kernel_regularizer=kernel_regularizer)(H)
H=BatchNormalization()(H)
H=Activation(activation)(H)
L=LSTM(16,return_sequences=True,kernel_regularizer=kernel_regularizer)(L_single)
L=BatchNormalization()(L)
L=Activation(activation)(L)
L=LSTM(32,return_sequences=True,kernel_regularizer=kernel_regularizer)(L)
L=BatchNormalization()(L)
L=Activation(activation)(L)
L=LSTM(64,return_sequences=True,kernel_regularizer=kernel_regularizer)(L)
L=BatchNormalization()(L)
L=Activation(activation)(L)
L=LSTM(128,return_sequences=True,kernel_regularizer=kernel_regularizer)(L)
L=BatchNormalization()(L)
L=Activation(activation)(L)
HL=Concatenate()([H,L])
HL1=LSTM(128,return_sequences=True,kernel_regularizer=kernel_regularizer)(HL)
HL1=BatchNormalization()(HL1)
HL1=Activation(activation)(HL1)
HL2=LSTM(64,return_sequences=True,kernel_regularizer=kernel_regularizer)(HL1)
HL2=BatchNormalization()(HL2)
HL2=Activation(activation)(HL2)
HL2=Lambda(lambda x:K.mean(x,axis=1))(HL2)
HL3=Dense(act_Y_train_train.shape[1],activation='softmax')(HL2)
H=LSTM(64,return_sequences=True,kernel_regularizer=kernel_regularizer)(HL1)
H=Activation('sigmoid')(H)
H=LSTM(32,return_sequences=True,kernel_regularizer=kernel_regularizer)(H)
H=Activation('sigmoid')(H)
H=LSTM(16,return_sequences=True,kernel_regularizer=kernel_regularizer)(H)
H=Activation('sigmoid')(H)
H=LSTM(X_spatial_train.shape[2],return_sequences=True,kernel_regularizer=kernel_regularizer)(H)
H_out=Activation('sigmoid')(H)
L=LSTM(64,return_sequences=True,kernel_regularizer=kernel_regularizer)(HL1)
L=Activation('sigmoid')(L)
L=LSTM(32,return_sequences=True,kernel_regularizer=kernel_regularizer)(L)
L=Activation('sigmoid')(L)
L=LSTM(16,return_sequences=True,kernel_regularizer=kernel_regularizer)(L)
L=Activation('sigmoid')(L)
L=LSTM(1,return_sequences=True,kernel_regularizer=kernel_regularizer)(L)
L_out=Activation('sigmoid')(L)
loss1=Lambda(lambda x:K.sqrt(K.sum(K.sum(K.square(x[0]-x[1]),axis=-1),axis=-1)),output_shape=[1,])([H_single,H_out])
loss2=Lambda(lambda x:K.sqrt(K.sum(K.sum(K.square(x[0]-x[1]),axis=-1),axis=-1)),output_shape=[1,])([L_single,L_out])
loss=Lambda(lambda x:x[0]+x[1],output_shape=[1,])([loss1,loss2])
network=keras.models.Model([H_input,L_input],[HL3,loss])
network.compile(loss=['categorical_crossentropy','mean_squared_error'],loss_weights=[1,1],optimizer=keras.optimizers.Adam(lr=lr,decay=0.01),metrics=['accuracy'])
network.summary()     
history=network.fit([X_spatial_train,LiDAR_train],[act_Y_train_train,np.zeros([act_Y_train_train.shape[0],1])],batch_size=256,epochs=iterations,shuffle=True,verbose=1)
predicted,q=network.predict([X_spatial_test,LiDAR_test])
