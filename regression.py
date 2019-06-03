import random
import math
import numpy as np
from keras.layers import Dense,Input,Lambda,Activation
import keras
from keras.models import Model
from keras import backend as K
import sys
import os
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, History
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

#!{sys.executable} -m pip install sklearn
#!{sys.executable} -m pip install keras

def generate_input():
    return [random.uniform(0, 1),random.uniform(0, 1),random.uniform(0, 1),random.uniform(0, 1),random.uniform(0, 1)]

def generate_output(inputs):
    x= (1/13)*(10*math.sin(math.pi*inputs[0]*inputs[1])+20*(inputs[2]-.5)**2+10*inputs[3]+5*inputs[4])-1
    return x

def generate_data(num_to_generate):
    inputs=[]
    outputs=[]
    for a in range(num_to_generate):
        x=generate_input()
        y=generate_output(x)
        inputs.append(x)
        outputs.append(y)
    return np.array(inputs), np.array(outputs)

def getData(test=False):
    if(test):
        #test data
        if(os.path.isfile('test_in.npy') and os.path.isfile('test_out.npy')):
            inputs=np.load('test_in.npy')
            outputs=np.load('test_out.npy')
        else:
            inputs,outputs=generate_data(10000)
            np.save('test_in.npy', inputs)
            np.save('test_out.npy',outputs)
    else:
        #train data
        if(os.path.isfile('train_in.npy') and os.path.isfile('train_out.npy')):
            inputs=np.load('train_in.npy')
            outputs=np.load('train_out.npy')
        else:
            inputs,outputs=generate_data(500)
            np.save('train_in.npy', inputs)
            np.save('train_out.npy',outputs)
    return inputs, outputs
    


def nn(num,H):
    nn=Dense(H,kernel_initializer='random_uniform', name='base1_'+str(num))(nn_inputs)
    nn=Dense(1,name='base2_'+str(num),kernel_initializer='random_uniform')(nn)
    return Model(inputs=nn_inputs, outputs=nn)


#Tried different weight initializations
#weight_init=keras.initializers.RandomUniform(minval = 0, maxval = 0.05)
#weight_init=keras.initializers.Zeros()
#weight_init=keras.initializers.RandomNormal(mean=0.1, stddev=0.05, seed=None)
#,kernel_initializer=weight_init
def moe_(H, num_models):
    nn=Dense(H, name='hiddenlayer1')(nn_inputs)
    #nn=Activation('sigmoid', name='hiddenactivation')(nn)
    nn=Dense(num_models,name='gate')(nn)
    return Model(inputs=nn_inputs, outputs=nn)    



def ensemble_average(branches):
    forLambda=[]
    forLambda.extend(branches)
    add= Lambda(lambda x:K.tf.transpose(sum(K.tf.transpose(forLambda[i]) for i in range(0,len(forLambda)))/len(forLambda)), name='final')(forLambda)
    return add

#MOE
def gating_multiplier(gate,branches):
    forLambda=[gate]
    forLambda.extend(branches)
    add= Lambda(lambda x:K.tf.transpose(
        sum(K.tf.transpose(forLambda[i]) * 
            forLambda[0][:, i-1] for i in range(1,len(forLambda))
           )
    ))(forLambda)
    return add



def slices_to_dims(slice_indices):
  """
  Args:
    slice_indices: An [N, k] Tensor mapping to column indices.
  Returns:
    An index Tensor with shape [N * k, 2], corresponding to indices suitable for
    passing to SparseTensor.
  """
  slice_indices = tf.cast(slice_indices, tf.int64)
  num_rows = tf.shape(slice_indices, out_type=tf.int64)[0]
  row_range = tf.range(num_rows)
  item_numbers = slice_indices * num_rows + tf.expand_dims(row_range, axis=1)
  item_numbers_flat = tf.reshape(item_numbers, [-1])
  return tf.stack([item_numbers_flat % num_rows, 
                   item_numbers_flat // num_rows], axis=1)








def plot(model):
    figure = plt.figure(figsize=(18, 16))
    X = np.arange(0,len(outputs))
    tick_plot = figure.add_subplot(1, 1, 1)
    tick_plot.plot(X,outputs,  color='green', linestyle='-', marker='*', label='Actual')
    tick_plot.plot(X, model.predict(inputs),  color='orange',linestyle='-', marker='*', label='Predictions')
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(loc='upper left')
    error=model.evaluate(inputs,outputs)
    plt.title('Total MSE: '+str(error))
    print("Ensemble MSE's:")
    print(errors)
    plt.show()
    
def sub_model_errors(model):
    errors=[]
    test_x, test_y=getData(test=True)
    #print(outputs[:1])
    for i in range(2,num_models+2):
        model1=Model(inputs=nn_inputs, outputs=model.layers[-i].output)
        model1.compile(loss='mse', optimizer=SGD(lr=0.1))
        errors.append(model1.evaluate(test_x, test_y))
    return errors





def cv_squared(x):
  """The squared coefficient of variation of a sample.
  Useful as a loss to encourage a positive distribution to be more uniform.
  Epsilons added for numerical stability.
  Returns 0 for an empty Tensor.
  Args:
    x: a `Tensor`.
  Returns:
    a `Scalar`.
  """
  epsilon = 1e-10
  float_size = tf.to_float(tf.size(x)) + epsilon
  mean = tf.reduce_sum(x) / float_size
  variance = tf.reduce_sum(tf.squared_difference(x, mean)) / float_size
  return variance / (tf.square(mean) + epsilon)






def sparseGating(inputs_,gates=2):
    indi=tf.cast(tf.math.top_k(inputs_,gates, sorted=False).indices,dtype=tf.int64)
    v=tf.math.top_k(inputs_,gates, sorted=False).values

    sparse_indices = slices_to_dims(indi)
    sparse = tf.SparseTensor( indices=sparse_indices, values=tf.reshape(v, [-1]),
                                               dense_shape=tf.cast(tf.shape(inputs_),dtype=tf.int64))
    c=tf.zeros_like(inputs_)
    d=tf.sparse_add(c, sparse)
    z =tf.ones_like(inputs_)*-np.inf
    mask = tf.less_equal(d,  tf.zeros_like(d))
    new_tensor = tf.multiply(z, tf.cast(mask, dtype=tf.float32))

    g=tf.where(tf.is_nan(new_tensor), tf.zeros_like(new_tensor), new_tensor)
    g=tf.sparse_add(g,sparse)
    b=Lambda(lambda a:g)(inputs_)
    return b



def preTrain(weight_file):
    maxVal=np.amax(train_y)
    minVal=np.amin(train_y)
    diff=maxVal-minVal
    for i in range(num_models):
        test_val_min=diff/num_models*i+minVal
        test_val_max=diff/num_models*(i+1)+minVal
        y=[j for j in range(len(train_y)) if train_y[j]>test_val_min and train_y[j]<test_val_max]
        x=train_x[y]
        y=train_y[y]

        models[i].compile(loss='mse', optimizer=SGD(lr=0.1))
        file=str(weight_file)+'_'+str(i)+'.h5'
        checkpointer=ModelCheckpoint(file, monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
        models[i].fit(x,y,epochs=100, verbose=1,batch_size=1,callbacks=[checkpointer])

def load_weights(model,weight_file):
    for a in range(num_models):
        m=models[a]
        file=str(weight_file)+'_'+str(a)+'.h5'
        m.load_weights(file,by_name=True)
        for b in m.layers:
            for l in model.layers:
                if(l.name==b.name):
                    l.set_weights(b.get_weights())
                    print("Loaded: "+l.name)
    for l in model.layers:
        if('base' in l.name):
            l.trainable=True
        else:
            print(l.name)

def moeLoss(yTrue,yPred):
    loss_calc=0
    j=0
    importance=[]
    for i in range(4):
        importance.append(tf.reduce_sum(model.get_layer('gate').output[:,i]))
    for i in reversed(range(2,num_models+2)):
        loss_calc+=(model.get_layer('act').output[:,j]*tf.math.exp(-1/2*(yTrue-model.layers[-i].output)**2))
        j+=1
    loss_calc=-tf.math.log(loss_calc)
    return (loss_calc+.1*cv_squared(importance))


def negativeCorrelation(yTrue,yPred):
    #return K.mean(K.square(yPred - yTrue), axis=-1)
    loss_calc=0
    for i in range(2,num_models+2):
        others=0
        for j in range (2,num_models+2):
            if(j==i):
                continue
            else:
                others+=(model.layers[-j].output-model.layers[-1].output)
        loss_calc+=1/2*(model.layers[-i].output-yTrue)**2+(lambda_*(model.layers[-i].output-model.layers[-1].output)*(others))
    
    return loss_calc



train_x, train_y=getData()
test_x, test_y=getData(test=True)

nn_inputs = Input(shape=(train_x.shape[1:]))

H=1
expert_nodes=2


model_to_run='neg_cor'
if(model_to_run=='single_nn'):
    #Best run on test set MSE: 0.03624
    model=nn("x", H)
    model.compile(loss='mse', optimizer=SGD(lr=0.1))
    model_single_nn=model
    checkpointer=ModelCheckpoint('single.h5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')

        

elif(model_to_run=='moe'):
    #Best run on test set MSE: 0.017827
    num_models=4
    models=[nn(i, H) for i in range(num_models)]

    moe=moe_(expert_nodes, num_models)
    layer=Activation("softmax",name='act')(moe.output)
    gate=gating_multiplier(layer,[m.layers[-1].output for m in models])
    model=Model(inputs=nn_inputs, outputs=gate)
    model.compile(loss=moeLoss, optimizer=SGD(lr=0.1))
    checkpointer=ModelCheckpoint('moe.h5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')

    
elif(model_to_run=='neg_cor'):

    num_models=4
    lambda_=0.1
    models=[nn(i, H) for i in range(num_models)]
    model_out=ensemble_average([models[0].output,models[1].output,models[2].output,models[3].output])
    model=Model(inputs=nn_inputs, outputs=model_out)
    model.compile(loss=negativeCorrelation, optimizer=SGD(lr=0.1))
    checkpointer=ModelCheckpoint('averages.h5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')


elif(model_to_run=='sparse'):
    #Best run on test set MSE models=4, gates=1: 0.02036
    #Best run on test set MSE models=4, gates=2: 0.023095

    #Best run on test set MSE models=8, gates=2: 0.02219
    #Best run on test set MSE models=8, gates=4: 0.017976
    
    num_models=8
    models=[nn(i, H) for i in range(num_models)]
    num_gates=4
    
    moe=moe_(expert_nodes, num_models)
    sparse_layer=sparseGating(moe.output, gates=num_gates)
    layer=Activation("softmax",name='act')(sparse_layer)

    layer=gating_multiplier(layer,[m.layers[-1].output for m in models])
    model=Model(inputs=nn_inputs, outputs=layer)

    checkpointer=ModelCheckpoint('sparse.h5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
    model.compile(loss=moeLoss, optimizer=SGD(lr=0.1))
elif(model_to_run=='moe_pretrained_elu'):
    #Best run on test set MSE: 0.0010685
    num_models=4
    models=[nn(i, H) for i in range(num_models)]

    #preTrain()
    moe=moe_(expert_nodes, num_models)
    layer=Activation("elu",name='act')(moe.output)
    layer=gating_multiplier(layer,[m.layers[-1].output for m in models])
    model=Model(inputs=nn_inputs, outputs=layer)
    load_weights(model)
    checkpointer=ModelCheckpoint('moe_pretrained_elu.h5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
    model.compile(loss='mse', optimizer=SGD(lr=0.1))

elif(model_to_run=='moe_pretrained'):
    #Best run on test set MSE: 0.01444
    num_models=4
    models=[nn(i, H) for i in range(num_models)]

    preTrain('base1')
    moe=moe_(expert_nodes, num_models)
    layer=gating_multiplier(moe.output,[m.layers[-1].output for m in models])
    model=Model(inputs=nn_inputs, outputs=layer)
    load_weights(model,'base1')
    checkpointer=ModelCheckpoint('moe_pretrained.h5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
    model.compile(loss='mse', optimizer=SGD(lr=0.1))
    
elif(model_to_run=='sparse_pretrained'):
    #Best run on test set MSE models=8, gates=2: 0.00907
    #Best run on test set MSE models=8, gates=4: 0.00049734
    
    num_models=8
    models=[nn(i, H) for i in range(num_models)]
    preTrain('base1')
    moe=moe_(expert_nodes, num_models)
    num_gates=4
    
    layer=sparseGating(moe.output, gates=num_gates)
    layer=Activation("softmax",name='act')(layer)
    layer=gating_multiplier(layer,[m.layers[-1].output for m in models])
    model=Model(inputs=nn_inputs, outputs=layer)
    
    load_weights(model,'base1')
    checkpointer=ModelCheckpoint('sparse_pretrained.h5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')

    model.load_weights('sparse_pretrained.h5')
    model.compile(loss='mse', optimizer=SGD(lr=0.1))
    



elif(model_to_run=='sparse_pretrained_elu'):
    #Best run on test set MSE models=8, gates=2: 0.004674
    #Best run on test set MSE models=8, gates=4: 0.001323
    
    num_models=8
    models=[nn(i, H) for i in range(num_models)]
    preTrain('base1')
    moe=moe_(expert_nodes, num_models)
    num_gates=4

    layer=Activation("elu",name='act2')(moe.output)
    
    layer=sparseGating(layer, gates=num_gates)
    layer=Activation("softmax",name='act')(layer)
    layer=gating_multiplier(layer,[m.layers[-1].output for m in models])
    model=Model(inputs=nn_inputs, outputs=layer)
    
    load_weights(model,'base1')
    checkpointer=ModelCheckpoint('sparse_pretrained_elu.h5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
    model.compile(loss='mse', optimizer=SGD(lr=0.1))


model.fit(train_x,train_y,epochs=100, verbose=1,batch_size=1,callbacks=[checkpointer])


#model.summary()
model.compile(loss='mse', optimizer=SGD(lr=0.1))
print("MSE: ")
print(model.evaluate(test_x, test_y))

if(model_to_run!='single_nn'):
    print("sub-model errors:")
    print(sub_model_errors(model))





intermediate_layer_model = Model(inputs=model.input,outputs=model.layers[-2].output) 
val = intermediate_layer_model.predict(train_x)
print(val[:1])

intermediate_layer_model = Model(inputs=model.input,outputs=model.layers[-3].output) 
val = intermediate_layer_model.predict(train_x)
print(val[:1])

intermediate_layer_model = Model(inputs=model.input,outputs=model.layers[-4].output) 
val = intermediate_layer_model.predict(train_x)
print(val[:1])

intermediate_layer_model = Model(inputs=model.input,outputs=model.layers[-5].output) 
val = intermediate_layer_model.predict(train_x)
print(val[:1])

print(train_y[:1])
print(model.predict(train_x[:1]))
