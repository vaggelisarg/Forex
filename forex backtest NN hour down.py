
# coding: utf-8

# In[115]:


import time
import numpy as np
import pandas as pd
import datetime as dt
import cufflinks as cf
from pylab import plt
cf.set_config_file(offline=True)
plt.style.use('seaborn')
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv('EURUSD60.csv')
#line_model_data = pd.read_csv('saved_model_line.csv')
#data['index1'] = np.arange(len(data))
#data = data.set_index(pd.DatetimeIndex(data['date']))
#data = data[(data.index.dayofweek <=4)]


data['linear_low_open_slope'] = data['open']-data['low'].shift(1) # / (data['index1']-data['index1'].shift(1))
data['linear_low_open'] = data['linear_low_open_slope']*2 + data['low'].shift(1)

data['linear_low_low_slope'] = data['low']-data['low'].shift(1) # / (data['index1']-data['index1'].shift(1))
data['linear_low_low'] = data['linear_low_low_slope']*2 + data['low'].shift(1)

data['linear_high_close_slope'] = data['close']-data['high'].shift(1) # / (data['index1']-data['index1'].shift(1))
data['linear_high_close'] = data['linear_high_close_slope']*2 + data['high'].shift(1)

data['linear_open_close_slope'] = data['close']-data['open'].shift(1) # / (data['index1']-data['index1'].shift(1))
data['linear_open_close'] = data['linear_open_close_slope']*2 + data['open'].shift(1)

data['linear_close_close_slope'] = data['close']-data['close'].shift(1) # / (data['index1']-data['index1'].shift(1))
data['linear_close_close'] = data['linear_close_close_slope']*2 + data['close'].shift(1)

data['linear_high_high_slope'] = data['high']-data['high'].shift(1) # / (data['index1']-data['index1'].shift(1))
data['linear_high_high'] = data['linear_high_high_slope']*2 + data['high'].shift(1)

data['linear_high_open_slope'] = data['open']-data['high'].shift(1) # / (data['index1']-data['index1'].shift(1))
data['linear_high_open'] = data['linear_high_open_slope']*2 + data['high'].shift(1)

data['linear_open_open_slope'] = data['open']-data['open'].shift(1) # / (data['index1']-data['index1'].shift(1))
data['linear_open_open'] = data['linear_open_open_slope']*2 + data['open'].shift(1)

data['linear_low_close_slope'] = data['close']-data['low'].shift(1) # / (data['index1']-data['index1'].shift(1))
data['linear_low_close'] = data['linear_low_low_slope']*2 + data['low'].shift(1)


data['L-s']= abs(data['linear_high_open'] - data['open'])
data['L-l']= abs(data['linear_high_open'] - data['low'])
data['L-h']= abs(data['linear_high_open'] - data['high'])
data['L-c']= abs(data['linear_high_open'] - data['close'])

data['L-s1']= abs(data['linear_low_open'] - data['open'])
data['L-l1']= abs(data['linear_low_open'] - data['low'])
data['L-h1']= abs(data['linear_low_open'] - data['high'])
data['L-c1']= abs(data['linear_low_open'] - data['close'])

data['L-s2']= abs(data['linear_low_low'] - data['open'])
data['L-l2']= abs(data['linear_low_low'] - data['low'])
data['L-h2']= abs(data['linear_low_low'] - data['high'])
data['L-c2']= abs(data['linear_low_low'] - data['close'])

data['L-s3']= abs(data['linear_open_close'] - data['open'])
data['L-l3']= abs(data['linear_open_close'] - data['low'])
data['L-h3']= abs(data['linear_open_close'] - data['high'])
data['L-c3']= abs(data['linear_open_close'] - data['close'])

data['L-s4']= abs(data['linear_close_close'] - data['open'])
data['L-l4']= abs(data['linear_close_close'] - data['low'])
data['L-h4']= abs(data['linear_close_close'] - data['high'])
data['L-c4']= abs(data['linear_close_close'] - data['close'])


data['L-s5']= abs(data['linear_open_open'] - data['open'])
data['L-l5']= abs(data['linear_open_open'] - data['low'])
data['L-h5']= abs(data['linear_open_open'] - data['high'])
data['L-c5']= abs(data['linear_open_open'] - data['close'])

data['L-s6']= abs(data['linear_low_close'] - data['open'])
data['L-l6']= abs(data['linear_low_close'] - data['low'])
data['L-h6']= abs(data['linear_low_close'] - data['high'])
data['L-c6']= abs(data['linear_low_close'] - data['close'])

data['x'] = abs(data['close']-data['low'])
data['x2'] = abs(data['close']-data['open'])

data['x1']= data['L-l']/(data['L-c']+ 0.0000000001)
data['x2']= data['L-s']/(data['L-c'] + 0.0000000001)
data['x3']= data['L-h']/(data['L-c'] + 0.0000000001)
data['x4']= data['L-l']/(data['x']+ 0.0000000001)
data['x5']= data['L-s']/(data['x'] + 0.0000000001)
data['x6']= data['L-h']/(data['x'] + 0.0000000001)
data['x7']= data['L-c']/(data['x'] + 0.0000000001)
data['x8']= data['L-l']/(data['x2']+ 0.0000000001)
data['x9']= data['L-s']/(data['x2'] + 0.0000000001)
data['x10']= data['L-h']/(data['x2'] + 0.0000000001)
data['x11']= data['L-c']/(data['x2'] + 0.0000000001)


data['x1n'] = (data['linear_high_open']-  data['close']) 
data['x2n'] = (data['linear_high_open'] - data['high'])
data['x3n'] = (data['linear_high_open']  -data['low'])
data['x4n'] = (data['linear_high_open']  -data['open'])
data['x5n'] = (data['linear_high_open']  -data['close'].shift(1))
data['x6n'] = (data['linear_high_open']  -data['high'].shift(1))
data['x7n'] = (data['linear_high_open']  -data['low'].shift(1))
data['x8n'] = (data['linear_high_open']  -data['open'].shift(1))

data['x1k'] = (data['linear_low_close']-  data['close']) / data['close']*100
data['x2k'] = (data['linear_low_close'] - data['high'])/data['high']*100
data['x3k'] = (data['linear_low_close']  -data['low'])/data['low']*100
data['x4k'] = (data['linear_low_close']  -data['open'])/data['open']*100
data['x5k'] = (data['linear_low_close']  -data['close'].shift(1))/data['close'].shift(1)*100
data['x6k'] = (data['linear_low_close']  -data['high'].shift(1))/ data['high'].shift(1)*100
data['x7k'] = (data['linear_low_close']  -data['low'].shift(1))/ data['low'].shift(1)*100
data['x8k'] = (data['linear_low_close']  -data['open'].shift(1))/data['open'].shift(1)*100


data['x1m'] = data['x1n'] - data['x5n']
data['x2m'] = data['x2n'] -data['x6n']
data['x3m'] = data['x3n'] -data['x7n']
data['x4m'] = data['x4n']- data['x8n']

















data['x1a']= data['L-l1']/(data['L-c1']+ 0.0000000001)
data['x2a']= data['L-s1']/(data['L-c1'] + 0.0000000001)
data['x3a']= data['L-h1']/(data['L-c1'] + 0.0000000001)
data['x4a']= data['L-l1'].shift(1)/(data['L-c1']+ 0.0000000001)
data['x5a']= data['L-s1'].shift(1)/(data['L-c1'] + 0.0000000001)
data['x6a']= data['L-h1'].shift(1)/(data['L-c1'] + 0.0000000001)
data['x7a']= data['L-c1'].shift(1)/(data['L-c1'] + 0.0000000001)

data['x1b']= data['L-l2']/(data['L-c2']+ 0.0000000001)
data['x2b']= data['L-s2']/(data['L-c2'] + 0.0000000001)
data['x3b']= data['L-h2']/(data['L-c2'] + 0.0000000001)
data['x4b']= data['L-l2'].shift(1)/(data['L-c2']+ 0.0000000001)
data['x5b']= data['L-s2'].shift(1)/(data['L-c2'] + 0.0000000001)
data['x6b']= data['L-h2'].shift(1)/(data['L-c2'] + 0.0000000001)
data['x7b']= data['L-c2'].shift(1)/(data['L-c2'] + 0.0000000001)

data['x1c']= data['L-l3']/(data['L-c3']+ 0.0000000001)
data['x2c']= data['L-s3']/(data['L-c3'] + 0.0000000001)
data['x3c']= data['L-h3']/(data['L-c3'] + 0.0000000001)
data['x4c']= data['L-l3'].shift(1)/(data['L-c3']+ 0.0000000001)
data['x5c']= data['L-s3'].shift(1)/(data['L-c3'] + 0.0000000001)
data['x6c']= data['L-h3'].shift(1)/(data['L-c3'] + 0.0000000001)
data['x7c']= data['L-c3'].shift(1)/(data['L-c3'] + 0.0000000001)

data['x1d']= data['L-l4']/(data['L-c4']+ 0.0000000001)
data['x2d']= data['L-s4']/(data['L-c4'] + 0.0000000001)
data['x3d']= data['L-h4']/(data['L-c4'] + 0.0000000001)
data['x4d']= data['L-l4'].shift(1)/(data['L-c4']+ 0.0000000001)
data['x5d']= data['L-s4'].shift(1)/(data['L-c4'] + 0.0000000001)
data['x6d']= data['L-h4'].shift(1)/(data['L-c4'] + 0.0000000001)
data['x7d']= data['L-c4'].shift(1)/(data['L-c4'] + 0.0000000001)

data['x1e']= data['L-l5']/(data['L-c5']+ 0.0000000001)
data['x2e']= data['L-s5']/(data['L-c5'] + 0.0000000001)
data['x3e']= data['L-h5']/(data['L-c5'] + 0.0000000001)
data['x4e']= data['L-l5'].shift(1)/(data['L-c5']+ 0.0000000001)
data['x5e']= data['L-s5'].shift(1)/(data['L-c5'] + 0.0000000001)
data['x6e']= data['L-h5'].shift(1)/(data['L-c5'] + 0.0000000001)
data['x7e']= data['L-c5'].shift(1)/(data['L-c5'] + 0.0000000001)

data['x1f']= data['L-l6']/(data['L-c6']+ 0.0000000001)
data['x2f']= data['L-s6']/(data['L-c6'] + 0.0000000001)
data['x3f']= data['L-h6']/(data['L-c6'] + 0.0000000001)
data['x4f']= data['L-l6'].shift(1)/(data['L-c6']+ 0.0000000001)
data['x5f']= data['L-s6'].shift(1)/(data['L-c6'] + 0.0000000001)
data['x6f']= data['L-h6'].shift(1)/(data['L-c6'] + 0.0000000001)
data['x7f']= data['L-c6'].shift(1)/(data['L-c6'] + 0.0000000001)


data['x1g']= data['L-l']/(data['L-c']+ 0.0000000001)
data['x2g']= data['L-s']/(data['L-c'] + 0.0000000001)
data['x3g']= abs(data['open']-data['close'])/ (data['linear_high_open'] + 0.0000000001)
data['x4g']= (abs(data['low']-data['close'])).shift(1)
data['x5g']= abs(data['low']-data['close'])
data['x6g']= data['x5g'] / (data['x4g'] + 0.0000000001)
data['x7g']= data['L-l'] / (data['L-c']     + 0.0000000001)
data['x8g'] = (data['high'] - data['low']).shift(1)
data['x9g'] = data['high'] - data['low']
data['x10g'] = data['x8g'] / (data['x9g']   + 0.0000000001)
data['x11g'] = data['L-l'] / (data['L-h']    + 0.0000000001)
data['x12g'] = data['high'] - data['open']
data['x13g'] = data['x12g'] / (data['L-s']     + 0.0000000001)
data['x14g'] =  data['x4g'] / (data['x9g'] + 0.0000000001)
data['x15g'] = data['linear_low_open'] - data['low'].shift(1)
data['x16g'] = data['x15g'] / ( data['L-c'] + 0.0000000001)
data['x17g'] = data['linear_low_open']  - data['linear_high_close'] 
data['x18g'] = abs(data['close'] - data['high']) / (abs(data['close'] - data['low']) +0.0000000001)
data['x19g'] = abs(data['close'].shift(1) - data['high']).shift(1) / (abs(data['close'].shift(1) - data['low'].shift(1)) +0.0000000001)



data['Returns'] = data['close'] - data['open']
data['Returns2'] =data['close'].shift(-1) - data['close']
#data['Returns_h'] = data['linear_low_open'] - data['close']
#data['Returns_h'] = data['close'].shift(-1) - data['linear_low_open']





#data['x'] = np.where( (data['Returns'].shift(1)<0) & (data['all']==1)  , 1, 0)

#data['x'] = np.where( (data['Returns'].shift(1)<0), 1, 0)

data['x'] = np.where( (data['Returns'].shift(1)<0) & (data['linear_high_open']>data['close'])   , 1, 0)


#'L-s','L-l','L-h','L-c','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x17'
#'L-s','L-l','L-h','x1g','x2g','x3g','x4g','x5g','x6g','x7g','x8g','x9g','x10g','x11g','x12g','x13g','x17g'
#'x1n','x2n','x3n','x4n','x5n','x6n','x7n','x8n','x9n','x10n','x11n','x12n','x13n','x14n','x15n','x16n'


data['target3'] = np.where(data['Returns2']>0, 1, 0)
#data['target3'] = np.where(data['number']==1, 1, 0)


m = data.loc[data['x']==1]
m3 = m.loc[m['Returns']<0]
m2=m3.dropna()
m2 = m2.reset_index()
#m2['line'] = line_model_data['up']
#m2 = m2.loc[m2['line']<0.5]
#m2 = m2.reset_index()
'L-s','L-l','L-h','L-c',

X = m2.loc[0:400, ['L-s','L-l','L-h','x1g','x2g','x3g','x4g','x5g','x6g','x7g','x8g','x9g','x10g','x11g','x12g','x13g','x17g']]
X2 = m2.loc[401:3200,['L-s','L-l','L-h','x1g','x2g','x3g','x4g','x5g','x6g','x7g','x8g','x9g','x10g','x11g','x12g','x13g','x17g']]
J = m2.loc[:, ['L-s','L-l','L-h','x1g','x2g','x3g','x4g','x5g','x6g','x7g','x8g','x9g','x10g','x11g','x12g','x13g','x17g']]


print(m2)
#m2.to_csv('example1.csv')


# In[116]:


import keras
from keras.utils import np_utils
import numpy as np

# Splitting the data input into X, and the labels y 
#X = np.array(X)[0:400,:]
#X2  = m2.loc[400:519,['L-s','L-l','L-l','L-c','x1','x2','x3','x4','x5','x8','x6','x7','x9','x10','x11','x12','x13']]
#X2 = np.array(X)[701:1111,:]
#X = X.astype('float32')
#X2 = X2.astype('float32')
#y = np.array(y)[1:700]
#y2 = np.array(y)[701:912]

#y= y[:700] 
y = m2.loc[0:400, ['target3']]
y2 = m2.loc[401:3200, ['target3']]
y = keras.utils.to_categorical(y['target3'],2)
y2 = keras.utils.to_categorical(y2['target3'],2)

#y = keras.utils.to_categorical(y,2)
#y2 = keras.utils.to_categorical(y2,2)


# In[117]:


# Checking that the input and output look correct
print("Shape of X:", X.shape)
print("\nShape of y:", y.shape)
print("\nFirst 10 rows of X")
print(X[:10])
print("\nFirst 10 rows of y")
print(y[:10])


# In[118]:


# break training set into training and validation sets
#(X_train, X_test) = X[1:], X[:1]
#(y_train, y_test) = y[1:], y[:1]

# print shape of training set
print('x_train shape:', X.shape)

# print number of training, validation, and test images
print(X.shape[0], 'train samples')
print(y.shape[0], 'test samples')


# In[119]:


# Imports
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

# Building the model
# Note that filling out the empty rank as "0", gave us an extra column, for "Rank 0" students.
# Thus, our input dimension is 7 instead of 6.
model = Sequential()
model.add(Dense(12, activation='relu', input_shape=(17,)))
model.add(Dropout(.2))
model.add(Dense(87, activation='relu'))
model.add(Dropout(.1))
model.add(Dense(2, activation='softmax'))

# Compiling the model
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[120]:


# Training the model0
model.fit(X, y, epochs=2000, batch_size=len(X), verbose=0)


# In[121]:


# Evaluating the model on the training and testing set
score = model.evaluate(X, y)
print("\n Training Accuracy:", score[1])
score = model.evaluate(X2, y2)
print("\n Testing Accuracy:", score[1])


# In[122]:


test_y_predictions = model.predict(X2)


# In[123]:


# Evaluating the model on the training and testing set
score = model.evaluate(X2, y2)
print("\n Training Accuracy:", score[1])
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y2, test_y_predictions)


# In[124]:


pd.set_option('display.max_rows',600)
pd.set_option('display.max_columns',200)
cluster= pd.DataFrame(test_y_predictions)
print(cluster)


# In[125]:


test_y_predictions = model.predict(J)
df = pd.DataFrame(test_y_predictions)
df.columns = ['a', 'b']


m2['position'] = df['b']
#m2['position2'] = line_model_data['up']
m2['total_return'] =np.where((m2['position']>0.5) & (m2['target3']==1) , m2['Returns2'] ,0 )
m2['total_return2'] =np.where((m2['position']>0.5) & (m2['target3']==0), m2['Returns2'],0 )
m2['total_return4'] =np.where((m2['position']<0.5) & (m2['target3']==0) , m2['Returns2'],0 )
m2['total_return6'] =np.where((m2['position']<0.5) & (m2['target3']==1) , m2['Returns2'],0 )


m2['total_return_all']= -(m2['total_return4'] + m2['total_return6']) + (m2['total_return'] + m2['total_return2']) 

m2['strategy'] =  m2['total_return_all']
#m2['strategy'].sum()
#print(m2['position'])


# In[126]:


m2.loc[J.index][['strategy']].cumsum().iplot(yTitle='Cumulative pips')


# In[127]:


m2['g3'] =np.where(m2['total_return']>0, 1, 0)
f1= m2['g3'].sum()
m2['g4'] =np.where(m2['total_return2']<0, 1, 0)
f2 = m2['g4'].sum()
m2['g5'] =np.where(m2['total_return4']<0, 1, 0)
f3 = m2['g5'].sum()
m2['g6'] =np.where(m2['total_return6']>0, 1, 0)
f4 = m2['g6'].sum()
f = (f1+f3)/ (f1 + f2+f3+f4)
print(f)


# In[113]:


print(f1,f2,f3,f4)


# In[114]:


#from keras.models import load_model

#model.save('mymodeldown1200_2_57%.h5')  # creates a HDF5 file 'my_model.h5'


# In[99]:



#test_y_predictions = model.predict(J)
#cluster= pd.DataFrame(test_y_predictions)
#cluster.to_csv('saved_model_line.csv')

