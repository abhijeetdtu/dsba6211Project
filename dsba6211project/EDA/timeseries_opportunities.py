%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np

import dsba6211project.Utils.getData as gd

from plotnine import *


df = gd.GetData().getOpportunityData()

df.shape

df.head()

df["CloseDate"].head()

from sklearn.preprocessing import MinMaxScaler
from math import log10

ts  = df.groupby("CloseDate").agg(amount_sum=("Amount", sum))
ts =  ts["amount_sum"].resample('30d').sum()
ts = ts[:-8]

#ts = pd.Series(MinMaxScaler((0,1)).fit_transform(np.array(ts).reshape(-1,1))[:,0] , index=ts.index)
ts = ts.apply(lambda x:log10(x) if x != 0 else 0)
ts.head()


def plotLogSeries(ts):
	ts= pd.DataFrame(ts , columns=["amount_sum"])
	ts["amount_sum"] = ts["amount_sum"].fillna(0)
	ts["logamount"] = ts["amount_sum"].apply(lambda x:log10(x) if x != 0 else 0)
	ts["rolling4"] = ts["logamount"].rolling(window=4).mean()
	ts["rolling44"] = ts["rolling4"].rolling(window=4).mean()

	p = (ggplot(ts , aes(x="ts.index" , y="logamount")) +
	    geom_line(color=colorlines[2]) +
	    geom_line(aes(x="ts.index" , y="rolling4") , color=colorlines[0]) +
	    geom_line(aes(x="ts.index" , y="rolling44") , color=colorlines[1]) +
	    geom_smooth(color="yellow") + mt)

	print(p)

plotLogSeries(ts)

ts.shape

valid_size = 36
train_size = ts.shape[0] - valid_size
train_ts = ts[:train_size]
valid_ts =  ts[train_size+1:]
look_back = 12
future = 4
n = 100

def create_dataset(dataset, look_back=1 , future=1):
	dataX, dataY = [], []
	for i in range(dataset.shape[0] -look_back-future-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back:i+look_back+future].values)


	return np.array(dataX), np.stack(dataY)


X_train,y_train = create_dataset(train_ts , look_back , future)
X_test,y_test = create_dataset(valid_ts , look_back , future)

X_train = X_train.reshape((X_train.shape[0] , X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0] , X_test.shape[1],1))
#y_train = y_train.reshape((y_train.shape[0] , y_train.shape[1] , 1))

X_train[0].shape
X_test[0].shape

import tensorflow as tf

BATCH_SIZE = 4
BUFFER_SIZE = 100

X_train.shape
y_train.shape
y_train[0].shape

train_univariate = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((X_test, y_test))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

from keras import Sequential
from keras.layers import Dense, LSTM

def build_model(n,future,shape):
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.LSTM(n, input_shape=shape))
	model.add(tf.keras.layers.Dense(future*2))
	model.add(tf.keras.layers.Dense(future))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

model = build_model(n,future,X_train.shape[-2:])

model.fit(train_univariate,
			steps_per_epoch=100,
			validation_data=val_univariate,
			validation_steps=10,
			epochs=100 ,
			verbose=2)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

import math
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, LSTM

from sklearn.metrics import mean_squared_error
trainScore = math.sqrt(mean_squared_error(y_train, train_pred))
valScore = math.sqrt(mean_squared_error(y_test, test_pred))
print(trainScore , valScore)

valid_size = 36
train_size = ts.shape[0] - valid_size
train_ts = ts[:train_size]
valid_ts =  ts[train_size+1:]

look_back = 12
future = 4
n = 100

def Experiment(train_ts,valid_ts,lookback,future,n):

	def create_dataset(dataset, look_back=1 , future=1):
		dataX, dataY = [], []
		for i in range(dataset.shape[0] -look_back-future-1):
			a = dataset[i:(i+look_back)]
			dataX.append(a)
			dataY.append(dataset[i + look_back:i+look_back+future].values)


		return np.array(dataX), np.stack(dataY)


	X_train,y_train = create_dataset(train_ts , look_back , future)
	X_test,y_test = create_dataset(valid_ts , look_back , future)

	X_train = X_train.reshape((X_train.shape[0] , X_train.shape[1],1))
	X_test = X_test.reshape((X_test.shape[0] , X_test.shape[1],1))
	#y_train = y_train.reshape((y_train.shape[0] , y_train.shape[1] , 1))


	BATCH_SIZE = 4
	BUFFER_SIZE = 100

	train_univariate = tf.data.Dataset.from_tensor_slices((X_train, y_train))
	train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

	val_univariate = tf.data.Dataset.from_tensor_slices((X_test, y_test))
	val_univariate = val_univariate.batch(BATCH_SIZE).repeat()


	def build_model(n,future,shape):
		model = tf.keras.Sequential()
		model.add(tf.keras.layers.LSTM(n, input_shape=shape))
		model.add(tf.keras.layers.Dense(future*2))
		model.add(tf.keras.layers.Dense(future))
		model.compile(loss='mean_squared_error', optimizer='adam')
		return model

	model = build_model(n,future,X_train.shape[-2:])

	model.fit(train_univariate,
				steps_per_epoch=100,
				validation_data=val_univariate,
				validation_steps=10,
				epochs=100 ,
				verbose=2)

	train_pred = model.predict(X_train)
	test_pred = model.predict(X_test)


	trainScore = math.sqrt(mean_squared_error(y_train, train_pred))
	valScore = math.sqrt(mean_squared_error(y_test, test_pred))
	print(trainScore , valScore)
	return trainScore,valScore




params = {
	"lookback" : [1, 4,6,12],
	"future" : [1,2,4],
	"n":[10,20,30,50]
}

from itertools import product

def expand_grid(dictionary):
   return pd.DataFrame([row for row in product(*dictionary.values())],
                       columns=dictionary.keys())

all_params = expand_grid(params)

scores = [Experiment(train_ts,valid_ts,row.lookback,row.future,row.n) for i,row in all_params.iterrows()]


scores[:2]

all_params["train_score"] = np.array(scores)[:,0]
all_params["valid_score"] = np.array(scores)[:,1]


colorbg = "#364958"
colorlines= ["#c9e4ca" , "#f4845f" , "#00b4d8", "#ef959d"]
mt = theme(panel_background=element_rect(fill=colorbg)
, plot_background=element_rect(fill=colorbg)
, panel_grid_major=element_blank() , panel_grid_minor=element_blank()
, axis_text=element_text(color="white")
, text=element_text(color="white")
, legend_box_background=element_rect(fill=colorbg)
, legend_background=element_rect(fill=colorbg))

(ggplot(all_params , aes(x="n" , y="train_score")) +
geom_line(color=colorlines[0]) +
geom_line(aes(y="valid_score") , color=colorlines[1])+mt+
xlab("Number of Hidden Units")+
ylab("Traning (Green) and Validation (orage) MSE"))


(ggplot(all_params , aes(x="factor(lookback)" , y="future" , fill="valid_score"))
+ geom_tile()
+ mt)


pltdf = pd.DataFrame({"x":[i for i in range(y_train.shape[0])],"y":y_train[:,0] , "yhat":train_pred[:,0]})
testpltdf = pd.DataFrame({"x":[i+y_train.shape[0] for i in range(y_test.shape[0])],"y":y_test[:,0] , "yhat":test_pred[:,0]})
wholepltdf = pd.concat([pltdf , testpltdf])
wholepltdf = pd.melt(wholepltdf , id_vars="x")
wholepltdf["cat"] = wholepltdf["variable"]  + np.where(wholepltdf["x"] <  pltdf.shape[0] , "train" , "test")

(ggplot(wholepltdf , aes(x="x" , y="value" , color="cat") ) +
geom_line() +
mt
)
