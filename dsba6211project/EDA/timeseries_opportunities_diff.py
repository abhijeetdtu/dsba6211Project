%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np

import dsba6211project.Utils.getData as gd

from plotnine import *

colorbg = "#364958"
colorlines= ["#c9e4ca" , "#f4845f" , "#00b4d8", "#ef959d"]

mt = theme(panel_background=element_rect(fill=colorbg)
, plot_background=element_rect(fill=colorbg)
, panel_grid_major=element_blank() , panel_grid_minor=element_blank()
, axis_text=element_text(color="white")
, text=element_text(color="white")
, legend_box_background=element_rect(fill=colorbg)
, legend_background=element_rect(fill=colorbg))



df = gd.GetData().getOpportunityData()

df.shape

df.head()

df["CloseDate"].head()

from sklearn.preprocessing import MinMaxScaler
from math import log10

ts  = df.groupby("CloseDate").agg(amount_sum=("Amount", sum))

tsmonthly = ts.copy()
tsmonthly = tsmonthly["amount_sum"].resample("30d").sum()

ggplot(pd.DataFrame(tsmonthly) , aes(x="tsmonthly.index" , y="amount_sum")) + geom_line(color="white") + mt

from pandas.plotting import autocorrelation_plot

autocorrelation_plot(ts["amount_sum"].resample('30d').sum())

ts =  ts["amount_sum"].resample('W-MON').sum()
ts = ts.loc["2014":"2018"]
ts = ts.diff()

ts = pd.Series(MinMaxScaler((0.1,1)).fit_transform(np.array(ts).reshape(-1,1))[:,0] , index=ts.index)
ts = ts.apply(lambda x:log10(x) if x != 0 else 0)
ts.head()

ts = ts.fillna(method="bfill")
ts= pd.DataFrame(ts , columns=["amount_sum"])

ggplot(ts , aes(x="ts.index" , y="amount_sum")) + geom_line(color="white") + mt


#ts = ts.loc["2014" :]
ts = ts.clip(-0.6 , 0)
#ts = ts.fillna(-1)
ts.shape

ggplot(ts , aes(x="ts.index" , y="amount_sum")) + geom_line(color="white") + mt

valid_size = 24
train_size = ts.shape[0] - valid_size
train_ts = ts[:train_size]
valid_ts =  ts[train_size+1:]
look_back = 4
future = 4
n = 100


def create_dataset(dataset, look_back=1 , future=1):
	dataX, dataY = [], []
	for i in range(dataset.shape[0] -look_back-future-1):
		a = dataset.iloc[i:(i+look_back) , :].values
		dataX.append(a)
		if future == 1:
			dataY.append(dataset.iloc[i + look_back , 0])
		else:
			dataY.append(dataset.iloc[i + look_back:i+look_back+future , 0].values)


	return np.array(dataX), np.stack(dataY)


X_train,y_train = create_dataset(train_ts , look_back , future)
X_test,y_test = create_dataset(valid_ts , look_back , future)

#X_train = X_train.reshape((X_train.shape[0] , X_train.shape[1],1))
#X_test = X_test.reshape((X_test.shape[0] , X_test.shape[1],1))
#y_train = y_train.reshape((y_train.shape[0] , y_train.shape[1] , 1))

X_train.shape
X_test[0].shape

import tensorflow as tf

BATCH_SIZE = 4
BUFFER_SIZE = 100

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


y_train_plt = y_train[:,0] if len(y_train.shape) != 1 else y_train
y_test_plt = y_test[:,0] if len(y_train.shape) != 1 else y_test
train_pred_plt = train_pred[:,0]
test_pred_plt = test_pred[:,0]

pltdf = pd.DataFrame({"x":[i for i in range(y_train.shape[0])],"y":y_train_plt , "yhat":train_pred_plt})
testpltdf = pd.DataFrame({"x":[i+y_train.shape[0] for i in range(len(y_test_plt))],"y":y_test_plt , "yhat":test_pred_plt})
wholepltdf = pd.concat([pltdf , testpltdf])
wholepltdf = pd.melt(wholepltdf , id_vars="x")
wholepltdf["cat"] = wholepltdf["variable"]  + np.where(wholepltdf["x"] <  pltdf.shape[0] , "train" , "test")


p = (ggplot(wholepltdf , aes(x="x" , y="value" , color="cat") ) +
geom_line() +
 scale_x_continuous(limits=(200 , 250))+
mt)
print(p)



from statsmodels.tsa.arima_model import ARIMA
arimaModel = ARIMA(train_ts , (50,0,0) )
model_fit = arimaModel.fit()
print(model_fit.summary())
