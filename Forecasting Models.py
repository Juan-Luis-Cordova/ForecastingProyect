# Import blueberries data
import pandas as pd
import numpy as np
import matplotlib as plt
from random import random

bb = pd.read_csv('BlueberriesData.csv', sep=';', decimal=',')
bbp=bb.iloc[0:245,2]
bbv=bb.iloc[0:245,3]

print(bb)
print(bbp)
print(bbv)

# Simple Exponential Smoothing (SES)
def SES(data1):
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    model = SimpleExpSmoothing(data1)
    model_fit = model.fit()
    SESresult = model_fit.predict(len(data1), len(data1))
    print('The period and the prediction from the SES Model is:', SESresult)
SES(bbp)

# Holt Winter’s Exponential Smoothing (HWES)
def HWES(data1):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    model = ExponentialSmoothing(bbp)
    model_fit = model.fit()
    HWESresult = model_fit.predict(len(bbp), len(bbp))
    print('The period and the prediction from the HWES Model is:', HWESresult)
HWES(bbp)

#AUTOREGRESSION (AR) 
def AR(data1):
    from statsmodels.tsa.ar_model import AR
    model = AR(data1)
    model_fit = model.fit()
    ARresult = model_fit.predict(len(bbp), len(bbp))
    print('The period and the prediction from the AR Model is:', ARresult)
AR(bbp)

#MOVING AVERAGE (MA)
def MA(data1):
    from statsmodels.tsa.arima_model import ARMA
    model = ARMA(bbp, order=(0, 1))
    model_fit = model.fit(disp=False)
    MAresult = model_fit.predict(len(data1), len(data1))
    print('The period and the prediction from the MA Model is:', MAresult)
MA(bbp)
 
#Autoregressive Integrated Moving Average (ARMA)
def ARMA(data1):
    from statsmodels.tsa.arima_model import ARMA
    model = ARMA(data1, order=(2, 1))
    model_fit = model.fit(disp=False)
    ARMAresult = model_fit.predict(len(data1), len(data1))
    print('The period and the prediction from the ARMA Model is:', ARMAresult)
ARMA(bbp)

## Autoregressive Integrated Moving-Average (ARIMA)
def ARIMA(data1):
    from statsmodels.tsa.arima_model import ARIMA
    model = ARIMA(data1, order=(1, 1, 1))
    model_fit = model.fit(disp=False)
    ARIMAresult = model_fit.predict(len(data1), len(data1), typ='levels')
    print('The period and the prediction from the ARIMA Model is:', ARIMAresult)
ARIMA(bbp)

# Seasonal Autoregressive Integrated Moving-Average (SARIMA)
def SARIMA(data1):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(data1, order=(1, 0, 1), seasonal_order=(1, 1, 1, 1))
    model_fit = model.fit(disp=False)
    SARIMAresult = model_fit.predict(len(data1), len(data1))
    print('The period and the prediction from the SARIMA Model is:', SARIMAresult)
SARIMA(bbp)

## Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX)
def SARIMAX(data1):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(data1, exog=bbv, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    model_fit = model.fit(disp=False)
    exog2=[bb.iloc[245,3]]
    SARIMAXresult = model_fit.predict(len(data1), len(data1), exog=[exog2])
    print('The period and the prediction from the SARIMAX Model is:', SARIMAXresult)
SARIMAX(bbp)

# Convolutional Neural Network (CNN)
def CNN(data1,data2):

CNN(bbp,bbv)
# Multi Step Convolutional Neural Network (MSCNN)

#Vector Autoregression (VAR)
#Vector Autoregression Moving-Average (VARMA)
#Vector Autoregression Moving-Average with Exogenous Regressors (VARMAX)





