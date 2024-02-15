import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


from statsmodels.tsa.seasonal import seasonal_decompose
def plot_seasonality(df,col=None,period=None):
    """plots the period cyclicity of values"""
    plt.clf()
    if not col or not period:
        print("Please choose a series and a period")
    else:
        values = df[col].values
        result = seasonal_decompose(values, model="additive",period=period,extrapolate_trend='freq')
        one_season = result.seasonal[0:period]
        result.plot()
        plt.show()
        plt.figure(figsize=(5,3))
        plt.plot(range(period),one_season)
        plt.title("Zoom on one period")
        plt.plot()

def get_seasonality(values,period):
    """returns trend and cyclicity of values"""
    result = seasonal_decompose(values, model="additive",period=period,extrapolate_trend='freq')
    season = result.seasonal
    trend = result.trend
    return season, trend


def add_temporality(axs,N_total,N_periods):
    """adds vertical bars for each period of x axis"""
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_xlabel('Date')
        for k in range(N_periods+1):
            ax.axvline(x=k*N_total//N_periods, color='black', linewidth=0.5)


def check_stationarity(df,cols,window=100):
    """plots rolling average"""
    fig,axs = plt.subplots(1,len(cols),figsize=(10,3))
    for i,col in enumerate(cols):
        axs[i].plot(range(df.shape[0]-window),df[col].rolling(window).mean()[window:])
        axs[i].set_ylabel(f"Moyenne glissante de {col}")
    return fig, axs


def get_best_period(values,plot=False):
    '''prints period of maximum peak in fft'''
    fft_result = np.fft.fft(values-np.mean(values))
    n = len(values)
    freq = np.fft.fftfreq(n,d=86400) #1 jour = 86400 secondes

    if plot:
        plt.figure(figsize=(5, 3))
        plt.plot(freq[:n//2], np.abs(fft_result)[:n//2])

        plt.title('FFT Amplitudes of Time Series')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    # Find the index of the maximum peak
    max_peak_idx = np.argmax(np.abs(fft_result)[:n//2])
    max_peak_freq = freq[max_peak_idx]
    print("Period : ",round((1/max_peak_freq)/(60*60*24),3))


from sklearn.impute import SimpleImputer
def impute_missing(values,missing_value=0,strategy="mean",fill_value=None):
    print("Nombre de valeurs manquantes : ",len(np.where(values==0)[0]))
    imputed_values = SimpleImputer(missing_values=missing_value, strategy=strategy,fill_value=fill_value).fit_transform(np.array(values).reshape((-1,1)))
    return imputed_values