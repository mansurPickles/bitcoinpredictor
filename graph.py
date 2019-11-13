import matplotlib.pyplot as plt
import pandas as pd
def graph():
    df = pd.read_csv('modified_w_ln.csv', names=['date', 'symbol', 'open', 'high', 'low', 'close', 'volume btc', 'volume', 'sma10', 'sma22', 'ln'])

    size = len(df['close'])

    X = list()

    for i in range(size):
        X.append(i)

    X_range = []
    Y_range = []

    for i in range(0, size, 2):
        temp = float(df['ln'][i])
        temp = round(temp,2)
        Y_range.append(temp)


        X_range.append(i)


    plt.scatter(X_range,Y_range)
    plt.ylabel('ln')
    plt.xlabel('time')
    plt.show()

graph()