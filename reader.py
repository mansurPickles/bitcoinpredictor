import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#import csv




def normalize():

    #min max feature scaling
    df = pd.read_csv("crossover", skiprows=1, names=['date', 'symbol', 'open', 'high', 'low', 'close',
                                                     'volume btc', 'volume', 'sma10', 'sma22', 'ln', 'high_norm',
                                                     'crossover', 'crossover_norm'])

    size = len(df['high']) #get size
    high = -math.inf
    low = math.inf

    high_norm = []

    for i in range(size):
        temp = df['volume'][i]
        if temp > high:
            high = temp
        if temp < low:
            low = temp

    for i in range(size):
        temp = df['volume'][i]
        high_norm.append((temp - low)/ (high-low))

    df['volume_norm'] = high_norm
    df.to_csv('volume_norm')



def add_crossover():
    df = pd.read_csv("modified_norm", skiprows=1, names=['date', 'symbol', 'open', 'high', 'low', 'close',
                                                         'volume btc', 'volume', 'sma10', 'sma22', 'ln', 'high_norm'])
    size = len(df['close'])

    crossover = []

    for i in range (size):
        crossover.append(df['sma10'][i] - df['sma22'][i])

    high = -math.inf
    low = math.inf

    df['crossover'] = crossover
    crossover_norm = []


    for i in range(size):
        temp = df['crossover'][i]
        if temp > high:
            high = temp
        if temp < low:
            low = temp

    div = max(abs(high), abs(low))

    #cant use normalization because its between 0 - 1 but it loses negative values
    for i in range(size):
        temp = df['crossover'][i]
        crossover_norm.append(temp/div)

    df['crossover_norm'] = crossover_norm

    df.to_csv('crossover')



def reader():
    df = pd.read_csv("Gdax_BTCUSD_d.csv", skiprows=1, names=['date', 'symbol', 'open', 'high', 'low', 'close', 'volume btc', 'volume'])

    sma10 = []  #new column sma 10 day
    sma22 = []  #new column sma 22 day
    average = 0.0
    size = len(df['high']) #get size

    ln = []
    for i in range(size):
        temp = df['high'][i]
        temp = math.log(temp,2.71828)
        ln.append(temp)


    #creation of 10 day SMA
    for i in range(9):
        average += df['high'][i]    #get the average for first SMA 10 day
        sma10.append(0)             #fill the first 9 days with 0

    for i in range(9, size):
        average += df['high'][i]    #get next day and add to average
        sma10.append(average/10)    #calculate the SMA 10 day and append it
        average -= df['high'][i-9]  #remove oldest day

    df['sma10'] = sma10             #append new vector to data frame


    average = 0.0

    for i in range(21):
        average += df['high'][i]        #get the average for first SMA 22 day
        sma22.append(0)                 #fill the first 21 days with 0

    for i in range(21, size):
        average += df['high'][i]        #get next day and add to average
        sma22.append(average / 22)      #calculate the SMA 10 day and append it
        average -= df['high'][i - 21]   #remove oldest day

    df['sma22'] = sma22                 #append new vector to data frame
    df['ln'] = ln

    df.to_csv('modified_w_ln.csv')           #export df to new csv called "modified.csv"

def first_regression():
    df = pd.read_csv("modified_norm", skiprows=1, names=['date', 'symbol', 'open', 'high', 'low', 'close',
                                                             'volume btc', 'volume', 'sma10', 'sma22', 'ln', 'high_norm'])
    #
    # theta0 = 4.8
    # theta1 = .5


    theta0 = .01
    theta1 = 100


    iterations = 1500
    alpha = .0000025
    size = len(df['high_norm'])
    for it in range(iterations):
        temp0 = 0
        temp1 = 0
        for i in range(size):
            hx = theta0 + (theta1 * i)
            cost = hx - df['high_norm'][i]

            temp0 += cost
            temp1 += (cost * i)

        theta0 = theta0 - (alpha * temp0)/size
        theta1 = theta1 - (alpha * temp1)/size
        print(f'cost: {temp0/size}')


    print(f'theta0: {theta0}')
    print(f'theta1: {theta1}')

def graph():

    df = pd.read_csv("modified_w_ln.csv", skiprows=1, names=['date', 'symbol', 'open', 'high', 'low', 'close',
                                                             'volume btc', 'volume', 'sma10', 'sma22', 'ln'])
    size = len(df['close'])

    theta0 = 4.799491281419004
    theta1 = 0.0032453101562049543

    y_pred = []
    index = []

    for i in range(size):
        temp = theta0 + (theta1 * i)
        # temp = pow(2.71828, temp)
        y_pred.append(temp)
        index.append(i)

    plt.scatter(index, df['ln'])
    plt.plot(index,y_pred, color='red')
    plt.show()

def graph2():

    df = pd.read_csv("modified_norm", skiprows=1, names=['date', 'symbol', 'open', 'high', 'low', 'close',
                                                             'volume btc', 'volume', 'sma10', 'sma22', 'ln', 'high_norm'])
    size = len(df['close'])

    theta0 = -0.09697544282640452
    theta1 = 0.0003511482543055084

    y_pred = []
    index = []

    for i in range(size):
        temp = theta0 + (theta1 * i)
        # temp = pow(2.71828, temp)
        y_pred.append(temp)
        index.append(i)

    plt.scatter(index, df['high_norm'])
    plt.plot(index,y_pred, color='red')
    plt.show()


def polynomialregression():

    df = pd.read_csv("modified_norm", skiprows=1, names=['date', 'symbol', 'open', 'high', 'low', 'close',
                                                             'volume btc', 'volume', 'sma10', 'sma22', 'ln', 'high_norm'])
    size = len(df['close'])

    theta0 = 0
    theta1 = 1
    theta2 = 0

    iterations = 1500
    alpha = .000000000000002
    size = len(df['high_norm'])
    for it in range(iterations):
        temp0 = 0
        temp1 = 0
        temp2 = 0
        for i in range(size):
            hx = theta0 + (theta1 * i) + (theta2 * pow(i,2))
            cost = hx - df['high'][i]

            temp0 += cost
            temp1 += (cost * i)
            temp2 += (cost * pow(i,2))

        theta0 = theta0 - (alpha * temp0)/size
        theta1 = theta1 - (alpha * temp1)/size
        theta2 = theta2 - (alpha * temp2)/size
        # print(f'cost: {temp0/size}')

    print(f'theta0: {theta0}')
    print(f'theta1: {theta1}')
    print(f'theta0: {theta2}')



def polynomialregression2():
    df = pd.read_csv("volume_norm", skiprows=1, names=['date', 'symbol', 'open', 'high', 'low', 'close',
                                                       'volume btc', 'volume', 'sma10', 'sma22', 'ln', 'high_norm',
                                                       'crossover', 'crossover_norm', 'volume_norm'])
    size = len(df['close'])

    theta0 = 0      #-1.1478000278800412e-06
    theta1 = 1      #0.9995517939775238
    theta2 = 0      #0.004091166207290906
    theta3 = 0      #2.516215999560053e-07
    theta4 = 0


    """
    
    theta0: -4.179335726486031e-09
    theta1: 0.9999980030023675
    theta2: -0.0008842685747219065
    theta3: -6.445359405395346e-11
    theta4: -3.247493312026266e-11
    
    """

    iterations = 1500
    # alpha = .000000000002
    alpha = .00000000000002


    for it in range(iterations):
        temp0 = 0
        temp1 = 0
        temp2 = 0
        temp3 = 0
        temp4 = 0
        for i in range(size):
            hx = theta0 + (theta1 * i) + (theta2 * pow(i, 2)) + (theta3 * df['crossover_norm'][i]) + (theta4 * df['volume_norm'][i])
            cost = hx - df['ln'][i]

            temp0 += cost
            temp1 += (cost * i)
            temp2 += (cost * pow(i, 2))
            temp3 += (cost * df['crossover_norm'][i])
            temp4 += (cost * df['volume_norm'][i])

        theta0 = theta0 - (alpha * temp0) / size
        theta1 = theta1 - (alpha * temp1) / size
        theta2 = theta2 - (alpha * temp2) / size
        theta3 = theta3 - (alpha * temp3) / size
        theta4 = theta4 - (alpha * temp4) / size

        print(f'cost: {temp0/size}')

    print(f'theta0: {theta0}')
    print(f'theta1: {theta1}')
    print(f'theta2: {theta2}')
    print(f'theta3: {theta3}')
    print(f'theta4: {theta4}')




def plot2d(x1, x2):
    df = pd.read_csv("modified_norm", skiprows=1, names=['date', 'symbol', 'open', 'high', 'low', 'close',
                                                             'volume btc', 'volume', 'sma10', 'sma22', 'ln', 'high_norm'])
    size = len(df['close'])

    theta0 = x1
    theta1 = x2

    y_pred = []
    index = []

    for i in range(size):
        temp = theta0 + (theta1 * i)
        # temp = pow(2.71828, temp)
        y_pred.append(temp)
        index.append(i)

    plt.scatter(index, df['close'])
    plt.plot(index,y_pred, color='red')
    plt.show()

def plot(x1,x2,x3 = 0, x4 = 0):
    df = pd.read_csv("modified_norm", skiprows=1, names=['date', 'symbol', 'open', 'high', 'low', 'close',
                                                             'volume btc', 'volume', 'sma10', 'sma22', 'ln', 'high_norm'])
    size = len(df['close'])

    theta0 = x1
    theta1 = x2
    theta2 = x3
    theta3 = x4

    y_pred = []
    index = []

    for i in range(size):
        temp = theta0 + (theta1 * i) + (theta2 * pow(i,2))
        # temp = pow(2.71828, temp)
        y_pred.append(temp)
        index.append(i)

    plt.scatter(index, df['close'])
    plt.plot(index,y_pred, color='red')
    plt.show()

def costcheck(theta0, theta1, theta2 = 0, theta3=0):
    df = pd.read_csv("crossover", skiprows=1, names=['date', 'symbol', 'open', 'high', 'low', 'close',
                                                         'volume btc', 'volume', 'sma10', 'sma22', 'ln', 'high_norm', 'crossover', 'crossover_norm'])
    size = len(df['close'])
    tcost = 0

    for i in range(size):
        hx = theta0 + (theta1 * i) + (theta2 * pow(i, 2)) + (theta3 * df['crossover_norm'][i])
        cost = hx - df['high'][i]
        tcost += cost
        print(cost)

    print(f'average cost {tcost/size}')



def plot3d():

    
    

    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    theta0 = -4.179335726486031e-09
    theta1 = 0.9999980030023675
    theta2 = -6.445359405395346e-11
    theta3 = -3.247493312026266e-11

    df = pd.read_csv("volume_norm", skiprows=1, names=['date', 'symbol', 'open', 'high', 'low', 'close',
                                                       'volume btc', 'volume', 'sma10', 'sma22', 'ln', 'high_norm',
                                                       'crossover', 'crossover_norm', 'volume_norm'])
    size = len(df['close'])

    x = []
    y = []
    z = []

    for i in range(size):
        x.append(df['crossover_norm'][i])
        hx = theta0 + (theta1 * i) + (theta2 * pow(i, 2)) + (theta3 * df['crossover_norm'][i])
        z.append(hx)
        y.append(df['volume_norm'][i])

    # ax.plot3D(x,y,z, 'red')
    ax.scatter(x, y, z, '-b')
    ax.set_xlabel('crossover_norm', fontsize=12, color='red')
    ax.set_ylabel('volume_norm', fontsize=12, color='red')
    ax.set_zlabel('predicted price', fontsize=12, color='red')




    plt.show()
    




# first_regression()
# graph2()
# normalize()
# add_crossover()
# polynomialregression()
# plot2d(-0.5534116156564547, 5.0737543002622)
# plot3d(1.9750050695121826e-09, 1.0000028366492624, 0.003689729078015151)
# polynomialregression2()
# costcheck(-1.1478000278800412e-06, 0.9995517939775238, 0.004091166207290906, 2.516215999560053e-07)
# normalize()

# df = pd.read_csv("volume_norm", skiprows=1, names = ['date', 'symbol', 'open', 'high', 'low', 'close',
#                                                          'volume btc', 'volume', 'sma10', 'sma22', 'ln', 'high_norm', 'crossover', 'crossover_norm', 'volume_norm'])
# print(df)
plot3d()