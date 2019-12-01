import pandas as pd
import math as mt
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def normalize(df, feature, csvName):
    size = len(df['high'])  # get size
    high = -math.inf
    low = math.inf

    high_norm = []

    for i in range(size):
        # temp = df[feature][i]
        temp = i
        if temp > high:
            high = temp
        if temp < low:
            low = temp

    for i in range(size):
        # temp = df[feature][i]
        temp = i
        high_norm.append((temp - low) / (high - low))

    df[feature] = high_norm
    df.to_csv(csvName)


def add_crossover():
    df = pd.read_csv("modified_norm", skiprows=1, names=['date', 'symbol', 'open', 'high', 'low', 'close',
                                                         'volume btc', 'volume', 'sma10', 'sma22', 'ln', 'high_norm'])
    size = len(df['close'])

    crossover = []

    for i in range(size):
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

    # cant use normalization because its between 0 - 1 but it loses negative values
    for i in range(size):
        temp = df['crossover'][i]
        crossover_norm.append(temp / div)

    df['crossover_norm'] = crossover_norm

    df.to_csv('crossover')


def make10and22(df):
    sma10 = []  # new column sma 10 day
    sma22 = []  # new column sma 22 day
    average = 0.0
    size = len(df['high'])  # get size

    ln = []
    for i in range(size):
        temp = df['high'][i]
        temp = math.log(temp, 2.71828)
        ln.append(temp)

    # creation of 10 day SMA
    for i in range(9):
        average += df['high'][i]  # get the average for first SMA 10 day
        sma10.append(0)  # fill the first 9 days with 0

    for i in range(9, size):
        average += df['high'][i]  # get next day and add to average
        sma10.append(average / 10)  # calculate the SMA 10 day and append it
        average -= df['high'][i - 9]  # remove oldest day

    df['sma10'] = sma10  # append new vector to data frame

    average = 0.0

    for i in range(21):
        average += df['high'][i]  # get the average for first SMA 22 day
        sma22.append(0)  # fill the first 21 days with 0

    for i in range(21, size):
        average += df['high'][i]  # get next day and add to average
        sma22.append(average / 22)  # calculate the SMA 10 day and append it
        average -= df['high'][i - 21]  # remove oldest day

    df['sma22'] = sma22  # append new vector to data frame
    df['ln'] = ln

    df.to_csv('modified_w_ln.csv')  # export df to new csv called "modified.csv"


def reader():
    df = pd.read_csv("date_norm", skiprows=1, names=['date', 'symbol', 'open', 'high', 'low', 'close',
                                                     'volume btc', 'volume', 'sma10', 'sma22', 'ln', 'high_norm',
                                                     'crossover', 'crossover_norm', 'volume_norm', 'date_norm'])

    return df


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

        theta0 = theta0 - (alpha * temp0) / size
        theta1 = theta1 - (alpha * temp1) / size
        print(f'cost: {temp0 / size}')

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
    plt.plot(index, y_pred, color='red')
    plt.show()


def graph2():
    df = pd.read_csv("modified_norm", skiprows=1, names=['date', 'symbol', 'open', 'high', 'low', 'close',
                                                         'volume btc', 'volume', 'sma10', 'sma22', 'ln', 'high_norm'])
    size = len(df['close'])

    theta0 = 4.799491281419004
    theta1 = 0.0003511482543055084

    y_pred = []
    index = []

    for i in range(size):
        temp = theta0 + (theta1 * i)
        # temp = pow(2.71828, temp)
        y_pred.append(temp)
        index.append(i)

    plt.scatter(index, df['high_norm'])
    plt.plot(index, y_pred, color='red')
    plt.show()


def polynomialregression(df, debug):
    size = len(df['close'])
    theta = []

    theta0 = 0  # -1.1478000278800412e-06
    theta1 = 0  # 0.9995517939775238
    theta2 = 0  # 0.004091166207290906
    theta3 = 0  # 2.516215999560053e-07
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
    alpha = .02

    if (debug == False):
        theta0 = -4.179335726486031e-09
        theta1 = 0.9999980030023675
        theta2 = -0.0008842685747219065
        theta3 = -6.445359405395346e-11
        theta4 = -3.247493312026266e-11
        theta.append(theta0)
        theta.append(theta1)
        theta.append(theta2)
        theta.append(theta3)
        theta.append(theta4)

        print('here')
        print(theta)
        return theta

    if (debug == True):
        for it in range(iterations):
            temp0 = 0
            temp1 = 0
            temp2 = 0
            temp3 = 0
            temp4 = 0
            for i in range(size):
                hx = theta0 + (theta1 * df['date_norm'][i]) + (theta2 * pow(df['date_norm'][i], 2)) + (
                            theta3 * df['crossover_norm'][i]) + (theta4 * df['volume_norm'][i])
                cost = hx - df['ln'][i]

                temp0 += cost
                temp1 += (cost * df['date_norm'][i])
                temp2 += (cost * pow(df['date_norm'][i], 2))
                temp3 += (cost * df['crossover_norm'][i])
                temp4 += (cost * df['volume_norm'][i])

            theta0 = theta0 - (alpha * temp0) / size
            theta1 = theta1 - (alpha * temp1) / size
            theta2 = theta2 - (alpha * temp2) / size
            theta3 = theta3 - (alpha * temp3) / size
            theta4 = theta4 - (alpha * temp4) / size

            print(f'cost: {temp0 / size}')

        print(f'theta0: {theta0}')
        print(f'theta1: {theta1}')
        print(f'theta2: {theta2}')
        print(f'theta3: {theta3}')
        print(f'theta4: {theta4}')

        return theta0, theta1, theta2, theta3, theta4


def plot2d(df, x1, x2):
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
    plt.plot(index, y_pred, color='red')
    plt.show()


def plot(df, x1, x2, x3=0, x4=0):
    size = len(df['close'])

    theta0 = x1
    theta1 = x2
    theta2 = x3
    theta3 = x4

    y_pred = []
    index = []

    for i in range(size):
        temp = theta0 + (theta1 * i) + (theta2 * pow(i, 2))
        # temp = pow(2.71828, temp)
        y_pred.append(temp)
        index.append(i)

    plt.scatter(index, df['close'])
    plt.plot(index, y_pred, color='red')
    plt.show()


def plotOriginalData(df):
    size = len(df['close'])
    index = []
    for i in range(size):
        index.append(i)

    plt.scatter(index, df['close'])
    plt.xlabel('time')
    plt.ylabel('close high')
    plt.show()


def costcheck(df, theta0, theta1, theta2=0, theta3=0):

    size = len(df['close'])
    tcost = 0

    for i in range(size):
        hx = theta0 + (theta1 * i) + (theta2 * pow(i, 2)) + (theta3 * df['crossover_norm'][i])
        cost = hx - df['high'][i]
        tcost += cost
        print(cost)

    print(f'average cost {tcost / size}')


def costSingle(date, crossover, volume):
    # theta0 = -4.179335726486031e-09
    # theta1 = 0.9999980030023675
    # theta2 = -0.0008842685747219065
    # theta3 = -6.445359405395346e-11
    # theta4 = -3.247493312026266e-11

    theta0 = 5.134356047292201
    theta1 = 2.5822354709810207
    theta2 = 1.8798891277907974
    theta3 = 0.3112200179135832
    theta4 = 0.7119714647959935

    #remove polynomial term
    pred = theta0 + theta1 * (date) + theta3 * (crossover) + theta4 * (volume)
    #with polynomial term
    # pred = theta0 + theta1*(date) + theta2*(date) + theta3*(crossover) + theta4*(volume)

    return pred


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


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoidDerivative(z):
    return z * (1-z)

def cost(input,weights):
    pass

def ANN(df):
    input = np.matrix([df['date_norm'], df['volume_norm'], df['crossover_norm']])
    input = np.transpose(input)
    weights = np.array([1, 1, 1])
    outputs = df['high_norm']
    print(weights)
    print(input)


    size = len(input)

    for i in range(1):
        cost = 0
        outputValues = 0
        adjustment = []

        outputs = sigmoid(np.dot(input,weights))

        error = np.transpose(input) - outputs
        print('*'*30)
        print(error)
        print('*'*30)
        print(outputs)


        print(np.shape(error))
        print(np.shape(outputs))
        print('*'*30)

        sigmoidOutputs = []
        temp = np.transpose(outputs)

        print(len(temp))

        for j in range(len(temp)):
            # sigmoidOutputs.append(sigmoidDerivative(temp[j]))
            a = sigmoidDerivative(temp[j])
            a = (a.item(0))
            sigmoidOutputs.append(a)
        print(sigmoidOutputs)









        # for j in range(size):
        #     print(f'({round(df["date_norm"][j], 2)} * {weights[0]}) + '
        #           f'({round(df["volume_norm"][j],2)} * {weights[2]}) + '
        #           f'({round(df["crossover_norm"][j],2)} * {weights[2]})')
        #     temp = (np.dot(input[j], weights))
        #     print(temp)
        #     cost += temp
        #     outputValues += outputs[i]
        #
        # error = outputValues - cost
        # print(error)
        # adjustments = error * sigmoidDerivative(outputs)
        # print(adjustments)


df = reader()
ANN(df)

# a = [[1]]
# print(a[0])