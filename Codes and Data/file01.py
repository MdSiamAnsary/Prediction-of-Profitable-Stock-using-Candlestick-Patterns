# Importing necessary packages
import talib
import yfinance as yf
import numpy as np
import pandas as pd
from collections import Counter
import imblearn.over_sampling
from imblearn.over_sampling import RandomOverSampler

# delete the existing csv file before starting the work
import os
if os.path.exists("file.csv"):
  os.remove("file.csv")
#------------------------------------------------------

# Fetching stock data using yahoo finance api
data = yf.download("SPY", start="2000-01-01", end="2022-05-17")
print(data.shape)

# Detecting the candlestick patterns of the stocks 

hammer = talib.CDLHAMMER(data['Open'], data['High'], data['Low'], data['Close'])
data['Hammer'] = hammer
hammer_days = data[data['Hammer'] != 0]

inverted_hammer = talib.CDLINVERTEDHAMMER(data['Open'], data['High'], data['Low'], data['Close'])
data['Inverted Hammer'] = inverted_hammer
invertedhammer_days = data[data['Inverted Hammer'] != 0]

dragonfly_doji = talib.CDLDRAGONFLYDOJI(data['Open'], data['High'], data['Low'], data['Close'])
data['Dragonfly Doji'] = dragonfly_doji
dragonflydoji_days = data[data['Dragonfly Doji'] != 0]

spinning_top = talib.CDLSPINNINGTOP(data['Open'], data['High'], data['Low'], data['Close'])
data['Spinning Top'] = spinning_top
spinningtop_days = data[data['Spinning Top'] != 0]

engulfing = talib.CDLENGULFING(data['Open'], data['High'], data['Low'], data['Close'])
data['Engulfing'] = engulfing
engulfing_days = data[data['Engulfing'] != 0]

harami = talib.CDLHARAMI(data['Open'], data['High'], data['Low'], data['Close'])
data['Harami'] = harami
harami_days = data[data['Harami'] != 0]

piercing = talib.CDLPIERCING(data['Open'], data['High'], data['Low'], data['Close'])
data['Piercing'] = piercing
piercing_days = data[data['Piercing'] != 0]

morning_star = talib.CDLMORNINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])
data['Morning Star'] = morning_star
morningstar_days = data[data['Morning Star'] != 0]

three_white_soldiers = talib.CDL3WHITESOLDIERS(data['Open'], data['High'], data['Low'], data['Close'])
data['Three White Soldiers'] = three_white_soldiers
threewhitesoldiers_days = data[data['Three White Soldiers'] != 0]

doji_star = talib.CDLDOJISTAR(data['Open'], data['High'], data['Low'], data['Close'])
data['Doji Star'] = doji_star
dojistar_days = data[data['Doji Star'] != 0]

morning_doji_star = talib.CDLMORNINGDOJISTAR(data['Open'], data['High'], data['Low'], data['Close'])
data['Morning Doji Star'] = morning_doji_star
morningdojistar_days = data[data['Morning Doji Star'] != 0]

three_inside = talib.CDL3INSIDE(data['Open'], data['High'], data['Low'], data['Close'])
data['Three Inside'] = three_inside
threeinside_days = data[data['Three Inside'] != 0]

three_outside = talib.CDL3OUTSIDE(data['Open'], data['High'], data['Low'], data['Close'])
data['Three Outside'] = three_outside
threeoutside_days = data[data['Three Outside'] != 0]

doji = talib.CDLDOJI(data['Open'], data['High'], data['Low'], data['Close'])
data['Doji'] = doji
doji_days = data[data['Doji'] != 0]

marubozu = talib.CDLMARUBOZU(data['Open'], data['High'], data['Low'], data['Close'])
data['Marubozu'] = marubozu
marubozu_days = data[data['Marubozu'] != 0]

hanging_man = talib.CDLHANGINGMAN(data['Open'], data['High'], data['Low'], data['Close'])
data['Hanging Man'] = hanging_man
hangingman_days = data[data['Hanging Man'] != 0]

shooting_star = talib.CDLSHOOTINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])
data['Shooting Star'] = shooting_star
shootingstar_days = data[data['Shooting Star'] != 0]

gravestone_doji = talib.CDLGRAVESTONEDOJI(data['Open'], data['High'], data['Low'], data['Close'])
data['Gravestone Doji'] = gravestone_doji
gravestonedoji_days = data[data['Gravestone Doji'] != 0]

dark_cloud_cover = talib.CDLDARKCLOUDCOVER(data['Open'], data['High'], data['Low'], data['Close'])
data['Dark Cloud Cover'] = dark_cloud_cover
darkcloudcover_days = data[data['Dark Cloud Cover'] != 0]

evening_star = talib.CDLEVENINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])
data['Evening Star'] = evening_star
eveningstar_days = data[data['Evening Star'] != 0]

'''
Extra column for indicating Bullish (Profitable = 1) or Bearish (Profitable = -1) or Neutral (Profitable = 0)
'''
data['Profitable'] = 0
# Initially all set to 0 (Neutral)

# Columns names
print(data.columns)

# DataFrame to Numpy Array 
arr = data.to_numpy()


'''
    For each stock , finding out if it is Bullish / Bearish / Neutral 
'''
for x in arr:
    sum_bullish = 0
    sum_neutral = 0
    sum_bearish = 0

    for i in range(6, 19):
        sum_bullish = sum_bullish + x[i]

    for j in range(19, 21):
        sum_neutral = sum_neutral + x[j]

    for k in range(21, 26):
        sum_bearish = sum_bearish + x[k]

    if (sum_bullish > sum_neutral and sum_bullish > sum_bearish):
        x[26] = 1
    elif (sum_bearish > sum_neutral and sum_bullish < sum_bearish):
        x[26] = -1


# Numpy Array to DataFrame
newData = pd.DataFrame(arr)
print(newData)

'''
    Dropping columns that are no longer needed (The columns as Candlestick patterns)
'''
newData.drop(newData.iloc[:, 4:26], inplace = True, axis = 1)

# Printing the column names 
print(newData.columns)


# ----------------- Undersampling start ---------------------
# Checking the number of instances of each class 
print(newData[26].value_counts())

# Number of Bullish stocks (Profitable = 1)
bullish_class_len = len(newData[newData[26]==1])
print(bullish_class_len)

'''
    Neutral stocks are most of the stocks. Hence, undersampling needs to be done on them.
    For this, at first, finding out in which rows the stocks are neutral.
    Then, we select n number of neutral stocks randomly in a way that while picking randomly one stock is not chose twice.
    Here, n is number of Bullish stocks. 
'''
neutral_class_len = len(newData[newData[26]==0])
print(neutral_class_len)
# Number of neutral stocks 

neutral_class_indices= newData[newData[26]==0].index
print(neutral_class_indices)
# Indices of neutral stocks 

# Randomly choosing neural stocks for undersampling 
random_neutral_class_indices = np.random.choice(neutral_class_indices, bullish_class_len, replace=False)
print(len(random_neutral_class_indices))

# Indices of bullish stocks
bullish_class_indices= newData[newData[26]==1].index
print(bullish_class_indices)

# Indices of bearish stocks 
bearish_class_indices= newData[newData[26]==-1].index
print(bearish_class_indices)

'''
    Now, we can create a new dataframe where we put the bullish stocks (same count as before),
    bearish stocks (same count as before) and neutral stocks (reduced to match count of bullish stocks)
    as we know in which indices the stocks are. 
'''

# concatenation is done 
under_sample_indices = np.concatenate([random_neutral_class_indices, bullish_class_indices, bearish_class_indices])
under_sample = newData.loc[under_sample_indices]
print(under_sample[26].value_counts())

# ------------------------------ Undersampling done ----------------------

# ------------------------------ Oversampling starts ---------------------

'''
    Now we need to do oversampling of Bearish stocks. 
'''
ros = RandomOverSampler()

# The RandomOverSampler needs features and target to be divided
# Hence, in X , we put the features and in Y, we put the target
X = under_sample[[0,1,2,3]]
Y = under_sample[[26]]

# Resampling / oversampling is done of bearish stocks 
x_ros, y_ros = ros.fit_resample(X,Y.values.ravel())
# x_ros and y_ros are arrays, we need to put them in dataframes and join them

# DataFrame for features
features_data = pd.DataFrame(x_ros)
# DataFrame for target
target_data = pd.DataFrame(y_ros)
print(target_data[0].value_counts())
# Concat the features and target
result = pd.concat([features_data, target_data], axis=1)

# ------------------------ Oversampling done -------------------------------

print(result)
print(result.shape)

# Save the dataframe in a csv file
result.to_csv("file.csv")














        
    








