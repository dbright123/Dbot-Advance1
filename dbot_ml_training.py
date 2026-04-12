import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

target_market = ["GBPUSD","USDCAD","AUDUSD","USDCHF","NZDUSD","EURUSD"]

for n in range(len(target_market)):
    print(target_market[n], " currently")
    df = pd.read_csv("Generated"+target_market[n] +" dbot.csv")
    print(len(df))
    df.head()
    # Change the position of the 'B' column
    df = df[['Time', 'Open', 'Tick Volume','Close']]
    df.tail()
    x = df.values
    y = x[:,-1]
    x = x[:,:3]

    print(x)
    print(y)

    

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.002, random_state=0)
    sc_x = StandardScaler()
    sc_y = StandardScaler()

    x_train = sc_x.fit_transform(x_train,y_train)
    y_train = sc_y.fit_transform(y_train.reshape((len(y_train),1)))
    y_train = y_train.reshape(-1)

    x_test = sc_x.transform(x_test)

    print(x_test.shape)

    regressor = RandomForestRegressor(n_estimators=400,verbose=1, n_jobs=10)
    regressor.fit(x_train,y_train)

    joblib.dump(sc_x,target_market[n]+' sc_x.joblib')
    joblib.dump(sc_y,target_market[n]+' sc_y.joblib')
    joblib.dump(regressor,target_market[n]+' regressor.joblib')

    score = regressor.score(
        x_test,
        sc_y.transform(y_test.reshape((len(y_test),1))).reshape(-1)
    )
    print(score)
    

quit()
