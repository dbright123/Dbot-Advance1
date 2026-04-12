import MetaTrader5 as mt5
import numpy as np

import time
#from datetime import datetime,timezone
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

print("MetaTrader5 package author: ",mt5.__author__)
print("MetaTrader5 package version: ",mt5.__version__)

lot = 0.01

target_market = ["EURGBP","USDCAD","AUDUSD","EURUSD"]
models = []
sc_xs = []
sc_ys = []

n = 0 # counter variable
refreshrate = 20
sl = 0

print("Please wait while i learn from the data showing from your broker")

def learning_data():
    try:
        for target in target_market:
            rates = []
            while(True):
                try:
                    rates = mt5.copy_rates_from_pos(target, mt5.TIMEFRAME_H4, 0, 50000)   
                    print(rates.shape)
                    print("Got successful in the fetching data of ",target)
                    break 
                except Exception as e:
                    print("Data gotten from metatrader is NoneType, trying again... on ",target)
                    time.sleep(5)
                    print(e)
            print("learning from ",target," currently")
            x = []
            y = []
            for i in range(len(rates)):
                x.append([rates[i][0],rates[i][1],rates[i][5]])
                y.append(rates[i][4])

            x = np.array(x)
            y = np.array(y)

            #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.002, random_state=42)
            x_train = x[:-100,:]
            y_train = y[:-100]

            x_test = x[-100:,:]
            y_test = y[-100:]
            sc_x = StandardScaler()
            sc_y = StandardScaler()

            x_train = sc_x.fit_transform(x_train,y_train)
            y_train = sc_y.fit_transform(y_train.reshape((len(y_train),1)))
            y_train = y_train.reshape(-1)

            x_test = sc_x.transform(x_test)

            print(x_test.shape)
            

            regressor = RandomForestRegressor(n_estimators=400,verbose=1, n_jobs=-1, random_state=42)
            regressor.fit(x_train,y_train)

            score = regressor.score(
                x_test,
                sc_y.transform(y_test.reshape((len(y_test),1))).reshape(-1)
            )
            print(score)
            
            if(score <= 0.50):
                print(target," has been trained, but it seems it will be better if attention is giving to the model")
            else:
                models.append(regressor)
                sc_xs.append(sc_x)
                sc_ys.append(sc_y)
            print("Done with ",target)
            time.sleep(10)
    except Exception as e:
        print("An error occured dorring the process of learning, trying the process again ",e)
        time.sleep(10)
        learning_data()

def s_and_d(close_price,k = 6):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(close_price.reshape((len(close_price),1)))
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    centroids = np.sort(centroids, axis=0)
    #print(centroids)
    return centroids


if not mt5.initialize():
    print('Initialization failed, check internet connection. You must have Meta Trader 5 installed.')
else:
    print(mt5.account_info()._asdict())
    print("\n")
    print(mt5.terminal_info()._asdict())
    learning_data()

    while(True):
        try:
            terminal = mt5.terminal_info()
            if(terminal.connected == True and terminal.trade_allowed == True):
                account = mt5.account_info()
                print(account)
                print(account.equity)
                print("AI is functional loading "+target_market[n])

                model = models[n]
                sc_x = sc_xs[n]
                sc_y = sc_ys[n]

                "Running close price in comparisn to supply and demand zone"
                
                t_close_price = []
                trates = mt5.copy_rates_from_pos(target_market[n], mt5.TIMEFRAME_M15, 0, 1000)
                print(trates.shape)
                for i in range(len(trates)):
                    t_close_price.append(trates[i][4])

                t_close_price = np.array(t_close_price)
                cluster = s_and_d(t_close_price)
                print("Cluster value")
                print(cluster)
                
                "Still on the progress of system design"

                rates = mt5.copy_rates_from_pos(target_market[n], mt5.TIMEFRAME_H4, 0, 100)
                print(rates.shape)

                data = []
                close_price = []

                for i in range(len(rates)):
                    data.append([rates[i][0],rates[i][1],rates[i][5]])
                    close_price.append(rates[i][4])

                data = np.array(data)
                close_price = np.array(close_price)

                data = sc_x.transform(data)

                r_squared = model.score(
                    data,
                    sc_y.transform(close_price.reshape((len(close_price),1))).reshape(-1)
                )
                print("stage 1")
                print(r_squared, " is the current prediction model performance")
                
                if(r_squared < 0.50):
                    print(target_market[n]+" will need re-training, please train the model again or check program for error, the prediction is too poor")
                    print("checking other market")
                    learning_data()
                    time.sleep(5)
                else:
                # Get the current datetime in UTC
                    rates = mt5.copy_rates_from_pos(target_market[n], mt5.TIMEFRAME_H4, 0, 1)
                    print(rates)
                    
                    allow_trade = True

                    data=[[rates[0][0],rates[0][1],rates[0][5]]]
                    close_price = [rates[0][4]]
                    data = np.array(data)
                    print(data)
                    close_price = np.array(close_price)
                    print(close_price)    

                    data = sc_x.transform(data)
                    y_pred = model.predict(data)
        
                    y_pred = sc_y.inverse_transform(y_pred.reshape((len(y_pred),1)))
                    y_pred = y_pred.reshape(-1)

                    price = mt5.symbol_info_tick(target_market[n]).bid
                    print("current price for ",target_market[n]," is ",price, " predicted price is ",y_pred[0], " difference in price is ",abs(price - y_pred[0]))

                    print("Trade activation on "+target_market[n])

                    print("Stage 2")
                    if(mt5.positions_total() == 0):
                        if(allow_trade):
                            if(y_pred[0] > price and abs(price - y_pred[0]) > 0.001):
                                result = mt5.Buy(symbol=target_market[n],volume=lot)
                                print(result)
                            elif(y_pred[0] < price and abs(price - y_pred[0]) > 0.001):
                                result = mt5.Sell(symbol=target_market[n],volume=lot)
                                print(result)
                        else:
                            print("Please wait for the starting of a new 4H candle stick")
                    else:
                        print("Adding to other market in the system")
                        #checking if the trade exist so as to modified it 
                        order_symbols = mt5.positions_get()
                        open_price = 0
                        market_exist = False
                        for target_order in order_symbols:
                            print("Stage 3")
                            #print(order_symbol)
                            if(target_market[n] == target_order.symbol):
                                print("seen")
                                market_exist = True
                                #Edit market over here for change in tp or sl
                                print(target_order)
                                open_price = target_order.price_open
                                if(target_order.profit >= target_order.volume * 80  and target_order.sl == 0):
                                    #modify the market
                                    sl = target_order.price_current + target_order.price_open
                                    sl = sl/2.0

                                    request = {
                                        "action": mt5.TRADE_ACTION_SLTP,
                                        "symbol": target_order.symbol,
                                        "sl": sl,
                                        "tp": float(target_order.tp),
                                        "position": target_order.ticket
                                    }
                                    result=mt5.order_send(request)
                                    print(result)
                                else:
                                    #changing of takeprofit incase of sudden volume changes

                                    order_symbols = mt5.positions_get()
                                    for order_symbol in order_symbols:
                                        if(target_market[n] == order_symbol.symbol):
                                            print("seen")
                                            if(float(target_order.tp) != float(y_pred[0])):
                                                request = {
                                                    "action": mt5.TRADE_ACTION_SLTP,
                                                    "symbol": target_order.symbol,
                                                    "sl": float(target_order.sl),
                                                    "tp": y_pred[0],
                                                    "position": target_order.ticket
                                                }
                                                result=mt5.order_send(request)
                                                print(result)
                                            break

                                

                                if(target_order.type == 0 and y_pred[0] < target_order.price_open):
                                    result = mt5.Close(target_order.symbol)
                                    if(y_pred[0] < target_order.price_current):
                                        result = mt5.Sell(symbol=target_market[n],volume=lot)
                                    else:
                                        print("price unstable")
                                elif(target_order.type == 1 and y_pred[0] > target_order.price_open):
                                    result = mt5.Close(target_order.symbol)
                                    if(y_pred[0] > target_order.price_current):
                                        result = mt5.Buy(symbol=target_market[n],volume=lot)
                                    else:
                                        print("price unstable")
                                print(result)
                                break
                        
                        if(market_exist == False):
                            print("Stage 4")
                            if(allow_trade):
                                if(y_pred[0] > price and abs(price - y_pred[0]) > 0.001):
                                    result = mt5.Buy(symbol=target_market[n],volume=lot)
                                    print(result)
                                elif(y_pred[0] < price and abs(price - y_pred[0]) > 0.001):
                                    result = mt5.Sell(symbol=target_market[n],volume=lot)
                                    print(result)
                            else:
                                print("Please wait for the starting of a new 4H candle stick")

                    if(n < len(target_market) - 1):
                        n += 1
                    else:
                        n = 0

                    print("Stage 5")

                    print(mt5.last_error())
                    
                    time.sleep(refreshrate)

            else:
                print("Please make sure metatrade 5 has internet and Algo Trade is Turn On")
                time.sleep(refreshrate)
        except Exception as e:
            print("Sth went wrong with ",target_market[n])
            print(e)
            if(n < len(target_market) - 1):
                n += 1
            else:
                n = 0

            print("Stage 5")

            print(mt5.last_error())
            
            time.sleep(refreshrate)


mt5.shutdown()
quit()