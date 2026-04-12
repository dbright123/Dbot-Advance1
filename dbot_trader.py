import MetaTrader5 as mt5
import numpy as np
import joblib
import time
from datetime import datetime,timezone

print("MetaTrader5 package author: ",mt5.__author__)
print("MetaTrader5 package version: ",mt5.__version__)



lot = 0.01

target_market = ["GBPUSD","USDCAD","AUDUSD","USDCHF","NZDUSD","EURUSD"]

models = []
sc_xs = []
sc_ys = []
for market in target_market:
    models.append(joblib.load(market+" regressor.joblib"))
    sc_xs.append(joblib.load(market+" sc_x.joblib"))
    sc_ys.append(joblib.load(market+" sc_y.joblib"))


n = 0 # counter variable
refreshrate = 10

if not mt5.initialize():
    print('Initialization failed, check internet connection. You must have Meta Trader 5 installed.')
else:
    print(mt5.account_info()._asdict())
    print("\n")
    print(mt5.terminal_info()._asdict())
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
                
                if(r_squared <= 0.90):
                    print(target_market[n]+" will need re-training, please train the model again or check program for error, the prediction is too poor")
                    print("checking other market")
                    if(n < len(target_market)-1):
                        n += 1
                    else:
                        n = 0

                    time.sleep(20)
                else:
                # Get the current datetime in UTC
                    rates = mt5.copy_rates_from_pos(target_market[n], mt5.TIMEFRAME_H4, 0, 1)
                    print(rates)

                    rate1h = mt5.copy_rates_from_pos(target_market[n], mt5.TIMEFRAME_H1, 0, 1)

                    print(rate1h[0][0], " compared to ", rates[0][0])
                    
                    allow_trade = True

                    if(rate1h[0][0] != rates[0][0]):
                        print("Please wait for the next opening of 4H candle sticks")
                        allow_trade = False

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
                            if(y_pred[0] > price and abs(price - y_pred[0]) > 0.0008):
                                result = mt5.Buy(symbol=target_market[n],volume=lot)
                                print(result)
                            elif(y_pred[0] < price and abs(price - y_pred[0]) > 0.0008):
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
                                if(target_order.profit >= target_order.volume * 100  and target_order.sl == 0):
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

                                break
                        
                        if(market_exist == False):
                            print("Stage 4")
                            if(allow_trade):
                                if(y_pred[0] > price and abs(price - y_pred[0]) > 0.0008):
                                    result = mt5.Buy(symbol=target_market[n],volume=lot)
                                    print(result)
                                elif(y_pred[0] < price and abs(price - y_pred[0]) > 0.0008):
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
        except:
            print("Sth went wrong with ",target_market[n])
            if(n < len(target_market) - 1):
                n += 1
            else:
                n = 0

            print("Stage 5")

            print(mt5.last_error())
            
            time.sleep(refreshrate)


mt5.shutdown()
quit()