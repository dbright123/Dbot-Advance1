import requests
from bs4 import BeautifulSoup
import datetime
import time

import MetaTrader5 as mt5

def connection():
    while True:
        try:
            url = f"https://ng.investing.com/economic-calendar/"
            response = requests.get(url)
            if response is not None: 
                return response
            else:
                time.sleep(1)
                print("Try again... on news website")
        except Exception as e:
            print("Try again... on news website ", e)



def initialize():  
    response = connection()   
    soup = BeautifulSoup(response.content, "html.parser")
    news = soup.find(id="economicCalendarData", class_="genTbl closedTbl ecoCalTbl persistArea js-economic-table")
    if news is not None:
        news = news.find("tbody")
        news = news.find_all("tr",class_="js-event-item")
        return(news)
    else: return None
    
    
    
def timeGMT(response):
    soup = BeautifulSoup(response.content, "html.parser")
    gmt = soup.find(id="timeZoneGmtOffsetFormatted")
    gmt = str(gmt).replace('<span id="timeZoneGmtOffsetFormatted">','')
    gmt = gmt.replace('</span>','')
    gmt = gmt.replace('(','')
    gmt = gmt.replace(')','')
    print(gmt)
    return gmt
        

if mt5.initialize():
    print("Connection is established")
    print(mt5.terminal_info())
    print(mt5.account_info())
else:
    print("Connection is not established")


#Initialization state
response = connection()
lot = 1

currency = ['USDCAD','USDJPY','EURUSD','GBPUSD','XAUUSD','AUDUSD']
impact_color = ['blackFont','greenFont','redFont']
order_operation = False
strength = []# 0 is neutral 1 is strong and 2 is weak

hour = 14
min = 30
 
t = ""

gmt = timeGMT(response)
print("Laoding !!!!")

if gmt is not None:
    t = gmt.split(" ")[1].replace('+',"").split(":")[0]
    t = int(t)
    print(t)


while True:
    while True:
        news = initialize()
        if news is not None:
            break
        else:
            print("Try again... on news website")
            time.sleep(5)
    
    # Get the current local time
    now = datetime.datetime.now()
    # Extract hour and minute
    hour = now.hour + ( t - 1 )
    min = now.minute
    d = now.date()
    week = d.weekday()

    print("current time:",hour,":",min)
    
    if week >= 5: 
        print("Time for the news analysis is up, program shutting down")
        break
    elif hour > 22:
        print("Please wait!!!, while news program prepares for the next day")
        time.sleep(60)
    else:
        print("Total number of news ",len(news))
        for i in range(len(news)):
            target_news = news[i].find_all("td")      
            t_currency = str(target_news[1]).split(' ')
            t_currency = t_currency[-1].replace("</td>","")
            
            if t_currency == "USD":#the currency i am concerned about
                #print(target_news)
                t_impact = str(target_news[2]).split("High Volatility Expected")
                #t_impact2 = str(target_news[2]).split("Moderate Volatility Expected")
                if len(t_impact) > 1: 
                    t_time = str(target_news[0]).replace('<td class="first left time js-time" title="">','')
                    t_time = t_time.replace("</td>",'')
                    d_time = t_time.split(":")
                    print(d_time)
                    if len(d_time) <= 1:
                        if len(t_time.split("min")) > 1:   
                            print(t_time, "Please wait as actual value will soon be released")
                    else:
                        if hour < 10:
                            if "0"+str(hour) == d_time[0]:
                                if "0"+str(min) == d_time[1]:
                                    for i in range(len(impact_color)):
                                        t_actual = str(target_news[4]).split(impact_color[i])
                                        if len(t_actual) <= 1 : 
                                            print("The news is still neutral at the moment") 
                                        else:
                                            if i == 1:
                                                print(t_currency, " is very strong and can be used for quick trading")
                                                order_operation = True
                                                strength.append(i)
                                                break
                                            elif i == 2:
                                                print(t_currency, " is very weak and can be used for quick trading")
                                                order_operation = True
                                                strength.append(i)
                                                break
                                            else:
                                                print(t_currency," is neutral, no point trading with it")
                                                break
                                elif str(min) == d_time[1]:
                                    for i in range(len(impact_color)):
                                        t_actual = str(target_news[4]).split(impact_color[i])
                                        if len(t_actual) <= 1 : 
                                            print("The news is still neutral at the moment") 
                                        else:
                                            if i == 1:
                                                print(t_currency, " is very strong and can be used for quick trading")
                                                order_operation = True
                                                strength.append(i)
                                                break
                                            elif i == 2:
                                                print(t_currency, " is very weak and can be used for quick trading")
                                                order_operation = True
                                                strength.append(i)
                                                break
                                            else:
                                                print(t_currency," is neutral, no point trading with it")
                                                break
                                else:
                                    print("Please wait on some minutes")
                                    print(t_time)
                            
                        else:
                            if str(hour) == d_time[0]:
                                if "0"+str(min) == d_time[1]:
                                    for i in range(len(impact_color)):
                                        t_actual = str(target_news[4]).split(impact_color[i])
                                        if len(t_actual) <= 1 : 
                                            print("The news is still neutral at the moment") 
                                        else:
                                            if i == 1:
                                                print(t_currency, " is very strong and can be used for quick trading")
                                                order_operation = True
                                                strength.append(i)
                                                break
                                            elif i == 2:
                                                print(t_currency, " is very weak and can be used for quick trading")
                                                order_operation = True
                                                strength.append(i)
                                                break
                                            else:
                                                print(t_currency," is neutral, no point trading with it")
                                                break
                                elif str(min) == d_time[1]:
                                    for i in range(len(impact_color)):
                                        t_actual = str(target_news[4]).split(impact_color[i])
                                        if len(t_actual) <= 1 : 
                                            print("The news is still neutral at the moment") 
                                        else:
                                            if i == 1:
                                                print(t_currency, " is very strong and can be used for quick trading")
                                                order_operation = True
                                                strength.append(i)
                                                break
                                            elif i == 2:
                                                print(t_currency, " is very weak and can be used for quick trading")
                                                order_operation = True
                                                strength.append(i)
                                                break
                                            else:
                                                print(t_currency," is neutral, no point trading with it")
                                                break
                                else:
                                    print("Please wait on some minutes")
                                    print(t_time)         

            if(i == len(news) - 1 and order_operation):
                #Monitor and Calculate type of trade and also pull out after 5 mins count
                strong  = 0
                weak = 0
                for i in range(len(strength)):
                    #Strength counting 
                    if strength[i] == 1:
                        strong += 1
                    elif strength[i] == 2:
                        weak += 1
                print("total number of strong news ",strong)   
                print("total number of weak news ",weak)  

                monitor = True

                if strong == len(strength):
                    print("USD is strong, and ready to trade")
                    for i in range(len(currency)):
                        if i <= 1:
                            print("Buy market", currency[i])
                            try:
                                result = mt5.Buy(symbol=currency[i],volume=lot, comment = "dbot_news")
                                print(result)
                            except:
                                print("Error in buy market")
                        else:
                            print("Sell market", currency[i])
                            try:
                                result = mt5.Sell(symbol=currency[i],volume=lot, comment = "dbot_news")
                                print(result)
                            except:
                                print("Error in sell market")
                elif weak == len(strength):
                    print("USD is weak, and ready to trade")
                    for i in range(len(currency)):
                        if i > 1:
                            print("Buy market ", currency[i])
                            try:
                                result = mt5.Buy(symbol=currency[i],volume=lot, comment = "dbot_news")
                                print(result)
                            except:
                                print("Error in buy market")
                            
                        else:
                            print("Sell market", currency[i])
                            try:
                                result = mt5.Sell(symbol=currency[i],volume=lot, comment = "dbot_news")
                                print(result)
                            except:
                                print("Error in sell market")
                            
                else:
                    print("USD today news is a waste of time, or very risky")
                    monitor = False

                if monitor:
                    for i in range(3 * 60):
                        time.sleep(1)
                        print("Please wait..., while i make sure break even is given to market ", i)
                        #Edit the prices
                        order_symbols = mt5.positions_get()
                        for target_order in order_symbols:
                            for target_market in currency:
                                print(target_market)
                                if(target_market == target_order.symbol):
                                    print("seen")
                                    #Edit market over here for change in sl
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

                    ## Closing all the market
                    for market in currency:
                        print(market, " closed")
                        result = mt5.Close(market)
                        print(result)
                    print("All market closed for the day")
                order_operation = False
                strength = []
        time.sleep(10)



