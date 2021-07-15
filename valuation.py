import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------------------------------------------------------

#Download function for tickers (single & multiple)
def get_Ticker(tickers, start, end):
    return yf.download(" ".join(tickers), start = start, end = end)

#---------------------------------------------------------------------------------------------------------------------------------

#function getting all tickers' balance sheet items
##Without Session yfinance does not return balance sheets and financials now
def get_bsitems(tickers):
    stocks = yf.Tickers(" ".join(tickers))
    bs = [stocks.tickers[i].get_balance_sheet() for i in tickers] #non-named list of balance sheets
    bs_items = [bs[i].index.values.tolist() for i in range(len(bs))]
    return set().union(*bs_items)

#---------------------------------------------------------------------------------------------------------------------------------

financials_items = {'Cost Of Revenue',
 'Discontinued Operations',
 'Ebit',
 'Effect Of Accounting Charges',
 'Extraordinary Items',
 'Gross Profit',
 'Income Before Tax',
 'Income Tax Expense',
 'Interest Expense',
 'Minority Interest',
 'Net Income',
 'Net Income Applicable To Common Shares',
 'Net Income From Continuing Ops',
 'Non Recurring',
 'Operating Income',
 'Other Items',
 'Other Operating Expenses',
 'Research Development',
 'Selling General Administrative',
 'Total Operating Expenses',
 'Total Other Income Expense Net',
 'Total Revenue'}

def finratios(ticker, sesh = None):
    
    ##SOLVE for empty financials!!!!
    
    if isinstance(ticker, yf.Ticker):
        fs = ticker.financials
        fin = fs.loc
    elif isinstance(ticker, str):
        fs = yf.Ticker(ticker, session = sesh).financials
        fin = fs.loc
    
    if all(item in financials_items for item in fs.index) == False:
        print(f"\033[30;0;41m WARNING!!! One or more balance sheet items were not recognized\n" + '-'*80)
        
    effective_tax = fin['Income Tax Expense']/fin['Income Before Tax']
    rev = fin['Total Revenue']
    ebit = fin['Ebit']
    noplat = ebit*(1-effective_tax)
    ni = fin['Net Income']
    ns_growth = fin['Total Revenue'].diff(-1)
    ni_growth = fin['Net Income'].diff(-1)
    gro_margin = fin['Cost Of Revenue']/fin['Total Revenue']
    oper_margin = fin['Operating Income']/fin['Total Revenue']
    net_margin = fin['Net Income']/fin['Total Revenue']
    nfe = fin['Ebit'] - fin['Income Before Tax']
    ir_exp = fin['Interest Expense']
    interest_rate = fin['Interest Expense']/fin['Net Income']
    sga_to_sales = fin['Selling General Administrative']/fin['Total Revenue']
    rd_intensity = fin['Research Development']/fin['Total Revenue']
    #effective_tax = pd.Series([effective_tax[i] if effective_tax[i] > 0 else np.nan for i in range(effective_tax.shape[0])],
                             #index = oper_margin.index) accounts for negative ratio
    return pd.DataFrame([rev, ni, ebit, noplat, ns_growth, ni_growth, gro_margin, oper_margin, net_margin, nfe,
                         ir_exp, interest_rate, effective_tax, sga_to_sales, rd_intensity],
                        index = ['Total Revenue','Net Income', 'EBIT', 'NOPLAT', 'gNS','gNI','Gross Margin',
                                 'Operating Margin','Net Margin', 'Net Financial Expense', 'Interest Expense',
                                 "Interest Rate", "Effective Tax Rate",
                                 'SGA_to_Sales','RD Intensity'])
#---------------------------------------------------------------------------------------------------------------------------------

balance_items = {'Accounts Payable',
 'Capital Surplus',
 'Cash',
 'Common Stock',
 'Deferred Long Term Asset Charges',
 'Deferred Long Term Liab',
 'Good Will',
 'Intangible Assets',
 'Inventory',
 'Long Term Debt',
 'Long Term Investments',
 'Minority Interest',
 'Net Receivables',
 'Net Tangible Assets',
 'Other Assets',
 'Other Current Assets',
 'Other Current Liab',
 'Other Liab',
 'Other Stockholder Equity',
 'Property Plant Equipment',
 'Retained Earnings',
 'Short Long Term Debt',
 'Short Term Investments',
 'Total Assets',
 'Total Current Assets',
 'Total Current Liabilities',
 'Total Liab',
 'Total Stockholder Equity',
 'Treasury Stock'}

def balratios(ticker, sesh = None):
    
    if isinstance(ticker, yf.Ticker):
        bs = ticker.get_balance_sheet()
        bal = bs.loc
    
    elif isinstance(ticker, str):
        bs = yf.Ticker(ticker, session = sesh).get_balance_sheet()
        bal = bs.loc
        
    if all(item in balance_items for item in bs.index) == False:
        print(f"\033[30;0;41m WARNING!!! One or more balance sheet items were not recognized\n" + '-'*80)
       
    def zerokey_solve(item): #incorporated KeyError
        try:
            x = bal[item]
        except KeyError:
            x = np.repeat(0, bal[:].shape[1])
        return pd.Series([i if i > 0 else 0 for i in x], index = bs.columns) #pd.Series so that math operations can be applied
    
    totas = bal['Total Assets']
    totli = bal['Total Liab']
    toteq = bal['Total Stockholder Equity']
    coa = bal['Total Current Assets'] - zerokey_solve('Short Term Investments') #all cash is operating
    col = zerokey_solve('Accounts Payable') + zerokey_solve('Other Current Liab') #bal['Total Current Liabilities'] - zerokey_solve('Short Long Term Debt')
    #yahoo finance has WC as CurrentAs - CurrentLi, we also substract ST Investments and ST Debt to get OPERATING WC
    wc = bal['Total Current Assets'] - zerokey_solve('Total Current Liabilities')
    oper_wc = coa - col
    toa = bal['Total Assets'] - bal['Total Current Assets'] - zerokey_solve('Long Term Investments')
    tol = bal['Total Liab'] - bal['Total Current Liabilities'] - zerokey_solve('Long Term Debt')
    #yf has NTA as TotAs - Li - Intangibles - Goodwill
    nta = toa - tol
    #yf has IC has equity + ST debt + LT debt
    ic = nta + oper_wc
    fl = bal['Total Liab'] - tol - col #total li - total li + current li + long term debt - current li + short term debt
    fa = bal['Total Assets'] - toa - coa  #total as - total as + current as + lt inv - cur as + st inv
    debt_net_cash = fl - zerokey_solve('Cash')
    #yf has ND as EQ + LT/ST debt, here it is ST debt + LT debt - ST inv - LT inv (not even excluding cash)
    nd = fl - fa
    cash_eq = bal['Cash'] + zerokey_solve('Short Term Investments')
    eeq = ic - nd
    #test that COA + TOA + FA = Total Assets
    #test that COL + TOL + FL = Total Liab
    #test that COA + TOA + FA - COL - TOL - FL - EQ = 0
    return pd.DataFrame({'Total Assets' : totas, 'Total Liabilities' : totli, 'Total Equity' : toteq,
            'Current Operating Assets' : coa, 'Current Operating Liabilties' : col,
            'Working Capital' : wc, 'Operating Working Capital' : oper_wc, 
            'LT Operating Assets' : toa, 'LT Operating Liabilities' : tol, 'NTA' : nta,
            'Financing Liabilities' : fl, 'Financing Assets' : fa, 
            'Invested Capital' : ic,'Debt-Cash' : debt_net_cash, "Net Debt" : nd, 'Cash and Equivalents' : cash_eq,
            'EEQ' : eeq}).T

#---------------------------------------------------------------------------------------------------------------------------------

#retrieved ratios for financial analysis
def fin_analysis(ticker, sesh = None):
    fin = finratios(ticker, sesh).loc
    bal = balratios(ticker, sesh).loc #need to merge formats
     
    # invested capital and interest rate needs tweaking 
    pm = fin['NOPLAT']/fin['Total Revenue']
    roe = fin['Net Income']/bal['EEQ'].shift(-1)
    roic = fin['NOPLAT']/bal['Invested Capital'].shift(-1)
    ir = fin['Interest Expense']/bal['Net Debt'].shift(-1)
    nfe_nd = fin['Net Financial Expense']/bal['Net Debt'].shift(-1)
    fin_lev = bal['Net Debt']/bal['EEQ']
    spr = roic-ir
    icto = fin['Total Revenue']/bal['Invested Capital'].shift(-1)
    ntat = fin['Total Revenue']/bal['NTA'].shift(-1)
    wct = fin['Total Revenue']/bal['Operating Working Capital'].shift(-1)
    
    return pd.DataFrame([pm, roe,roic,nfe_nd,ir, fin_lev,spr, icto,ntat,wct],
                        index = ['PM','ROE','ROIC','netIR','IR','FINLEV','SPR','ICTO','NTAT','WCT'])

#---------------------------------------------------------------------------------------------------------------------------------

def get_quarter(date):
    return (date.month - 1)//3 + 1

#---------------------------------------------------------------------------------------------------------------------------------

def get_endPrevQuarter(date):
    '''
    Returns the ending date of the last quarter. If the ending date of the CURRENT quarter is given, function returns PREVIOUS quarter ending date, i.e. when
    xxxx-12-31 is given function returns xxxx-09-31 end of Q3.
    '''
    quarter = (date.month - 1)//3 + 1
    return dt.date(date.year, 3*(quarter-1)%12 + 1,1) - dt.timedelta(days = 1)

#---------------------------------------------------------------------------------------------------------------------------------

def get_lastQPrice(ticker, date = dt.date.today()):
    '''
    Returns the closing prices prevailing at the ending date of the last quarter (+ 5 trading days). Uses yf.Ticker.history to fetch the prices
    '''
    cur_quarter = get_quarter(date)
    prev_q_date = get_endPrevQuarter(date)
    # Ensuring to get at least 1 entry, fetch 5 trading days
    delta = prev_q_date + dt.timedelta(days = 5)
    if isinstance(ticker, yf.Ticker):
        price = ticker.history(start = prev_q_date, end = delta)['Close'][1]
    elif isinstance(ticker, str):
        price = yf.Ticker(ticker).history(start = prev_q_date, end = delta)['Close'][1]
    return price

#---------------------------------------------------------------------------------------------------------------------------------

def get_lastQ_mktcap(ticker, date = dt.date.today()):
    '''
    Returns ticker's market capitalisation at the end of the previous quarter,
    using the price prevailing at the end of the last quarter and last reported (implied) sharesOutstanding.
    Implied used if the stock has several classes of stocks
    '''
    price = get_lastQPrice(ticker)
    #Both string and yf.Ticker inputs are allowed
    try:
        if isinstance(ticker,yf.Ticker):
            shares_outstanding = ticker.info['sharesOutstanding'] if ticker.info['impliedSharesOutstanding'] is None else ticker.info['impliedSharesOutstanding']
        elif isinstance(ticker, str):
            t = yf.Ticker(ticker).info
            shares_outstanding = t['sharesOutstanding'] if t['impliedSharesOutstanding'] is None else t['impliedSharesOutstanding']
    ##To handle weird behaviour of impliedSharesOutstanding key which keeps disappearing ???? --> just assign mktcap of the individual stock...
    except KeyError:
        if isinstance(ticker,yf.Ticker):
            shares_outstanding = ticker.info['sharesOutstanding']
        elif isinstance(ticker, str):
            t = yf.Ticker(ticker).info
            shares_outstanding = t['sharesOutstanding']
    return price*shares_outstanding

#---------------------------------------------------------------------------------------------------------------------------------

def Book_Multiples(ticker, sesh = None):
    
    '''
    Returns multiples computed based on reconstructed balance sheets and financials.
    Standartises the numerator by taking the values market cap and enterprise value on the last quarter.
    Numerator is in accordance to the last12M trailing values.
    NOTE: The discrepancies due to differences individual companies' financial calendar,
    i.e. financial reporting habits and timings are not accounted for.
    The market cap and enterprise value are based on standard year's quarters.
    Initialises balance sheets and financials for ticker.
    '''
    
    bal = balratios(ticker, sesh)
    fin = finratios(ticker, sesh)
    mktcap = get_lastQ_mktcap(ticker)
    
    ev = mktcap + bal.loc['Net Debt'][0]
    ev_sales = ev/fin.loc['Total Revenue'][0]
    ev_ebit = ev/fin.loc['EBIT'][0] if fin.loc['EBIT'][0] > 0 else np.nan
    pbv = mktcap/bal.loc['EEQ'][0]
    ps = mktcap/fin.loc['Total Revenue'][0]
    pe = mktcap/fin.loc['Net Income'][0] if fin.loc['Net Income'][0] > 0 else np.nan
    
    return pd.DataFrame([pbv, ps, pe, ev_sales, ev_ebit, ev, mktcap],
                        index = ['P/BV',"P/S", "P/E","EV/Sales","EV/EBIT","EV","MktCap"],
                        columns = [ticker])

#---------------------------------------------------------------------------------------------------------------------------------

#gets multiples by current price (non-adjusted)
def getMultiples(ticker):
    m_names = ['priceToBook','priceToSalesTrailing12Months','enterpriseToRevenue','enterpriseToEbitda','pegRatio']
    #Need to handle KeyError, some tickers return empty output or throw keyerror
    if isinstance(ticker, yf.Ticker):
        m = {name : ticker.info[name] for name in m_names}
        m['PE'] = ticker.info['marketCap']/ticker.info['netIncomeToCommon']
    elif isinstance(ticker, str):
        m = {name : yf.Ticker(ticker).info[name] for name in m_names}
        m['PE'] = yf.Ticker(ticker).info['marketCap']/yf.Ticker(ticker).info['netIncomeToCommon']
    #to get accurate trailing12M PE need to take market cap from the last quarterly report!
    return pd.DataFrame(m.values(), index = ['P/BV',"P/S","EV/Sales","EV/EBITDA","PEG","P/E"])

#==================================================================================================================================
#==================================================================================================================================
#==================================================================================================================================
#=================================================##MULTIPLES VALUATION CLASS##====================================================

class Multiples:
    
    def __init__(self, ticker, sesh = None):
        ### yf.ticker attribute temporary, gonna be passed down further up the chain (parent class or else)
        self.yfticker = yf.Ticker(ticker, session = sesh) 
        self.ticker = ticker
        
        ### ISSUE: balratios and finratios re-initiliase yf.Ticker instead of using class attribute
        ## balratios and finratios could be defined as parent class method further up the road
        self.bal = balratios(self.yfticker, sesh)
        self.fin = finratios(self.yfticker, sesh)
        self.fin_analysis = fin_analysis(self.yfticker, sesh)
        self.mktcap = get_lastQ_mktcap(self.yfticker)
        self.multiples = None
        self.current_mult = None
        
    def get_Multiples(self, current = False):
        
        ###Fetch BOOK multiples
        if current is False:
            ev = self.mktcap + self.bal.loc['Net Debt'][0]

            ev_sales = ev/self.fin.loc['Total Revenue'][0]
            ev_ebit = ev/self.fin.loc['EBIT'][0] if self.fin.loc['EBIT'][0] > 0 else np.nan
            pbv = self.mktcap/self.bal.loc['EEQ'][0]
            ps = self.mktcap/self.fin.loc['Total Revenue'][0]
            pe = self.mktcap/self.fin.loc['Net Income'][0] if self.fin.loc['Net Income'][0]>0 else np.nan
            self.multiples = pd.DataFrame([pbv,ps,pe,ev_sales,ev_ebit,ev,self.mktcap],
                               index = ['P/BV',"P/S", "P/E","EV/Sales","EV/EBIT","EV","MktCap"],
                               columns = [self.ticker])
            return self.multiples
        
        ###Fetch CURRENT multiples (based on current mktcap and enterprise value)
        if current is True:
            self.current_mult = getMultiples(self.yfticker)
            self.current_mult.columns = [self.ticker]
            return self.current_mult
        
        else:
            raise ValueError(f'current takes only True or False: {current}')
            
    def peer_multiples(self, tickers,sesh = None, current = False):
        '''
        Session is defined to fetch yf.Tickers of the peer companies. May throw empty dataframe is sesh is undefined.
        '''
        if current is False:
            mult = [Book_Multiples(comp, sesh = sesh) for comp in tickers]

            if self.multiples is None:
                self.get_Multiples()

            mult.append(self.multiples)
            self.peer_mult = pd.concat(mult, axis = 1)
            
            return self.peer_mult
        
        elif current is True:
            if self.current_mult is None:
                self.get_Multiples(current = True)
                
            t = [yf.Ticker(comp, session = sesh) for comp in tickers]
            m = {comp : getMultiples(ticker) for comp, ticker in zip(tickers, t)}
            m[self.ticker] = self.current_mult
            
            return pd.concat(m, axis = 1).droplevel(level = 1, axis = 1)
        else:
            raise ValueError(f'current takes only True or False: {current}')
    
    def estimate_multiples(self):
        valuation = [self.bal.loc['EEQ'][0], self.fin.loc['Total Revenue'][0], self.fin.loc['Net Income'][0],
                     self.fin.loc['Total Revenue'][0], self.fin.loc['EBIT'][0]]
        min_val = self.peer_mult[:5].apply(np.min,1)*valuation
        avg_val = self.peer_mult[:5].apply(np.mean,1)*valuation
        max_val = self.peer_mult[:5].apply(np.max,1)*valuation
        
        self.mult_est = pd.concat([min_val, avg_val, max_val], axis = 1, keys = ['Min','Avg','Max'])
        return self.mult_est
    
    def plot_Multiples(self, thickness = .5, cmap = "Accent"):
        ##Use convenience plot method
        self.peer_mult[:5].plot.barh(figsize = (12,6), colormap = cmap)
        plt.legend(fontsize = 14, title = "Companies", title_fontsize = 14)
        plt.yticks(font="Helvetica", fontweight = "heavy", fontsize = 12)
        plt.xticks(np.arange(0,self.peer_mult[:5].max().max(),10))
        plt.grid(True, alpha = .75, linewidth = .75, ls = "--")
        
    def visual_mult(self, thickness = .5):
        '''
        Method plots BOOK multiples from min to max range of the peer multiples with averages noted.
        Computes the ranges from .mult_est attribute which uses peer_multiples.
        Requires .peer_mult to be established and estimate_multiples to be ran before.
        '''
        ###Works only with BOOK multiples!!!!
        ###Does not assign attribute of a class --> TO RESOLVE: assigning plot to a class attribute??
        fig, ax = plt.subplots(figsize = (10,6))
        ax.barh(self.mult_est.index, self.mult_est['Min'], label = 'Min', height = thickness)
        ax.barh(self.mult_est.index, (self.mult_est['Max']-self.mult_est['Min']), label = 'Max', height = thickness, left = self.mult_est['Min'])
        
        ###Plot Averages
        for i,bar in zip(range(len(self.mult_est.index)),ax.patches):
            ax.plot((self.mult_est['Avg'][i],self.mult_est['Avg'][i]), (bar.get_y()-.1,bar.get_y()+ thickness + .1),"k-")
            ax.annotate("AVG", (self.mult_est['Avg'][i],bar.get_y()), xytext = (5,10), textcoords = "offset points", rotation = 90, fontsize = 10)
        
        ###Plot MKTCAP and Enterprise Value of the MAIN ticker
        ##Mktcap for Price multiples: pbv, ps, pe
        ax.plot((self.multiples.loc['MktCap'], self.multiples.loc['MktCap']),
                (ax.patches[0].get_y()-.15, ax.patches[2].get_y()+ thickness + .15), "b--x", linewidth = .8, label = "MktCap")
        
        ##Enterprise Value for EV multiples: ev_sales,ev_ebit
        ax.plot((self.multiples.loc['EV'], self.multiples.loc['EV']),
                (ax.patches[3].get_y()-.15, ax.patches[4].get_y()+ thickness +.15), "m--o", linewidth = .8, label = "EV")
        
        ###Cosmetics
        ax.set_title(f"Multiple Valuation Estimates ({self.ticker})", font = "Avenir", fontsize = 14)
        ax.set_xlabel("Enterprise Value/Market Cap")
        ax.xaxis.set_major_formatter(lambda x, p: '${:,.0f}m'.format(int(x)/10**6))
        ax.legend(fontsize = 10)
        ax.grid(True, linewidth = .75, alpha = .7, ls = "--")