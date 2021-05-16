import pandas as pd
import numpy as np
import yfinance as yf

#---------------------------------------------------------------------------------------------------------------------------------

#Download function for tickers (single & multiple)
def get_Ticker(tickers, start, end):
    return yf.download(" ".join(tickers), start = start, end = end)

#---------------------------------------------------------------------------------------------------------------------------------

#function getting all tickers' balance sheet items
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

def finratios(ticker):
    
    fs = yf.Ticker(ticker).financials
    fin = fs.loc
    
    if all(item in financials_items for item in fs.index) == False:
        print(f"\033[30;0;41m WARNING!!! One or more balance sheet items were not recognized\n" + '-'*80)
        
    effective_tax = fin['Income Tax Expense']/fin['Income Before Tax']
    rev = fin['Total Revenue']
    noplat = fin['Ebit']*(1-effective_tax)
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
    
    return pd.DataFrame([rev, ni, noplat, ns_growth, ni_growth, gro_margin, oper_margin, net_margin, nfe,
                         ir_exp, interest_rate, effective_tax, 
                         sga_to_sales, rd_intensity],
                        index = ['totrev','ni','noplat', 'ns_growth','ni_growth','gmargin',
                                 'omargin','nmargin', 'nfe', 'int_exp', "ir", "eftax",
                                 'sga_to_sales','rd_intensity'])
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

def balratios(ticker):
    
    bs = yf.Ticker(ticker).get_balance_sheet()
    bal = bs.loc
   
    if all(item in balance_items for item in bs.index) == False:
        print(f"\033[30;0;41m WARNING!!! One or more balance sheet items were not recognized\n" + '-'*80)
       
    def zerokey_solve(item):
        try:
            x = bal[item]
        except KeyError:
            x = np.repeat(0, bal[:].shape[1])
        return pd.Series([i if i > 0 else 0 for i in x], index = x.index) #pd.Series so that math operations can be applied
    
    totas = bal['Total Assets']
    totli = bal['Total Liab']
    toteq = bal['Total Stockholder Equity']
    coa = bal['Total Current Assets'] - zerokey_solve('Short Term Investments') #all cash is operating
    col = zerokey_solve('Accounts Payable') + zerokey_solve('Other Current Liab') 
    #yf has WC as CurrentAs - CurrentLi, we also substract ST Investments and ST Debt to get OPERATING WC
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
    return {'Total Assets' : totas, 'Total Liabilities' : totli, 'Total Equity' : toteq,
            'Current Operating Assets' : coa, 'Current Operating Liabilties' : col,
            'Working Capital' : wc, 'Operating Working Capital' : oper_wc, 
            'LT Operating Assets' : toa, 'LT Operating Liabilities' : tol,
            'NTA' : nta,
            'Financing Liabilities' : fl, 'Financing Assets' : fa, 
            'Invested Capital' : ic,'Debt-Cash' : debt_net_cash, "Net Debt" : nd, 'Cash and Equivalents' : cash_eq,
            'EEQ' : eeq}

#---------------------------------------------------------------------------------------------------------------------------------

#retrieved ratios for financial analysis
def fin_analysis(ticker):
    fin = finratios(ticker).loc
    bal = pd.DataFrame(balratios(ticker)).T.loc #need to merge formats
     
    # invested capital and interest rate needs tweaking 
    pm = fin['noplat']/fin['totrev']
    roe = fin['ni']/bal['EEQ'].shift(-1)
    roic = fin['noplat']/bal['Invested Capital'].shift(-1)
    ir = fin['int_exp']/bal['Net Debt'].shift(-1)
    nfe_nd = fin['nfe']/bal['Net Debt'].shift(-1)
    fin_lev = bal['Net Debt']/bal['EEQ']
    spr = roic-ir
    icto = fin['totrev']/bal['Invested Capital'].shift(-1)
    ntat = fin['totrev']/bal['NTA'].shift(-1)
    wct = fin['totrev']/bal['Operating Working Capital'].shift(-1)
    
    return pd.DataFrame([pm, roe,roic,nfe_nd,ir, fin_lev,spr, icto,ntat,wct],
                        index = ['PM','ROE','ROIC','netIR','IR','FINLEV','SPR','ICTO','NTAT','WCT'])

#---------------------------------------------------------------------------------------------------------------------------------