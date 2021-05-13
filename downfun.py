import pandas as pd
import numpy as np
import yfinance as yf

#-----------------------------------------------------------------------------------------------------

#Download function for tickers (single & multiple) --> both individual stocks and portfolio
def get_Ticker(tickers, start, end):
    return yf.download(" ".join(tickers), start = start, end = end)

#-----------------------------------------------------------------------------------------------------

def finratios(ticker):
    fin = yf.Ticker(ticker).financials.loc
    
    ns_growth = fin['Total Revenue'].diff(-1)
    ni_growth = fin['Net Income'].diff(-1)
    gro_margin = fin['Cost Of Revenue']/fin['Total Revenue']
    oper_margin = fin['Operating Income']/fin['Total Revenue']
    net_margin = fin['Net Income']/fin['Total Revenue']
    interest_rate = fin['Interest Expense']/fin['Net Income']
    effective_tax = fin['Income Tax Expense']/fin['Income Before Tax']
    sga_to_sales = fin['Selling General Administrative']/fin['Total Revenue']
    rd_intensity = fin['Research Development']/fin['Total Revenue']
    #effective_tax = pd.Series([effective_tax[i] if effective_tax[i] > 0 else np.nan for i in range(effective_tax.shape[0])],
                             #index = oper_margin.index) accounts for negative ratio
    return pd.DataFrame([ns_growth, ni_growth, gro_margin, oper_margin, net_margin, interest_rate, effective_tax,
                        sga_to_sales, rd_intensity],
                        index = ['ns_growth','ni_growth','gmargin','omargin','nmargin',"ir","eftax",'sga_to_sales','rd_intensity'])

#-----------------------------------------------------------------------------------------------------

def balratios(ticker):
    
    bal = yf.Ticker(ticker).get_balance_sheet().loc
    
    def zerokey_solve(item): #incorporated KeyError
        try:
            x = bal[item]
        except KeyError:
            x = np.repeat(0, bal[:].shape[1])
            
        return [i if i > 0 else 0 for i in x]
    
    totas = bal['Total Assets']
    totli = bal['Total Liab']
    toteq = bal['Total Stockholder Equity']
    try:
        coa = bal['Total Current Assets'] - zerokey_solve('Short Term Investments')
    except KeyError:
        coa = bal['Total Current Assets']
    try:
        col = bal['Total Current Liabilities'] - zerokey_solve('Short Long Term Debt')
    except KeyError:
        col = bal['Total Current Liabilities']
    #yahoo finance has WC as CurrentAs - CurrentLi, we also substract ST Investments and ST Debt to get OPERATING WC
    wc = bal['Total Current Assets'] - zerokey_solve('Total Current Liabilities')
    oper_wc = coa - col
    toa = bal['Total Assets'] - bal['Total Current Assets'] - zerokey_solve('Long Term Investments')
    tol = bal['Total Liab'] - bal['Total Current Liabilities'] - zerokey_solve('Long Term Debt')
    #yf has IC has equity + ST debt + LT debt
    ic = toa - tol + oper_wc
    fl = bal['Total Liab'] - tol - col
    fa = bal['Total Assets'] - toa - coa
    #yf has ND as
    nd = fl - fa
    #test that COA + TOA + FA = Total Assets
    #test that COL + TOL + FL = Total Liab
    #test that COA + TOA + FA - COL - TOL - FL - EQ = 0
    return {'Total Assets' : totas, 'Total Liabilities' : totli, 'Total Equity' : toteq,
            'Current Operating Assets' : coa, 'Current Operating Liabilties' : col,
            'Working Capital' : wc, 'Operating Working Capital' : oper_wc, 
            'LT Operating Assets' : toa, 'LT Operating Liabilities' : tol,
            'Financing Liabilities' : fl, 'Financing Assets' : fa, 
            'Invested Capital' : ic, "Net Debt" : nd}
