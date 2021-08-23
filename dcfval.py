import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import requests
import time
import warnings
from bs4 import BeautifulSoup
from tqdm import tqdm

idx = pd.IndexSlice
#from damoscrape import *
from damoscrape import *


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

SPDR_ETF = {'Technology' : 'XLK', 'Energy' : 'XLE', 'Financials' : 'XLF',
            'Financial Services' : 'XLF', 'Consumer Cyclical' : 'XLY',
            'Consumer Defensive' : 'XLP', 'Healthcare' : 'XLV',
            'Real Estate' : 'XLRE', 'Basic Materials' : 'XLB',
            'Utilities' : 'XLU','Industrials' : 'XLI'}

def finratios(ticker, sesh = None):
    
    '''
    Returns pandas DataFrame with reconstructed financials, including margins, growth rates, interest rate and effectvie tax measures.
    
    First column represents Trailing 12 Months data till the last reported quarter. Remaining columns represent annual financials reports.
    Note: NA values are filled with zeroes
    
    Args
    -----------
    ticker: str or yf.Ticker object
        Company's ticker according to Yahoo Finance. Alternatively, corresponding yf.Ticker object can be passed
    sesh: requests.Session object
        Defined requests.Session() to avoid empty outputs (sometimes due to unrecognized requests)
        
    Warnings
    -----------
    Hands out warning if some financials items are not recognised (based on default pre-existing set of items)
    '''
    
    if isinstance(ticker, yf.Ticker):
        f, f_q = ticker.financials, ticker.quarterly_financials
    else:
        t = yf.Ticker(ticker, session = sesh)
        f, f_q = t.financials, t.quarterly_financials
        
    f_q = f_q.apply(np.sum,1, result_type = 'broadcast').iloc[:, 0]
    fs = pd.concat([f_q, f], axis = 1).fillna(0)
    fin = fs.loc
    
    if all(item in financials_items for item in f.index) is False:
        print(f"\033[30;0;41m WARNING!!! One or more balance sheet items were not recognized\n" + '-'*80)
    
    effective_tax = fin['Income Tax Expense']/fin['Income Before Tax']
    rev = fin['Total Revenue']
    ebit = fin['Ebit']
    noplat = ebit*(1-effective_tax)
    ni = fin['Net Income']
    ns_growth = fin['Total Revenue'].pct_change(-1)
    ni_growth = fin['Net Income'].pct_change(-1)
    gro_margin = fin['Cost Of Revenue']/fin['Total Revenue']
    oper_margin = fin['Operating Income']/fin['Total Revenue']
    net_margin = fin['Net Income']/fin['Total Revenue']
    nfe = fin['Ebit'] - fin['Income Before Tax']
    ir_exp = fin['Interest Expense']
    interest_rate = fin['Interest Expense']/fin['Net Income']
    icr = ebit/-ir_exp
    sga_to_sales = fin['Selling General Administrative']/fin['Total Revenue']
    rd_intensity = fin['Research Development']/fin['Total Revenue']
    
    return pd.DataFrame([rev, ni, ebit, noplat, ns_growth, ni_growth, gro_margin, oper_margin, net_margin,
                         nfe, ir_exp, interest_rate, icr, effective_tax, sga_to_sales, rd_intensity],
                        index = ['Total Revenue','Net Income', 'EBIT', 'NOPLAT', 'gNS','gNI','Gross Margin',
                                 'Operating Margin','Net Margin', 'Net Financial Expense', 'Interest Expense',
                                 "Interest Rate",'ICR', "Effective Tax Rate",'SGA_to_Sales','RD Intensity'])

def balratios(ticker, sesh = None):
    
    '''
    Returns pandas DataFrame with reconstructed balance sheet items more suited for valuation.
    Note: NA values are filled with zeroes
    
    Arguments
    -----------
    ticker: str or yf.Ticker object
        Company's ticker according to Yahoo Finance. Alternatively, corresponding yf.Ticker object can be passed
    sesh: requests.Session object
        Defined requests.Session() to avoid empty outputs (sometimes due to unrecognized requests)
        
    Warnings
    -----------
    Hands out warning if some balance sheet items are not recognised (based on default pre-existing set of items)
    '''
    
    if isinstance(ticker, yf.Ticker):
        bs, bs_q = ticker.balancesheet, ticker.quarterly_balancesheet
    else:
        t = yf.Ticker(ticker, sesh)
        bs, bs_q = t.balancesheet, t.quarterly_balancesheet
    
    bs = pd.concat([bs_q.iloc[:,0], bs], axis = 1).fillna(0)
    bal = bs.loc
    if all(item in balance_items for item in bs.index) is False:
        print(f"\033[30;0;41m WARNING!!! One or more balance sheet items were not recognized\n" + '-'*80)

    def zerokey_solve(item):
        '''
        Helper function dealing with KeyError if item is not found array of zeroes is substituted instead.
        '''
        try:
            x = bal[item]
        except KeyError:
            x = pd.Series(np.repeat(0, bal[:].shape[1]), index = bs.columns)
        return x

    totas = bal['Total Assets']
    totli = bal['Total Liab']
    toteq = bal['Total Stockholder Equity']
    
    ###Operating Items
    ##COA assumets all cash is operating
    coa = bal['Total Current Assets'] - zerokey_solve('Short Term Investments')
    ##Alternatively could be defined as zerokey_solve('Accounts Payable') + zerokey_solve('Other Current Liab')
    col = bal['Total Current Liabilities'] - zerokey_solve('Short Long Term Debt')
    # zerokey_solve('Accounts Payable') + zerokey_solve('Other Current Liab')
    ##Yahoo finance has WC as CurrentAs - CurrentLi
    ##...to get OPERATING WC ST Investments and ST Debt are subtracted
    wc = bal['Total Current Assets'] - zerokey_solve('Total Current Liabilities')
    oper_wc = coa - col
    ##Long Term Operating Items
    toa = bal['Total Assets'] - bal['Total Current Assets'] - zerokey_solve('Long Term Investments')
    tol = bal['Total Liab'] - bal['Total Current Liabilities'] - zerokey_solve('Long Term Debt')
    ##YF has NTA as TotAs - Li - Intangibles - Goodwill
    #NTA = Net Long Term Operating Assets
    nta = toa - tol
    ##YF has IC has equity + ST debt + LT debt
    #LT assets - LTinv - LT Liab + LT Debt + CurAs - ST Inv - CurLi + ST Debt = TotAs - TotLi + LT Debt + StDebt - StInv - LtInv
    #The defined IC does not include short term and long term investments compared to YF!!!
    ic = nta + oper_wc
    ic_inv = nta + oper_wc + zerokey_solve('Long Term Investments') + zerokey_solve('Short Term Investments')
    
    ###Financing Items
    ##TotLi - TotLi + CurrentLi + LT Debt - CurrentLi + ST Debt --> ST + LT Debt
    fl = bal['Total Liab'] - tol - col
    ##TotAs - TotAs + CurAs + LT Investments - CurAs + ST Investments --> LT + ST Investments
    fa = bal['Total Assets'] - toa - coa
    ##Debt Metrics
    debt_net_cash = fl - zerokey_solve('Cash')
    #YF has ND as EQ + LT/ST debt
    #Here defined as ST debt + LT debt - ST inv - LT inv (cash excluded!!!)
    nd = fl - fa
    
    cash_eq = bal['Cash'] + zerokey_solve('Short Term Investments')
    
    ###Total Equity Check
    eeq = ic - nd

    ###Test that COA + TOA + FA = Total Assets, May be exchanged for soft checks only
    assert all(coa + toa + fa == totas), f'Balance Sheet unstable, COA + TOA + FA does not equal Total Assets'
    ###Test that COL + TOL + FL = Total Liab
    assert all(col + tol + fl == totli), f'Balance Sheet unstable, COL + TOL + FL does not equal Total Liabilities'
    ###Test that COA + TOA + FA - COL - TOL - FL - EQ = 0, Soft check only since some tickers on Yahoo Finance have recorded non-matching balance sheets
    ##Some balance sheet items not recorded correctly as equity minority interest preferred equity etc not included in yfinance balance sheet
    #Only warnings thrown out instead of assert statements
    if all(coa + toa + fa - col - tol - fl != toteq):
        warnings.warn(f'WARNING: Balance sheet is unstable, Total Assets + Total Liabilities != EEQ. Proceed with caution: \
        {ticker.ticker if isinstance(ticker,yf.Ticker) else ticker}', UserWarning)
    
    if all(toteq != eeq):
        warnings.warn(f'WARNING: Balance sheet is unstable, IC - ND != EEQ. Proceed with caution: \
        {ticker.ticker if isinstance(ticker,yf.Ticker) else ticker}', UserWarning)
    #assert all(coa + toa + fa - col - tol - fl == toteq), f'Balance Sheet unstable, Total Assets + Total Liabilities != EEQ'

    return pd.DataFrame({'Total Assets' : totas, 'Total Liabilities' : totli, 'Total Equity' : toteq,
                         'Current Operating Assets' : coa, 'Current Operating Liabilities' : col, 'Working Capital' : wc,
                         'Operating Working Capital' : oper_wc, 'LT Operating Assets' : toa, 'LT Operating Liabilities' : tol,
                         'NTA' : nta, 'Financing Liabilities' : fl, 'Financing Assets' : fa, 'Invested Capital' : ic,
                         'IC (incl. Investment)' : ic_inv, 'Debt-Cash' : debt_net_cash, "Net Debt" : nd,
                         'Cash and Equivalents' : cash_eq, 'EEQ' : eeq}).T
    
def fin_analysis(ticker, sesh = None):
    '''
    Returns key financial and valuation metrics, using both finratios and balratios functions
    
    Arguments
    -----------
    ticker: str or yf.Ticker object
        Company's ticker according to Yahoo Finance. Alternatively, corresponding yf.Ticker object can be passed
    sesh: requests.Session object
        Defined requests.Session() to avoid empty outputs (sometimes due to unrecognized requests)
    '''
    
    fin = finratios(ticker, sesh).loc
    bal = balratios(ticker, sesh).loc

    ###Invested capital and interest rate needs tweaking 
    ##ROIC (inc. inv) uses Invested Capital with inclusion of Investments (assumes financial income is part of operating income.)
    pm = fin['NOPLAT']/fin['Total Revenue']
    roe = fin['Net Income']/bal['EEQ'].shift(-1)
    roic = fin['NOPLAT']/bal['Invested Capital'].shift(-1)
    roic_inv = fin['NOPLAT']/bal['IC (incl. Investment)'].shift(-1)
    ir = fin['Interest Expense']/bal['Net Debt'].shift(-1)
    nfe_nd = fin['Net Financial Expense']/bal['Net Debt'].shift(-1)
    fin_lev = bal['Net Debt']/bal['EEQ']
    spr = roic-ir
    icto = fin['Total Revenue']/bal['Invested Capital'].shift(-1)
    ntat = fin['Total Revenue']/bal['NTA'].shift(-1)
    wct = fin['Total Revenue']/bal['Operating Working Capital'].shift(-1)

    return pd.DataFrame([pm, roe,roic, roic_inv, nfe_nd,ir, fin_lev,spr, icto,ntat,wct],
                        index = ['PM','ROE','ROIC', 'ROIC (incl. inv)','netIR','IR','FINLEV','SPR','ICTO','NTAT','WCT'])

#---------------------------------------------------------------------------------------------------------------------------------

def get_quarter(date):
    '''
    Returns current (!CALENDAR!) quarter.
    '''
    return (date.month - 1)//3 + 1

#---------------------------------------------------------------------------------------------------------------------------------

def get_endPrevQuarter(date):
    '''
    Returns the ending date of the last quarter. If the ending date of the CURRENT quarter is given,
    function returns PREVIOUS quarter ending date, i.e. when xxxx-12-31 is given function returns xxxx-09-31 end of Q3.
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

def getMktcap(ticker, date = dt.date.today() + dt.timedelta(days = - 2)):
    '''
    Returns ticker's market capitalisation at the end of the previous quarter,
    using the price prevailing at the end of the last quarter and last reported sharesOutstanding.
    
    Market capitalisation computed using the closing price on the specified date and CURRENT number of shares outstanding.
    Args
    -----------
    ticker: str or yf.Ticker object
        Company's ticker on Yahoo Finance or yf.Ticker from yfinance module
    date: datetime obj, default two days before the current date
        Date when market cap is recorded
    '''
    try:
        if isinstance(ticker, yf.Ticker):
            price = ticker.history(start = date, end = date + dt.timedelta(days = 5)).iloc[1]['Close']
            shares_out = ticker.info['sharesOutstanding'] if ticker.info['impliedSharesOutstanding'] is None else ticker.info['impliedSharesOutstanding']
        else:
            t = yf.Ticker(ticker)
            price = t.history(start = date, end = date + dt.timedelta(days = 5)).iloc[1]['Close']
            shares_out = t.info['sharesOutstanding'] if t.info['impliedSharesOutstanding'] is None else t.info['impliedSharesOutstanding']
    ##Except Clause handling peculiar cases when the impliedSharesOutstanding key instead of being None is not included
    except KeyError:
        if isinstance(ticker,yf.Ticker):
            price = ticker.history(start = date, end = date + dt.timedelta(days = 5)).iloc[1]['Close']
            shares_out = ticker.info['sharesOutstanding']
        else:
            t = yf.Ticker(ticker)
            shares_out = t.info['sharesOutstanding']
            
    return price*shares_out

#---------------------------------------------------------------------------------------------------------------------------------

#==================================================================================================================================
#==================================================================================================================================
#==================================================================================================================================
#=================================================##DCF VAlUATION CLASS##==========================================================

class Valuation:
    '''
    Class to represent a company with methods to compute discounted cash flow (DCF) valuation.
    
    WORKFLOW: 
        1. Default beta is CAPM, if unlevered beta is preferred, run UnleveredBeta
        2. To compute Cost of Debt via synthetic method DefaultSpread and CountryDefaultPremium need to be defined
            2.1 Run/Define CountryRP and DefaultSpread if synthetic method is going to be used
        3. Run Cost of Debt
        4. Compute WACC
        5. Forecast (can be ran independent of previous steps)
        6. HorizonValue requires Forecast and WACC (if wacc_cont is left None)
        7. getPV requires Forecast, HorizonValue and WACC
                
    Attributes:
    ----------------
        ticker: str
            Company's ticker according to Yahoo Finance
        sesh: requests.Session() object
        yfticker: yf.Ticker()
            yf.Ticker() object from yfinance module
        sector: str
            Company's sector according to Yahoo Finance
        damo: DamoData() class object
            Class required to scrape Damodaran's website for macroeconomic data
        fin: pandas.DataFrame
            Reconstructed financials from yf.Ticker().financials
        bal: pandas.DataFrame
            Reconstructed balance sheet from yf.Ticker().get_balance_sheet()
        fin_analysis: pandas.DataFrame
            DataFrame contianing key financial and value metrics
        mktcap: int
            Market Capitalisation according consistent with the last reported quarter
        nd: int
            Last recorded Net Debt from bal dataframe
        beta: int
            CAPM beta computed using 5 years of monthly data
            
    Methods:
    ----------------
        finratios(ticker)
            Returns reconstructed financials statement of specified ticker (company)
        balratios(ticker)
            Returns reconstructed balance sheet of specified ticker (company)
        fin_analysis(ticker)
            Returns key valuation and financial metrics of specified ticker (company)
        CAPM_beta(period, frequency, rf = None)
            Returns CAPM beta and assigns it to the beta attribute of the class
        UnleveredBeta(self, industry, tax = None , firm_de = None)
            Computes Unlevered beta of the company and assigns the value to the beta attribute
        CostofEquity(rf = None, rp = "hist")
            Computes cost of equity (kE) based on specified risk-free rate and risk premium.
            If rf = None, average value over the last month of 10Y US Treasury is considered.
            Assigns the value to ke attribute.
        CountryDefaultPremium(regions, weights)
            Computes CDS as a rough weighted average of specified operating segments and average
            country default spreads. Assigns the value to country_defaultspread attribute
        DefaultSpread(spread = 0, show_synthetic_rating = False, large_cap = True)
            Sets company's default spread (default_spread attribute). Output options can be augmented
            to show Damodaran's inefered default spread from synthetic credit rating.
        CostofDebt(rf = 0, bond_yield = None, method = "book")
            Computes cost of debt (kD) and assigns it to the kd attribute. Cost of debt can be set if
            bond_yield is specified and method is "bond".
        WACC(self, tax)
            Based on cost of equity and cost of debt computs weighted average cost of capital and assigns
            it to the wacc attribute.
        Forecast(gns, pm, xat, wct, tax)
            Based on specified growth rates, profit margins, returns on invested capital, net long-term
            asset turnovers, working capital turnovers and tax rate, computes explicit forecast of key 
            valuation metrics: Net Sales, EBIT, NOPLAT, Net Long-Term Assets, Operating Working Capital,
            Invested Capital, change in IC, ROIC, Free Cash Flow. Assigns the pandas.DataFrame to the
            forecast attribute.
        HorizonValue(gns, roic, ronic, wacc_cont = None)
            Computes continuing value of the company based on specified horizon values of net sales growth,
            roic, ronic, and wacc. Assigns the value to the hv attribute
        getPV(wacc_cont, out = "eqval")
            Using explicit forecast and horizon value returns present value of the cash flows. Assigns the
            estimated value of equity to the estimate attribute.
        SensitivityAnalysis(gns, ronic, range1 = None, range2 = None, plot = False, cmap = 'spring')
            Returns either the sensitivity table or plot. Sensitivity of the estimated valued of equity with
            respect to ROIC and WACC.
    
    Raises:
    ---------------
    ValueError if company's sector is not defined (proxy for non-existent ticker) & if company's sector is
    Utilities/Financials/Bank, etc.
    '''
    
    
    def __init__(self, ticker,sesh = None):
        self.sesh = sesh
        self.ticker = ticker.upper()
        self.yfticker = yf.Ticker(self.ticker, session = self.sesh)
        ###Proxy control for invalid ticker
        try:
            self.sector = self.yfticker.info['sector']
        except Exception:
            raise ValueError(f'Unable to retrieve: {ticker}. Company may be delisted or not recorded')
        
        if any(sector in self.yfticker.info['sector'] for sector in ['Utility','Utili','Financ','Bank']):
            raise ValueError(f'Invalid ticker: {self.sector}. Companies in Financials and Utilities sectors are invalid for valuation.')
        
        #Composition Method instead of inheriting all
        self.damo = DamoData()
        
        self.fin = finratios(self.yfticker, sesh = self.sesh)
        self.bal = balratios(self.yfticker, sesh = self.sesh)
        self.fin_analysis = fin_analysis(self.yfticker, sesh = self.sesh)
        self.mktcap = getMktcap(self.yfticker, date = self.fin.columns[0])
        self.nd = self.bal.loc['Net Debt'][0]
        self.beta = self.CAPMbeta('5y','1m')
        
        ##Attributes to be set later
        self.ke = None
        self.default_spread = None
        self.country_defaultspread = None
        self.kd = None
        self.wacc = None
        self.forecast = None
        self.horizon_wacc = None
        self.hv = None
        self.estimate = None
    
    ###------------------------------------------------------------------------------------------------------------------------------------------------
    ###Dumbproofing via properties
    @property
    def wacc(self):
        return self._wacc
    
    @wacc.setter
    def wacc(self, value):
        if value and value < 0:
            raise ValueError(f'WACC cannot be negative')
        self._wacc = value
        
    @property
    def default_spread(self):
        return self._default_spread
    
    @default_spread.setter
    def default_spread(self, value):
        if value and value < 0:
            raise ValueError(f'Default spread cannot be negative')
        self._default_spread = value
            
    @property
    def ke(self):
        return self._ke
    
    @ke.setter
    def ke(self, value):
        if value and value < 0:
            raise ValueError(f'Cost of Equity (kE) cannot be negative')
        self._ke = value
        
    @property
    def kd(self):
        return self._kd
    
    @kd.setter
    def kd(self, value):
        if value and value < 0:
            raise ValueError(f'Cost of Debt (kD) cannot be negative')
        self._kd = value
        
    
    ###------------------------------------------------------------------------------------------------------------------------------------------------
    ###------------------------------------------------------------------------------------------------------------------------------------------------
    def CAPMbeta(self, period, frequency, rf = None):
        '''
        Returns CAPM beta using data from defined period and frequency.
        
        Market portfolio proxied by S&P 500 index (^GSPC ticker on YF).
        Risk-free rate by default proxied by 10Y Treasury bonds (^TNX ticker on YF).
        Args:
        ----------------
        period: str
            Data window
        frequency: str
            Frequency of the data
        rf: int, float, default None
            By default (rf = None) 10Y Treasury Bond is used as the risk-free rate. Ticker '^TNX' on Yahoo Finance.
            Ability to define own numeric risk free rate.
        '''
        
        ret = self.yfticker.history(period = period).resample(frequency).ffill()['Close']
        rf = yf.Ticker('^TNX').history(period = period).resample(frequency).ffill()['Close'] if rf is None else rf
        mkt = yf.Ticker('^GSPC').history(period = period).resample(frequency).ffill()['Close']

        ex_ret = (ret.pct_change(1)-rf/100).dropna()
        mkt_prem = (mkt.pct_change(1)-rf/100).dropna()

        self.beta = np.polyfit(mkt_prem, ex_ret,1)[0]
        return self.beta
    
    def BottomUpBeta(self, industry, weights = None, tax = None , firm_de = None):
        
        '''
        Returns bottom up beta according to the industry classification of Damodaran.
        Assigns the value to the beta attribute.
        
        If list of industries and weights is given, the industry unlevered beta is computed
        as the weighted average of the respective betas.
        
        Args:
        ----------------
        industry: str, list
            Industry according to the Damodaran classification.
        weights: list
            List of weights corresponding to industries
        tax: int, float, by default None
            Effective Tax rate for the calculation.
            If None effective tax rate from the last quarterly report is used.
        firm_de: int, float, default None
            Company's net debt to market value of equity ratio.
            If None ratio from the last quarterly report is used.
            
        Raises
        ----------------
        AssertionError if weights are defined but industry is not a list
        AssertionError if weights are defined and dimensions of industry and weights to not match
        AssertionError if weights are defined but do not sum up to one
        '''
        
        if weights:
            assert isinstance(industry, list), f'Weights given but industry is not list argument'
            assert len(industry) == len(weights), f'Dimensions of industry and weights do not match'
            assert sum(weights) == 1, f'Weights do not sum up to one!'
        
        tax = self.fin.loc['Effective Tax Rate'][0] if tax is None else tax
        firm_de = self.nd/self.mktcap if firm_de is None else firm_de
        betas = IndustryData('betas').parseTable().loc[industry, 'Unlevered beta']
        if isinstance(industry, list):
            ind_unlev_beta = sum([i * j for i, j in zip(betas, weights)])
            self.beta = ind_unlev_beta*(1+firm_de*(1-tax))
        if isinstance(industry, str):
            self.beta = betas*(1+firm_de*(1-tax))
        
        return self.beta
            
    ###------------------------------------------------------------------------------------------------------------------------------------------------
    def CostofEquity(self, rf = None, rp = "hist"):
        '''
        Returns computed Cost of Equity using the before specified beta (by default 5y monthly CAPM beta). Assigns the value to the
        `ke` attribute.
        
        Risk free rate by default is smoothed rate on 10Y Treasury bond. 'Smoothed' meaning average value across the last 30 days.
        
        Args:
        ----------------
        rf: int, float, default None
            By default 10Y Treasury Bond is used as risk-free rate. User defined (int, float) value can be specified.
        rp: str, int, float, default 'hist'
            Either 'hist' for historical equity risk premium and 'impl' for implied equity risk premium according to Damodaran data.
            Alternatively, if int, float is inputed the the corresponding value is used as the risk premium.
            
        Raises
        ----------------
        ValueError if rp argument wrongly specified (int, float, 'hist', and 'impl' entries eligible)
        '''
        
        rf = yf.Ticker('^TNX').history(start = dt.date.today() - dt.timedelta(days = 30))['Close'].mean() if rf is None else rf
                             
        if rp == 'hist':
            erp_table = self.damo.getERP(sort = rp)
            erp = erp_table.iloc[erp_table.shape[0]-1][idx['Annual Risk Premium', 'Historical risk premium']]
        elif rp == 'impl':
            erp_table = self.damo.getERP(sort = rp)
            erp = erp_table.iloc[erp_table.shape[0]-1]['Implied Premium (FCFE)']
        elif isinstance(rp, (int, float)):
            erp = rp
        else:
            raise ValueError(f'Invalid risk premium method: {rp}. Historical (\'hist\'), Implied (\'impl\') or int/float value eligible.')
            
        self.ke = rf/100 + self.beta*erp
        
        return self.ke
    ###------------------------------------------------------------------------------------------------------------------------------------------------
    def CountryDefaultPremium(self, regions, weights):
        '''
        Returns country default premium based on user inputted operating geographic regions and corresponding weights. Assigns the value to the 
        `country_defaultspread` attribute.
        
        Args:
        ----------------
        regions: list
            List of operating regions. The respective country default premiums are based on Damodaran's Country Risk Premium dataset.
            Available regions: Middle East, Eastern Europe & Russia, Western Europe, Africa, Central and South America, Caribbean,
            Australia & New Zealand, Asia, North America
            The match of regions is achieved by .str.contains method, thus 'Europe would account for both Eastern and Western Europe.'
        weights: list
            Corresponding weights for the operating segments. Must sum up to one.
            
        Raises:
        ----------------
        AssertionError if sum(weights) != 1
        AssertionError if length of regions and weights arguments are not equal
        '''
        
        assert sum(weights) == 1, f'Weights do not sum up to one: {weights}'
        assert len(regions) == len(weights), f'Lengths of regions and weights arguments not equal'
        
        tab = self.damo.getCountryRP()
        self.country_defaultspread = sum([tab.loc[tab['Region'].str.contains(r),'Default Spread'].mean()*w for r,w in zip(regions, weights)])
        return self.country_defaultspread
    
    def DefaultSpread(self, spread = 0, show_synthetic_rating = False, large_cap = True):
        '''
        Returns user defined default spread and assigns it to the `default_spread` attribute.
        
        Args:
        ----------------
        spread: int, float, default 0
            Optional; DefaultSpread to be assigned
        show_synthetic_rating: bool, default False
            if True method returns Damodaran's table for determining default spread using synthetic rating
        large_cap: bool, default True
            if False table for small_cap companies' default spread is shown.
        '''
        ###ISSUE: user-input or self.calculated??
        if show_synthetic_rating is True:
            return self.damo.getDefaultSpread(large_cap)
        elif show_synthetic_rating is False:
            self.default_spread = spread
        
    def CostofDebt(self, rf = 0, bond_yield = None, method = "book"):
        '''
        Returns cost of debt and assigns the variable to the `kd` attribute.
        
        Args:
        ----------------
        rf: int, float, default 0
            Used in calculation of cost of debt via synthetic method
        bond_yield: int, float, default None
            Used in calculation via bond method.
        method: str, ['book','synthetic','bond'], default 'book'
            Three viable methods for calculation of the cost of debt.
            
        Raises:
        ----------------
        AssertionError if method is not one of book, synthetic, bond
        ValueError if method is 'synthetic' but one of default_spread or country_defaultspread is not computed yet
        ValueError if method is 'bond' but no bond_yield has been specified
            
        '''
        
        assert method in ['book','synthetic','bond'], f'Unavailable method: {method}. Viable methods are book, synthetic and bond'
        
        if method == "book":
            self.kd = (-self.fin.loc['Interest Expense']/self.bal.loc['Financing Liabilities'])[0]
            
        elif method == "synthetic":
            ##Check whether necessary attributes are set already
            if any([self.default_spread, self.country_defaultspread]) is False:
                raise ValueError(f'One of Default Spread, Country Default Spread attributes not defined')
            else:
                self.kd = rf/100 + self.default_spread + self.country_defaultspread
                
        elif method == "bond":
            if bond_yield is None:
                raise ValueError(f'Bond Yield not specified: {bond_yield}')
            else:
                self.default_spread = bond_yield/100 - rf/100
                self.kd = bond_yield/100
        
        return self.kd
    ###------------------------------------------------------------------------------------------------------------------------------------------------
    def WACC(self, tax):
        '''
        Returns weighted average cost of capital and assigns the value to the `wacc` attribute. Also assigns `ndweight` attribute as Net Debt over the sum 
        of net debt and equity value.
        
        Args:
        ----------------
        tax: int, float
            Tax rate used to calculate WACC. Does not have default value.
        '''
        self.ndweight = self.nd/(self.nd + self.mktcap)
        self.wacc = (1-self.ndweight)*self.ke + self.ndweight*(1-tax)*self.kd
        return self.wacc
    ###------------------------------------------------------------------------------------------------------------------------------------------------
    def Forecast(self, gns, pm, xat, wct, tax, **kwargs):
        '''
        Returns explicitly forecasted financials including Net Sales, EBIT, NOPLAT, NTA, WC, IC, change in IC and ROIC in pandas DataFrame.
        
        Args:
        ----------------
        gns: list
            List of net sales (revenue) growth. Needs to be of length +1 compared to others.
        pm: list
            List of operating profit margins
        xat: list
            List of net long term asset turnovers
        wct: list
            List of operating working capital turnover
        tax: list
            List of tax rates
        
        Raises:
        ----------------
        AssertionError if the dimensions of pm, xat, wct, and tax are not equal to the dimension of gns - 1, i.e. gns needs to be forecasted for t+1 periods.
        '''
        #Assert dimensions of thre forecasted ratio
        assert all(len(arg) == len(gns)-1 for arg in [pm, xat, wct, tax]), 'Wrong dimensions of the inputs. The dimensions of pm, xat, wct and tax\
        must be equal and gns must be of t+1 dimension where t is length of the forecasted period.'
        
        gns = [1 + g for g in gns]
        mult_gns = [math.prod(gns[:i]) for i in range(1, len(gns)+1)]
        NetSales = [self.fin.loc['Total Revenue'][0] * g for g in mult_gns]
        EBIT = [i * j for i, j in zip(NetSales[:-1], pm)]
        NOPLAT = [i * (1-j) for i, j in zip(EBIT, tax)]
        NTA = [i / j for i, j in zip(NetSales[1:], xat)]
        WC = [i / j for i, j in zip(NetSales[1:], wct)]
        IC = [i + j for i, j in zip(NTA, WC)]
        delta_IC = [IC[0] - self.bal.loc['Invested Capital'][0]]
        delta_IC.extend([subitem for subitem in [IC[i+1] - IC[i] for i in range(0,len(IC)-1)]])
        FCF = [i - j for i, j in zip(NOPLAT, delta_IC)]
        ROIC = [i / j for i, j in zip(NOPLAT, IC)]
        
        self.forecast = pd.DataFrame([NetSales, EBIT, NOPLAT, NTA, WC, IC, delta_IC, FCF, ROIC], 
                                     index = ['Net Sales', 'EBIT','NOPLAT', 'Net Fixed Assets',
                                     'Working Capital', 'Invested Capital', 'Change in IC',
                                     'Free Cash Flow', 'ROIC'])
    def getForecastTable(self):
        
        '''
        Returns comprehensive forecasted table together with historical coutnerparts for more insight
        
        Raises
        ----------------
        AssertionError if forecast has not been made (forecast attribute is None)
        
        '''
        
        assert self.forecast is not None, f'Forecast has not been made yet. .forecast attribute is None!'
        
        table = pd.concat([self.fin.loc[['Total Revenue', 'EBIT', 'NOPLAT']],self.bal.loc[['NTA', 'Operating Working Capital','Invested Capital']]],
                          axis = 0)
        table.index = ['Net Sales', 'EBIT', 'NOPLAT', 'Net Fixed Assets', 'Working Capital', 'Invested Capital']
        delta_IC = self.bal.loc['Invested Capital'].diff(-1)
        fcf = self.fin.loc['NOPLAT'] - delta_IC
        table.loc['Change in IC'] = delta_IC
        table.loc['Free Cash Flow'] = fcf
        table.loc['ROIC'] = self.fin_analysis.loc['ROIC']
        table.columns = self.fin.columns.year
        ###Reverse Order of columns
        table = table.iloc[:, ::-1]
        forecast_table = pd.concat([table, self.forecast], axis = 1, keys = ['Historical','Forecasted'])
        cols = [self.fin.columns.year[0] + i for i in range(1, self.forecast.shape[1]+1)]
        cols.extend(self.fin.iloc[:, ::-1].columns.year)
        forecast_table.columns.set_levels(cols, level = 1, inplace = True)
        return forecast_table
    
    def HorizonValue(self, gns, roic, ronic, wacc_cont = None):
        '''
        Returns continuing (horizon) value based on user inputs.
        
        Args:
        ----------------
        gns: int, float
            Continuing growth rate (growth in net sales). Should be the growth rate (gns) used in explicit forecast.
        roic: int, float
            Return on invested capital. Typically, last ROIC from the explicit forecast.
        ronic: int, float
            Return on new invested capital. Expected ROIC for the continuing operations, respectively capital.
        wacc_cont: int, float, default None
            Weighted average cost of capital expected for the continuing value. If left None the wacc stored in the
            wacc attribute is used, i.e. the same used for the explicit forecast.
            
        Raises:
        ----------------
        ZeroDivisionError:
            if gns == wacc_cont (null denumerator)
        Warning:
            if gns is larger than wacc_cont or wacc, produces negative horizon value
        '''
        
        if wacc_cont is None:
            if not self.wacc:
                raise ValueError(f'WACC has not been set')
            self.horizon_wacc = self.wacc
        else:
            self.horizon_wacc = wacc_cont
        if gns > self.horizon_wacc:
            warnings.warn(f'WARNING: gns > wacc will result in negative value')
        elif gns == self.horizon_wacc:
            raise ZeroDivisionError(f'Dividing by zero, gns == wacc')
        self.hv = (self.forecast.loc['Invested Capital'][self.forecast.shape[1]-2]*roic*(1-gns/ronic))/(self.horizon_wacc-gns)
        return self.hv
    
    def getPV(self, out = "eqval"):
        
        '''
        Returns the present value of the cash flow, i.e. estimate value of the equity based on explicit forecast
        and horizon value previously computed. Assigns the value to the attribute estimate.
        
        Args:
        ----------------
        wacc_cont: int, float, default None
            Weighted average cost of capital rate used to discount the horizon value. If left None same wacc rate as for explicit
            forecast is used (stored in .wacc attribute)
        out: {'eqval' ,'pps'}, default 'eqval'
            If 'eqval' estimated value of equity is returned (market capitalisation). 'pps' returns price per share.
        
        Raises:
        ----------------
        AssertionError if 'out' argument is not in ['eqval', 'pps']
        '''
        
        assert out in ['eqval', 'pps'], 'Invalid output, out must be eqval (Equity/Market Value) or pps (Price per Share)'

        if not self.wacc:
            raise ValueError(f'WACC has not been set. Computation of PV of FCF from the forecast requires WACC')
        
        pv = sum([self.forecast.loc['Free Cash Flow'][i]/(1+self.wacc)**(i+1) for i in range(self.forecast.shape[1]-1)])
        pv_cont = self.hv/(1+ self.horizon_wacc)**(self.forecast.shape[1]-1)
        self.estimate = pv + pv_cont - self.nd
        if out == "eqval":
            return self.estimate
        elif out == 'pps':
            try:
                return self.estimate/self.yfticker.info['impliedSharesOutstanding']
            except (KeyError, TypeError, ZeroDivisionError):
                return self.estimate/self.yfticker.info['sharesOutstanding']

    ###------------------------------------------------------------------------------------------------------------------------------------------------

    def SensitivityAnalysis(self, gns, ronic, roic = None, range1 = None, range2 = None, plot = False, cmap = 'spring'):
        
        '''
        Returns two-input sensitivity analysis table or plot. Sensitivity analysis based on varying ROIC and WACC values.
        
        Table returns values in millions. If the WACC rates differ in the explicit forecast value and horizon value,
        triple sensitivity table is given with maintaining constant spread between WACC and horizon/normalised WACC (nWACC).
        If the WACC rates differ plot option is not available.
        
        Args:
        ----------------
        gns: int, float
            Growth rate used in the computation of horizon value.
        ronic: int, float
            Return on new invested capital used for computaiton of horizon value.
        range1: list, default None
            Range of ROIC inputs for the sensitivity analysis. If left None standard range around the last ROIC from explicit forecast with
            (-10%, + 20%) spread with 2% steps inbetween is used.
        range2: list, default None
            Range of WACC inputs for the sensitivity analysis. If left None standard range around the WACC rate stored in .wacc attribute is used.
            Spread of 3% with .5% steps in between.
        plot: bool, default False
            If plot is False returns sensitivity table, if True returns sensitivity plot.
        cmap: matplotlib colormap, default 'spring'
            Colormap for the sensitivty plot. By default 'spring'.
        
        Raises
        ----------------
        ValueError if plot is True when explicit forecast WACC (.wacc attribute) and horizon value WACC (.horizon_wacc attribute) are not equal.
        '''
        if self.wacc != self.horizon_wacc and plot is True:
            raise ValueError(f'Plot option is not available for triple variable sensitivity table.')

        if self.wacc == self.horizon_wacc:
            roic = self.forecast.loc['ROIC'][self.forecast.shape[1]-2] if roic is None else roic
            ic = self.forecast.loc['Invested Capital'][self.forecast.shape[1]-2]
            range1 = np.arange(self.wacc - .03, self.wacc + .03, .005) if range1 is None else range1
            range2 = np.arange(roic - .10, roic + .20,.02) if range2 is None else range2

            forecast_range = [sum([self.forecast.loc['Free Cash Flow'][i]/(1+wacc)**(i+1) for i in range(self.forecast.shape[1]-1)]) for wacc in range1]
            horizon_range = [[((ic*r*(1-gns/ronic))/(w-gns))/(1+w)**(self.forecast.shape[1]-1) for r in range2] for w in range1]
            pv_range = [i + j - self.nd  for i, j in zip(horizon_range, forecast_range)]
            
            tab = pd.DataFrame(pv_range, index = range1, columns = range2)
            tab.index.name = 'WACC'
            tab.columns.name = 'ROIC'

            if plot is False:
                tab.index = ['{:.2%}'.format(x) for x in range1]
                tab.columns = ['{:.2%}'.format(x) for x in range2]
                tab.index.name = 'WACC'
                tab.columns.name = 'ROIC'
                
                return tab.style.format(lambda x:'{:,.0f}'.format(x/10**6)).background_gradient(cmap = cmap)
            
            elif plot is True:
                tab = pd.melt(tab, value_name = 'Value', ignore_index = False).reset_index()
                r1_diff = np.diff(range1).mean()
                r2_diff = np.diff(range2).mean()
                fig, ax = plt.subplots(figsize = (12,8))
                s = ax.scatter(x = 'WACC', y = 'ROIC', c = 'Value',
                               data = tab,
                               cmap = cmap)
                ax.text(x = self.wacc, y = roic, s = self.ticker, 
                        ha = 'center', va = 'center',
                        fontsize = 12, fontfamily = 'Helvetica')
                r = patches.Rectangle((self.wacc - 2*r1_diff, roic - 3*r2_diff),
                                      width = 4*r1_diff,
                                      height = 6*r2_diff,
                                      alpha = .1, color = 'lightcoral')
                ax.add_patch(r)
                ax.set_xlabel('WACC')
                ax.set_ylabel('ROIC')
                fig.colorbar(s)
                plt.show()
        else:
            roic = self.forecast.loc['ROIC'][self.forecast.shape[1]-2] if roic is None else roic
            ic = self.forecast['Invested Capital'][self.forecast.shape[1]-2]
            range1 = np.arange(self.wacc - .03, self.wacc + .03, .005) if range1 is None else range1
            range2 = np.arange(roic - .10, roic + .20,.02) if range2 is None else range2
            range3 = range1 + (self.horizon_wacc - self.wacc)

            forecast_range = [sum([self.forecast.loc['Free Cash Flow'][i]/(1+wacc)**(i+1) for i in range(self.forecast.shape[1]-1)]) for wacc in range1]
            horizon_range = [[((ic*r*(1-gns/ronic))/(w-gns))/(1+w)**(self.forecast.shape[1]-1) for r in range2] for w in range3]
            pv_range = [i + j - self.nd  for i, j in zip(horizon_range, forecast_range)]
            
            multi_cols = pd.MultiIndex.from_arrays([range3, range1])
            
            tab = pd.DataFrame(pv_range, index = multi_cols, columns = range2)
            tab.index.set_names(['nWACC','WACC'], inplace = True)
            tab.columns.name = 'ROIC'
            tab.columns = ['{:.2%}'.format(x) for x in range2]
            tab.index = tab.index.set_levels(['{:.2%}'.format(x) for x in range3], level = 0)
            tab.index  = tab.index.set_levels(['{:.2%}'.format(x) for x in range1], level = 1)

            return tab.style.format(lambda x:'{:,.0f}'.format(x/10**6)).background_gradient(cmap = cmap)
        
    
    
    def Benchmark(self, tickers = None, period = ''):
        
        ret = (1 + self.yfticker.history(period = period)['Close'].pct_change()).cumprod()
        
        fig, ax = plt.subplots(figsize = (12, 8))
        
        if not tickers:
            
            sector_index = SPDR_ETF.get(self.sector)
            sp_ret = sp_ret = (1 + yf.Ticker('^GSPC').history(period = period)['Close'].pct_change()).cumprod()
            nasdaq_ret = (1 + yf.Ticker('^IXIC').history(period = period)['Close'].pct_change()).cumprod()
            sector_ret = (1 + yf.Ticker(sector_index).history(period = period)['Close'].pct_change()).cumprod()
            
            ax.plot(ret, color = 'royalblue', label = self.ticker, lw = .8)
            ax.plot(sp_ret, color = 'orangered', label = 'S&P 500', lw = .8)
            ax.plot(nasdaq_ret, color = 'darkorchid', label  = 'NASDAQ', lw = .8)
            ax.plot(sector_ret, color = 'silver', label = sector_index, lw = .8)
            
            ax.yaxis.set_major_formatter(lambda x,p: '${:.2f}'.format(x))
            ax.set_title(f'Index Benchmarking ({self.ticker})', font = 'Avenir', fontsize = 14)

            ax.grid(axis = 'x')
            ax.legend(fontsize = 12)

            plt.show()
            
        if tickers:
            
            t = yf.Tickers(" ".join(tickers).upper())
            prices = t.history(period = period)['Close']
            cumulative_return = (1 + prices.pct_change()).cumprod()
            
            ax.plot(ret, label = self.ticker)
            
            for stock in [t.upper() for t in tickers]:
                ax.plot(cumulative_return[stock], label = stock, lw = .7)
            
            ax.yaxis.set_major_formatter(lambda x,p: '${:.2f}'.format(x))
            ax.grid(axis = 'x')
            ax.set_title(f'Benchmarking ({self.ticker})', font = 'Avenir', fontsize = 14)
            
            plt.show()
            
            
            