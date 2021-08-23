from dcfval import *


#---------------------------------------------------------------------------------------------------------------------------------
def getBook_Multiples(ticker, sesh = None):
    
    '''
    Using reconstructed balance sheets and financials returns price and enterprise value multiples.
    
    Mktcap and Enterprise Value used according to the date of the last reported quarter.
    
    Args:
    ----------------
    ticker: str
        Company's ticker on Yahoo Finance
    sesh: requests.Session() object
        requests.Session
    '''
    
    bal = balratios(ticker, sesh)
    fin = finratios(ticker, sesh)
    mktcap = getMktcap(ticker, date = fin.columns[0])
    
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

def getMultiples(ticker):
    '''
    Returns current (Yahoo Finance) multiples from the yf.Ticker module in pandas DataFrame.
    
    Outputs are P/BV, P/S, EV/Net Sales, EV/EBITDA, PEG and P/E.
    
    Mktcap and Enterprise Value are based on current prices.
    
    
    Args:
    ----------------
    ticker: str
        Company's ticker on Yahoo Finance
    '''
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

def getIndustryMultiples(industry, additional = None):
    '''
    Returns selected industry multiples by initialising repsective IndustryData classes and subsequently parsing all tables for relevant multiples.
    
    Returns Trailing PE, PEG Ratio, Price/Sales, EV/Sales, PBV, EV/EBITDA, EV/EBIT (predetermined set, can be extended)
    
    Args:
    ----------------
    industry: str
        Industry classification according to Damodaran
    additional: str, list, default None
        Additional multiples to be retrieved if specified. Incorrect names are skipped.
        
    Raises:
    ----------------
    ValueError if industry is not found in the tables.
    '''
    m_id = ['Current PE', 'Trailing PE','Forward PE', 'PEG Ratio', 'Price/Sales', 'EV/Sales', 'PBV','EV/EBITDA','EV/EBIT','EV/EBIT (1-t)']
    tables = ['pedata','psdata','pbvdata','vebitda']
    multiples = {}
    
    if additional:
        m_id.append(additional)
    
    for t in tqdm(tables, colour = 'green', leave = False):
        time.sleep(.5)
        ind = IndustryData(t)
        tab = ind.getTable()
        ##Handle KeyError of industry before-hand
        if industry not in tab.index:
            raise ValueError(f"Incorrect Industry: {industry}")
        
        for m in m_id:
            try:
                multiples[m] = tab.loc[industry, m]
            except KeyError:
                continue
            
    return pd.DataFrame(multiples.values(), index = multiples.keys(), columns = [industry])

def HighlightDf(df, max_col = 'lightgreen', min_col = 'lightcoral'):
    
    '''
    Styler function highlighting Min Max in pandas DataFrames
    '''
    return df.style.highlight_max(axis=1, color = max_col).highlight_min(axis = 1, color = min_col)

#==================================================================================================================================
#==================================================================================================================================
#==================================================================================================================================
#=================================================##MULTIPLES VALUATION CLASS##====================================================

class Multiples(Valuation):
    '''
    Multiples Valuation Class
    Class inherits from Valuation class. Class represents company with methods children methods to return book and current multiples.
    
    Attributes:
    ----------------
    multiples: default None
        To be assigned book multiples based on 12 months trailing data from the last reported quarter.
    current_mult: default None
        To be assigned multiples based on current values of market capitalisation.
    
    Methods:
    ----------------
    get_Multiples(current = False)
        if current = False, returns book multiples based on last reported quarter financial results
        if current = True, returns multiples based on current market price/capitalisation
    peer_multiples(tickers, current = False)
        Returns multiples of specified peer tickers/companies
    estimate_multiples()
        Using peer_mults attribute returns min-avg-max ranges of of value based on peer multiples.
    plot_Multiples(thickness = .5, cmap = 'Accent')
        Plots company and peers' multiples on horizontal bar chart.
    visual_mult(thickness = .5)
        Returns plot of estimated value of equity and enterprise value based on peer's min, avg, max ranges of multiples.
    '''
    
    def __init__(self, ticker, sesh = None):
        '''
        Calls parent's (Valuation) constructor and initialsies two empty attributes 'multiples', 'current_mult' and 'mult_est' with None default values.
        '''
        super().__init__(ticker, sesh)
        self.multiples = None
        self.current_mult = None
        self.peer_mult_current = None
        self.mult_est = None
        self.industry_multiples = None
        
    
    def get_Multiples(self, current = False):
        '''
        Returns book or current multiples (P/BV, P/S, P/E, EV/SALES, EV/EBIT)
        
        Args:
        ----------------
        current: bool, default False
            if False, method returns book multiples based on 12 months trailing data from the last quarterly financial report.
            if True, method returns multiples based on current valuations, respectively price
        '''
        
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
            
    def peer_multiples(self, tickers, current = False):
        '''
        Returns multiples of specified peers (tickers). Book multiples if current is False, else current multiples are returned.
        Assigns the peer multiples to the peer_mult attribute (only for if current = False)
        
        Args:
        ----------------
        tickers: list
            list of Yahoo Finance tickers
        current: bool, default False
            if False, method returns book multiples based on 12 months trailing data from the last quarterly financial report.
            if True, method returns multiples based on current valuations, respectively price
        '''
        
        if current is False:
            mult = [getBook_Multiples(comp, sesh = self.sesh) for comp in tickers]

            if self.multiples is None:
                self.get_Multiples()

            mult.append(self.multiples)
            self.peer_mult = pd.concat(mult, axis = 1)
            
            return self.peer_mult
        
        if current is True:
            if self.current_mult is None:
                self.get_Multiples(current = True)
                
            t = [yf.Ticker(comp, session = self.sesh) for comp in tickers]
            m = {i : getMultiples(j) for i, j in zip(tickers, t)}
            m[self.ticker] = self.current_mult
            self.peer_mult_current = pd.concat(m, axis = 1).droplevel(level = 1, axis = 1)
            return self.peer_mult_current
    
    def estimate_multiples(self):
        '''
        Returns Min, Avg, Max ranges of peer multiples, previously defined. Assigns the pandas DataFrame to the mult_est attribute.
        '''
        valuation = [self.bal.loc['EEQ'][0], self.fin.loc['Total Revenue'][0], self.fin.loc['Net Income'][0],
                     self.fin.loc['Total Revenue'][0], self.fin.loc['EBIT'][0]]
        min_val = self.peer_mult[:5].apply(np.min,1)*valuation
        avg_val = self.peer_mult[:5].apply(np.mean,1)*valuation
        max_val = self.peer_mult[:5].apply(np.max,1)*valuation
        
        self.mult_est = pd.concat([min_val, avg_val, max_val], axis = 1, keys = ['Min','Avg','Max'])
        return self.mult_est
    
    def plot_Multiples(self, thickness = .5, cmap = "Accent"):
        '''
        Plots company and peers' multiples on a horizontal bar chart.
        
        Args:
        ----------------
        thickness: int, float, default 0.5
            Controls the height of the respective bars
        cmap: matplotlib colormap code, default 'Accent'
            matplotlib colormap
        '''
        ##Use convenience plot method
        self.peer_mult[:5].plot.barh(figsize = (12,6), colormap = cmap)
        plt.legend(fontsize = 14, title = "Companies", title_fontsize = 14)
        plt.yticks(font="Helvetica", fontweight = "heavy", fontsize = 12)
        plt.grid(True, alpha = .75, linewidth = .75, ls = "--")
    
    def visual_mult(self, thickness = .5):
        '''
        Plots the estimate of value of equity and enterprise value based on the minimum, average and maximum ranges.
        
        Method plots BOOK multiples from min to max range of the peer multiples with averages denoted.
        
        Args:
        ----------------
        thickness: int, float, default 0.5
            Controls the heigh of the respective bars
        '''
        ###Works only with BOOK multiples!!!!
        ###Does not assign attribute of a class --> TO RESOLVE: assigning plot to a class attribute??
        fig, ax = plt.subplots(figsize = (12,6))
        ax.barh(self.mult_est.index, self.mult_est['Min'], label = 'Min', height = thickness)
        ax.barh(self.mult_est.index, (self.mult_est['Max']-self.mult_est['Min']), label = 'Max', height = thickness, left = self.mult_est['Min'])
        
        ###Plot Averages
        for i,bar in zip(range(len(self.mult_est.index)),ax.patches):
            ax.plot((self.mult_est['Avg'][i],self.mult_est['Avg'][i]), (bar.get_y()-.1,bar.get_y()+thickness+.1),"k-")
            ax.annotate("AVG", (self.mult_est['Avg'][i],bar.get_y()), xytext = (5,10), textcoords = "offset points", rotation = 90, fontsize = 10)
        
        ###Plot MKTCAP and Enterprise Value of the MAIN ticker
        ##Mktcap for Price multiples: pbv, ps, pe
        ax.plot((self.multiples.loc['MktCap'], self.multiples.loc['MktCap']),
                (ax.patches[0].get_y()-.15, ax.patches[2].get_y()+thickness+.15), "b--x", linewidth = .8, label = "MktCap")
        ##Enterprise Value for EV multiples: ev_sales,ev_ebit
        ax.plot((self.multiples.loc['EV'], self.multiples.loc['EV']),
                (ax.patches[3].get_y()-.15, ax.patches[4].get_y()+thickness+.15), "m--o", linewidth = .8, label = "EV")
        
        ###Cosmetics
        ax.set_title(f"Multiple Valuation Estimates ({self.ticker})", font = "Avenir", fontsize = 14)
        ax.set_xlabel("Enterprise Value/Market Cap")
        ax.xaxis.set_major_formatter(lambda x, p: '${:,.0f}m'.format(int(x)/10**6))
        ax.legend(fontsize = 10)
        ax.grid(True, linewidth = .75, alpha = .7, ls = "--")
        
    
    def IndustryPerformance(self, industry):
        
        if self.industry_multiples is None:
            self.industry_multiples = getIndustryMultiples(industry)
        if self.multiples is None:
            self.get_Multiples(current = False)
        
        self.industry_multiples.index = ['P/E', 'Trailing PE', 'Forward PE', 'PEG', 'P/S', 'EV/Sales', 'P/BV', 'EV/EBITDA', 'EV/EBIT', 'EV/EBIT (1-t)']
        df = pd.concat([self.industry_multiples, self.multiples], axis = 1)
        df.fillna(np.nan, inplace = True)
        df = df.applymap(pd.to_numeric)
        df['diff'] = df[self.ticker] - df[industry]
        df['pct_diff'] = df['diff']/df[industry]
        df['pos_neg'] = ['tomato' if i < 0 else 'springgreen' for i in df['pct_diff']]
        df = df.loc[df['pct_diff'].notna()]
        
        #df.loc[df['pct_diff'].notna(),'pct_diff'].plot.bar(figsize = (12,8), color = ['lightcoral' if m is True else 'lightgreen' for m in df['pos_neg']], width = .5)
        fig, ax = plt.subplots(figsize = (12,8))
        
        cond = df['pct_diff'].notna()
        ax.barh(df.index, df['pct_diff'], color = df['pos_neg'], height = .5)
        ax.set_title(f'Industry Multiples Comparison ({self.ticker})', font = 'Avenir', fontsize = 14)
        ax.set_xlabel('Under/Over Performance Industry (%)')
        ax.xaxis.set_major_formatter(lambda x, p: '{:.0%}'.format(x))
        
        for i, bar in zip(df.index, ax.patches):
            val = df.loc[i, 'pct_diff']
            lab = '{:,.2%}'.format(val)
            x_cord = bar.get_width()
            y_cord = bar.get_y() + bar.get_height()/2
            delta = 5
            ha = 'left'
            
            if val < 0:
                delta *= -1
                ha = 'right'
                
            ax.annotate(lab, 
                        (x_cord, y_cord),
                        xytext = (delta, 0),
                        textcoords = 'offset points',
                        va = 'center',
                        ha = ha,
                        font = 'Avenir',
                        fontsize = 13)
        
        ax.margins(x = .2)
        ax.grid(axis = 'x')
        plt.show()
        
        