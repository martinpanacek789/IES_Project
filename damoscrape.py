import pandas as pd
import numpy as np
import requests
import re
import time
from tqdm import tqdm
from itertools import compress
from bs4 import BeautifulSoup

###Helper Function
def clean_idx(df):
    '''
    Helper function to clean indices of parse tables. Returns pandas Dataframe with cleaned indices/columns
    
    Args:
    ----------------
    df: pandas DataFrame
    '''
    df.index = df.index.str.replace(r"\(","").str.replace(r"\)","")
    df.index = df.index.str.replace("\s\s+" ," ")
    df.columns = df.columns.str.replace("\s\s+" ," ")
    return df


#==================================================================================================================================
#==================================================================================================================================
#==================================================================================================================================
#=================================================##DAMODARAN DATA CLASS##=========================================================

class DamoData:
    '''
    Class for scraping Damodaran's datafiles
    
    
    Attributes
    ----------------
        base_url: str
            Base URL link to Damodaran's web files
        core_url: str
            Core URL link to Damodaran's home web
        industry_xls: str
            URL link to the xls file containing information on industry classificaiton of individual firms
        soup: BeautifulSoup object
            BeautifulSoup object of the base_url link
        hrefs: list
            href characteristics of <a> html tags from the soup object
        core_tables: dict
            Empty dictionary to be assigned all core tables if method getCoreTables() is called
        htms: default None
            to be assigned .htm links
        ind_group: default None
            to be assigned pandas DataFrame contianing data on the industry groupings
            
    Methods
    ----------------
        getSoup(url)
            returns BeautifulSoup object from the URL link
        getCoreTables()
            Returns all CORE (36) tables indexed by first column and row and assigns them to the core_tables attribute in dict form
        getLinks()
            Returns a list of all .htm links from the base URL link and assigns the list to the .htms attribute
        getPngs()
            Returns a list of all .png links.
        getXls()
            Returns a list of all .xls links.
        listNames()
            List names of all the files with all [dot] and [underscore] extensions removed.
        lookupCompany(name = "", ticker = "", industry = "", country = "", **kwargs)
            Returns information on the industry grouping of companies filtered by name, ticker, industry, and/or country.
            NOTE: First call takes approx. 30s due to requesting large XLS file
        getCountryRP()
            Returns pandas DataFrame containing data on country default spreads.
        getDefaultSpread(large_cap = True)
            Returns multiindexed pandas DataFrame with data on default spreads based on synthetic credit ratings.
        getERP(sort = 'hist')
            Returns pandas DataFrame with data on historical equity risk premium or historical implied equity risk premium.
    '''
    
    def __init__(self):
        self.base_url = "http://people.stern.nyu.edu/adamodar/New_Home_Page/datafile/"
        self.core_url = 'http://people.stern.nyu.edu/adamodar/New_Home_Page/'
        self.industry_xls = "http://www.stern.nyu.edu/~adamodar/pc/datasets/indname.xls?raw=true"
        self.soup = self.getSoup(self.base_url)
        self.hrefs = [l['href'] for l in self.soup.find_all('a')]
        self.core_tables = {}
        self.htms = None
        self.ind_group = None


    def getCoreTables(self):
        '''
        Returns a dictionary containing all 36 core tables from Damodaran's dataset.
        
        All pandas DataFrames are indexed by first row and column.
        Note: Only first dataframes from the html link is retrieved. Dataframes need a lot of prettyfying.
        
        If link is not accessible it is skipped in the loop and message is printed.
        '''
        soup = self.getSoup(self.core_url + 'datacurrent.html')
        trows = soup.find('table',{'border':2, 'cellpadding': 0, 'cellspacing':2}).find_all('tr')
        row_col = [row.find_all('td') for row in trows]
        alinks = [row.find_all('a') for row in trows]
        flat_alinks = [item for sublist in alinks for item in sublist]
        hrefs = []
        ##Not all a-tags have href attribute weirdly enough.
        for a in flat_alinks:
            try:
                hrefs.append(a['href'])
            except KeyError:
                continue
        ###Get only .html links and parse them to retrieved clean name later for dictionary assigning
        html_hrefs = list(compress(hrefs, [h.endswith('.html') for h in hrefs]))
        clean_hrefs = [re.sub('.*/|\\..*','',href) for href in html_hrefs]
        ###Retrieve tables from all href leading links, append to empty tables list
        ##Tables not valid for custom index/columns cleaning (int, float-like indices) are append in raw form.
        tables = []
        for url in tqdm(html_hrefs, colour = 'green', leave = False):
            time.sleep(.5)
            try:
                r = requests.get(self.core_url + url)
            except requests.HTTPError as httpe:
                print(f'Request failed for {self.core_url + url}',httpe)
                continue
            tab = pd.read_html(r.content, index_col = 0, header = 0)[0]
            try:
                tables.append(clean_idx(tab))
            except Exception:
                tables.append(tab)

        self.core_tables = {name:tab for name, tab in zip(clean_hrefs, tables)}
        
        return self.core_tables
            
        
    def getSoup(self, url):
        '''
        Returns BeautifulSoup object from url
        '''
        r = requests.get(url)
        r.raise_for_status()
        return BeautifulSoup(r.text,"lxml")
        
    def getLinks(self):
        '''
        Returns all .htm links. Assigns the list of htm links to the htms attribute
        '''
        self.htms = list(compress(self.hrefs,[ref.endswith("htm") for ref in self.hrefs]))
        return [self.base_url + link for link in self.htms]
    
    def getPngs(self):
        '''
        Returns all links to .png files.
        '''
        png_links = list(compress(self.hrefs,[ref.endswith("png") for ref in self.hrefs]))
        return [self.base_url + link for link in png_links]
    
    def getXls(self):
        '''
        Returns all links to .xls files.
        '''
        xls_links = list(compress(self.hrefs,[ref.endswith("xls") for ref in self.hrefs]))
        return [self.base_url + link for link in xls_links]
    
    def listNames(self):
        '''
        List names of the files in a concise way, i.e. removing [dot] and [underscore] extensions.
        '''
        return [re.sub("\\..*|_.*","", ref) for ref in self.hrefs]
    
    def lookupCompany(self, name = "", ticker = "", industry = "", country = "", **kwargs):
        '''
        Returns information about the industry classification based on defined filters (name, ticker, industry, country).
        The whole retrieved dataframe of industry classification is then stored in the ind_group attribute.
        
        First call of this method attempts to read .xls file stored at URL stored in .industry_xls attribute, takes approx 30s.
        The pandas DataFrame is then stored in the .ind_group attribute, which is then searched through to find the match to user inputs.
        Subsequent calls of the method only go through the dataframe stored in the .ind_group attribute and skip the pd.read_excel attempt.
        
        Args
        ----------------
        name: str, default ""
            Name of the company
        ticker: str, default ""
            Company's ticker (based on the stock exchange they may differ)
        industry: str, default ""
            Industry classification according to Damodaran
        country: str, default ""
            Country where the company is based.
            
        Raises
        ----------------
        HTTPError in case of unsuccessful initial request for the xls file
        '''
        if self.ind_group is None:
            r = requests.get(self.industry_xls)
            r.raise_for_status()
            self.ind_group = pd.read_excel(r.content, sheet_name = "Global alphabetical")
            #only the first call of the method initiliases df --> takes long time
            #all subsequent calls will use the industry grouping dataframe initialised in the first call
        name_cond = self.ind_group['Company Name'].str.contains(name, na = False)
        ticker_cond = self.ind_group['Exchange:Ticker'].str.contains('.*:'+ticker, na = False) #does not go through Exchanges' names
        ind_cond = self.ind_group['Industry Group'].str.contains(industry, na = False)
        country_cond = self.ind_group['Country'].str.contains(country, na = False)
        cond = [name_cond, ticker_cond, ind_cond, country_cond]
    
        return self.ind_group[np.logical_and.reduce(cond)]
    
    def getCountryRP(self):
        '''
        Parses through the country risk premium table and returns pandas DataFrame.
        '''
        url = self.base_url + 'ctryprem.htm'
        r = requests.get(url)
        r.raise_for_status()
        table = pd.read_html(r.text, flavor = 'bs4', header = 0, index_col = 0)[0]
        table = table.applymap(lambda x: float(x.strip("%"))/100 if isinstance(x,str) and "%" in str(x) else x)
        ###There are two tables in the link which are glued together by pd.read_html. Second table differs in the structure of the columns
        ##Second column of the second table is numeric instead of str as in first table --> Filter rule distinguishing between two tables
        table = table.loc[[type(i) is str for i in table.iloc[:,1]]][:-2]
        table = clean_idx(table)
        table.columns = ['Region','Moody rating','Default Spread','ERP','Country RP']
        table['Region'] = table['Region'].str.replace('\s\s+'," ")
        return table
    
    def getDefaultSpread(self, large_cap = True):
        '''
        Prases through the default spread table and returns multiindexed pandas DataFrame
        
        
        Args
        ----------------
        large_cap: bool, default True
            if True returns the table for large cap companies (>$5bn)
        '''
        url = self.base_url + 'ratings.htm'
        r = requests.get(url)
        r.raise_for_status()
        table = pd.read_html(r.text, flavor = 'bs4', header = [0,1])[0]
        
        if large_cap is True:
            table.drop(index = np.arange(16, table.shape[0]), columns = table.columns[1], inplace = True)
            table.columns = pd.MultiIndex.from_product([["LargeCap (>$5bn)"],['ICR>','Rating','Spread']])
        elif large_cap is False:
            table.drop(index = np.arange(0,19), columns = table.columns[1], inplace = True)
            table.columns = pd.MultiIndex.from_product([["SmallCap (<$5bn)"],['ICR>','Rating','Spread']])
        else:
            raise ValueError(f'True/False value accepted: {large_cap}')
        
        return table
    
    def getERP(self, sort = 'hist'):
        '''
        Prases through either historical premium table or historical implied equity premium table.
        
        Args
        ----------------
        sort: str, ['hist', 'impl'], default 'hist'
            'hist' returns historical equity risk premium table. 'impl' returns historical IMPLIED ERP table.
        '''
        
        assert sort in ['hist','impl'], f'Only \'hist\' for Historical and \'impl\' for Implied ERP eligible.: {sort}'
        
        if sort == 'hist':
            url = self.base_url + 'histretSP.htm'
            r = requests.get(url)
            r.raise_for_status()
            t = pd.read_html(r.text, flavor = 'bs4', index_col = 0, header = [0,1])[0]
            t.columns = t.columns.set_levels(t.columns.levels[0].str.replace('\s\s+',' '), level = 0)
            t.columns = t.columns.set_levels(t.columns.levels[1].str.replace('\s\s+',' '), level = 1)
            t = t.applymap(lambda x: float(x.strip("%"))/100 if isinstance(x,str) and "%" in str(x) else x)
            return t
        elif sort == 'impl':
            url = self.base_url + 'histimpl.html'
            r = requests.get(url)
            r.raise_for_status()
            t = pd.read_html(r.text, flavor = 'bs4', index_col = 0, header = [0])[0]
            t.columns = t.columns.str.replace('\s\s+',' ')
            t = t.applymap(lambda x: float(x.strip("%"))/100 if isinstance(x,str) and "%" in str(x) else x)
            return t

#==================================================================================================================================
#==================================================================================================================================
#=================================================##INDUSTRY DATA CLASS##==========================================================


class IndustryData(DamoData):
    '''
    Class representing industry data
    
    Attributes
    ----------------
        var: str
            Upper case variable name
        links: list
            List of .htm links from getLinks parent method
        url: str
            URL link for the specified variable
    Methods
    ----------------
        getTable()
            Returns industry data table, indexed by industry classification by Damodaran
        rawTable()
            Returns raw table retrieved by simply iterating over rows and columns of HTML table
        parseTable()
            Returns table parsed by pd.read_html function
    '''
    
    def __init__(self, var):
        super().__init__()
        self.var = var.lower() #work around to standartise lookup for lower case
        self.links = super().getLinks() # get self.htms initiliased which is then subset 
        
        assert self.var+'.htm' in self.htms, "Variable not available, see available variables in .htms attribute or .listNames()"
            
        self.url = self.links[[x.lower() for x in self.htms].index(self.var+".htm")] #to get htm table format
        #includes workaround turning all names into lowercase
    
    def getTable(self):
        '''
        Strictly deals with tables containing INDUSTRY data.
        '''
        table_body = super().getSoup(self.url).find('body')
        trows = table_body.find('table').find_all('tr')
        row_col = [row.find_all('td') for row in trows]
        data = pd.DataFrame([[cell.text for cell in rc] for rc in row_col]).dropna().reset_index(drop = True)
        data.columns = data.loc[0].apply(lambda x: " ".join(re.sub("\r\n","",x).split()))
        data.drop(0, axis = 0, inplace = True)
        if "Industry Name" in data.columns.tolist():
            data['Industry Name'] = data['Industry Name'].apply(lambda x: " ".join(re.sub("\r\n","",x).split()))
            return data.set_index('Industry Name', drop = True)
        elif "Industry name" in data.columns.tolist():
            data['Industry name'] = data['Industry name'].apply(lambda x: " ".join(re.sub("\r\n","",x).split()))
            return data.set_index('Industry name', drop = True)
        else:
            print("Other method required, see .rawTable()")
            
    def rawTable(self):
        '''
        Returns raw table retrieved by iterating through rows and columns of html table only
        '''
        table_body = super().getSoup(self.url).find('body')
        trows = table_body.find('table').find_all('tr')
        row_col = [row.find_all('td') for row in trows]
        data = pd.DataFrame([[cell.text for cell in rc] for rc in row_col])
        return data.dropna()
    
    def parseTable(self, idx = 0, col = 0):
        '''
        Returns parsed table using pd.read_html function.
        
        Args
        ----------------
        idx: int, list, default 0
            int or list defining columns to be defined as index
        col: int, list default 0
            Rows to be defined as columns
        '''
        ###For multiindexed dataframes neet to redefine header and/or index_col. Alternatively skip first row.
        ##Handling if .HTM is not available --> check .html instead
        #Direct handling of requests used for pd.read_html
        try:
            r = requests.get(self.url)
            r.raise_for_status()
            table = pd.read_html(r.text, flavor = 'bs4', header = col, index_col = idx)[0]
        except requests.HTTPError:
            r = requests.get(self.url.replace('.htm','.html'))
            r.raise_for_status()
            table = pd.read_html(r.text, flavor = 'bs4', header = col, index_col = idx)[0]
        
        table = table.applymap(lambda x: float(x.strip("%"))/100 if isinstance(x,str) and "%" in str(x) else x)
        
        ##If index is str-like apply cleaning function
        if table.index.dtype not in [np.dtype(t) for t in ['i','int','f','float','c','complex']]:
            table = clean_idx(table)
            
        return table