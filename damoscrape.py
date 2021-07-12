import pandas as pd
import numpy as np
import requests
import re
import time
from itertools import compress
from bs4 import BeautifulSoup

class DamoData:
    
    def __init__(self):
        self.base_url = "http://people.stern.nyu.edu/adamodar/New_Home_Page/datafile/"
        self.industry_xls = "http://www.stern.nyu.edu/~adamodar/pc/datasets/indname.xls?raw=true"
        self.soup = self.getSoup(self.base_url)
        self.hrefs = [l['href'] for l in self.soup.find_all('a')]
        self.htms = None
        self.ind_group = None
        
    def getSoup(self, url):
        r = requests.get(url)
        r.raise_for_status() #check for 4xx and 5xx status
        return BeautifulSoup(r.text,"lxml")
        
    def getLinks(self): #get only HTM links
        self.htms = list(compress(self.hrefs,[ref.endswith("htm") for ref in self.hrefs]))
        return [self.base_url + link for link in self.htms]
    
    def getPngs(self):
        png_links = list(compress(self.hrefs,[ref.endswith("png") for ref in self.hrefs]))
        return [self.base_url + link for link in png_links]
    
    def getXls(self):
        xls_links = list(compress(self.hrefs,[ref.endswith("xls") for ref in self.hrefs]))
        return [self.base_url + link for link in xls_links]
    
    def listNames(self):
        return [re.sub("\\..*|_.*","", ref) for ref in self.hrefs]
    
    def lookupCompany(self, name = "", ticker = "", industry = "", country = ""):
        '''
        First call of this method attempts to read xls file stored at URL stored in industry_xls attribute of the class, takes approx 30s.
        The pandas dataframe is then stored in the .ind_group attribute, which is then searched through to find the match to user inputs.
        Subsequent calls of the method only go through the dataframe stored in the .ind_group attribute and skip the pd.read_excel attempt.
        '''
        if self.ind_group is None:
            r = requests.get(self.industry_xls)
            self.ind_group = pd.read_excel(r.content, sheet_name = "Global alphabetical")
            #only the first call of the method initiliases df --> takes long time
            #all subsequent calls will use the industry grouping dataframe initialised in the first call
        name_cond = self.ind_group['Company Name'].str.contains(name, na = False)
        ticker_cond = self.ind_group['Exchange:Ticker'].str.contains('.*:'+ticker, na = False) #does not go through Exchanges' names
        ind_cond = self.ind_group['Industry Group'].str.contains(industry, na = False)
        country_cond = self.ind_group['Country'].str.contains(country)
        cond = [name_cond, ticker_cond, ind_cond, country_cond]
    
        return self.ind_group[np.logical_and.reduce(cond)]
        
class IndustryData(DamoData):
    
    def __init__(self, var):
        super().__init__()
        self.var = var.lower() #work around to standartise lookup for lower case
        self.links = super().getLinks() # get self.htms initiliased which is then subset 
        
        if (self.var+".htm" not in self.htms):
            raise KeyError("Variable not available, see available variables in .htms attribute or .listNames()")
            
        self.url = self.links[[x.lower() for x in self.htms].index(self.var+".htm")] #to get htm table format
        #includes workaround turning all names into lowercase
        
    def getTable(self): 
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
        table_body = super().getSoup(self.url).find('body')
        trows = table_body.find('table').find_all('tr')
        row_col = [row.find_all('td') for row in trows]
        data = pd.DataFrame([[cell.text for cell in rc] for rc in row_col])
        return data.dropna()
    
    def parseTable(self):
        table = pd.read_html(self.url, flavor = 'bs4', header = 0, index_col = 0)[0]
        table = table.applymap(lambda x: float(x.strip("%"))/100 if isinstance(x,str) and "%" in str(x) else x)
        table.index = table.index.str.replace(r"\(","").str.replace(r"\)","") #parentheses
        table.index = table.index.str.replace("\s\s+" ," ") #doublespaces and tabs
        return table