#Importing necessary dictionaries
import pandas as pd
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_rows',8000)
pd.set_option('display.max_columns',20)
pd.set_option('display.width',8000)
from wordcloud import WordCloud
import datetime as dt
from functools import reduce
import missingno as msno
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
from scipy.stats import linregress
from sklearn.model_selection import RepeatedStratifiedKFold
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

#DATA EXTRACTION - WEB SCRAPING

#1 - Imdb movies dataset
import requests
file = requests.get("https://www.kaggle.com/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows/download")
df = file.text
#df
#read file and assign it to variable
#give write permission to the opened file
#write the contents of the heart_dis variable to the file, then close the file by appyling the changes
dfa = open("imdb_top_1000.csv", "w")
dfa.write(df)
dfa.close()
#2-Oscar award movies dataset
file = requests.get("https://www.kaggle.com/unanimad/the-oscar-award/download")
df = file.text
#read file and assign it to variable
#give write permission to the opened file
#write the contents of the heart_dis variable to the file, then close the file by appyling the changes
oscar_df = open("oscar.csv", "w")
oscar_df.write(df)
oscar_df.close()
#Import csv files into a Pandas DataFrame
dfa = pd.read_csv('C:\\Users\\serta\\Desktop\\IMDB_kaggle\\imdb_top_1000.csv')
dfa.head()
oscar_df = pd.read_csv("C:\\Users\\serta\\Desktop\\my_data\\the_oscar_award.csv")
oscar_df.head()
#MERGING IMDB AND OSCAR MOVIES DATASET
#extracting Oscar awarded movies
oscar_winner_df = oscar_df[oscar_df['winner'] == True]
df = oscar_winner_df[['year_ceremony','film','winner']]
#Changing the names of the columns
df.columns = ['Ceremony Year','Series_Title','win']
#sorting according to series title
df.sort_values('Series_Title').head()
#sorting according to series title
dfa = dfa.sort_values(by = 'Series_Title')
df.head(1)
left = dfa
right = df
#merging two data frames on Series title column
#merging outer methof for taking union of both datasets
merged_data = pd.merge(left,right,on = 'Series_Title',how = 'outer')
merged_data.head()
#DATATYPE MANIPULATION
#Getting information of missing values and datatypes
merged_data.info()
#filling missing win values with false because they are not awarded
merged_data['win'].fillna('False',inplace = True)
merged_data.head(1)
#getting information of duplicated rows
merged_data[merged_data.duplicated()].head()
#dropping duplicates
merged_data.drop_duplicates(inplace= True)

merged_data.dropna(subset = ['Poster_Link'],inplace = True)
merged_data[merged_data['Ceremony Year'].isnull()].head(10)
#data is missing since the movies were not awarded so there was no ceremony.
#filling nan with 1900 ,it is a random year which I pick because there was no Oscar awards in that years
merged_data['Ceremony Year'].fillna(1900,inplace = True)
merged_data.info()
#Replacing 'minutes' and changing datatype from object to integer
merged_data['Runtime'] = [int(str(i).replace("min", "")) for i in merged_data['Runtime']]
#filling nan values with 0 to be able to change the datatype from object to integer
merged_data['Gross'] = merged_data['Gross'].fillna(0)
merged_data['Gross'] = [int(str(i).replace(",", "")) for i in merged_data['Gross']]
numerical_attributes = ['Ceremony Year','Runtime','No_of_Votes']
#there is no day or month on ceremony year column so instead of date format,it will be changed to integer
merged_data[numerical_attributes] = merged_data[numerical_attributes].astype('int64')
merged_data['Gross'] = merged_data['Gross'].astype('float64')
#writing list comprehension for win column instead of for loop which is more declarative and shorter
merged_data['win'] = [1 if x == True else 0 for x in merged_data['win']]
merged_data.info()
#looking for unique values for detecting anormal values
merged_data['Released_Year'].unique()
#finding out the series title of released year with a value of 'PG'
merged_data[merged_data['Released_Year'] == 'PG']
#replacing 'PG' with correct year value after finding out the correct value of released year via internet
merged_data['Released_Year'] = merged_data['Released_Year'].str.replace('PG','1995')
#Changing datatype from object to datetime
merged_data['Released_Year'] = pd.to_datetime(merged_data['Released_Year'])
#Creating new column by taking year values
merged_data['Year_of_release'] = merged_data['Released_Year'].dt.year
merged_data['Year_of_release'].head(1)
#dropping the released year column
merged_data.drop('Released_Year',inplace = True,axis = 1)
merged_data.info()
#Merged data should have 1000 entries but it has 1007 so it will be examined :
#extracting duplicated movies on series title
duplicated_titles = merged_data[merged_data.duplicated(subset='Series_Title' )]
duplicated_titles.iloc[:,[1,15,16]]
duplicated_titles[duplicated_titles['Ceremony Year'] < duplicated_titles['Year_of_release']]
#ceremony year of awarded movie cannot be smaller than year of release so 36 and 686 will be removed.
merged_data[merged_data['Series_Title'].isin(['Titanic','Little Women','A Star Is Born','Drishyam','King Kong','Up'])].iloc[:,[1,15,16,17]]
#will drop index ceremony year is smaller than released year and duplicated Drishyam
merged_data.drop(index = [35,36,336,685,686,1336,1387],inplace = True)
merged_data.info()
#MISSING VALUES IMPUTATION
merged_data.isna().sum()
#Missing value visualization for detecting if the missing values have any relationhip between different variables to detect missingness type(MCAR,MNAR,MAR)
import numpy as np
import missingno as msno
#%matplotlib inline
#Visualize missingness
msno.matrix(merged_data)
plt.show()
#calculating percentage of missing values
def missing_values_percentage(df):
    return (df.isna().sum())/len(df.values)*100
missing_values_percentage(merged_data)
#IMPUTATING GROSS COLUMN
#for datatype manipulation ,nan values were imputed with zero before,now these 0 values will be imputed after having descriptive statistical values of the gross column
merged_data['Gross'].describe()
sns.histplot(merged_data['Gross'])
plt.xlabel('Gross',fontsize=10)
plt.show()
merged_data['Gross'].describe()
len(merged_data[merged_data['Gross'] == 0])
merged_data['Gross'].median()
#because the number of 0 values are nearly %17 of the values instead of mean median values will be used for imputation
merged_data['Gross'] = [merged_data['Gross'].median() if x == 0 else x for x in merged_data['Gross']]
merged_data['Gross'].describe()
merged_data['Gross'].median()
sns.histplot(merged_data['Gross'])
plt.xlabel('Gross',fontsize=10)
plt.show()
merged_data['Gross'].nsmallest()
merged_data['Gross'].nlargest()
merged_data['Gross'].isnull().sum()
#IMPUTATING CERTIFICATE COLUMN ACCORDING TO GENRES COLUMN



