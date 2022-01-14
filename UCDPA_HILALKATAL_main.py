"""Importing necessary dictionaries"""
from wordcloud import WordCloud
import datetime as dt
from functools import reduce
import matplotlib.pyplot as plt
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

"""DATA EXTRACTION - WEB SCRAPING"""

"""1 - Imdb movies dataset"""
"""Sending http request with request module"""
import requests
file = requests.get("https://www.kaggle.com/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows/download")
print(file.status_code)
df = file.text
"""read file and assign it to variable
give write permission to the opened file
write the contents of the heart_dis variable to the file, then close the file by appyling the changes"""
dfa = open("imdb_top_1000.csv", "w")
dfa.write(df)
dfa.close()
"""2-Oscar award movies dataset"""
file = requests.get("https://www.kaggle.com/unanimad/the-oscar-award/download")
print(file.status_code)
df = file.text
"""read file and assign it to variable
give write permission to the opened file
write the contents of the heart_dis variable to the file, then close the file by appyling the changes"""
oscar_df = open("oscar.csv", "w")
oscar_df.write(df)
oscar_df.close()
"""3- Imdb title basics data"""
r = requests.get('https://datasets.imdbws.com/title.basics.tsv.gz')
with open('C:\\Users\\serta\Downloads\\title.basics.tsv.gz', 'wb') as file:
    file.write(r.content)
print(r.status_code)
""" Read csv files into dataframe with pandas"""
import pandas as pd
dfa = pd.read_csv('C:\\Users\\serta\\Desktop\\IMDB_kaggle\\imdb_top_1000.csv')
dfa.head()
oscar_df = pd.read_csv("C:\\Users\\serta\\Desktop\\my_data\\the_oscar_award.csv")
oscar_df.head()
"""There is no column of title type(e.g. movies,tv series) in dfa file.
Since the target variable in the analysis is whether to win an Oscar or not, we will only extract movies from the file.
Because only movies are honoured with Oscar award.Therefore, title.basics.tsv.gz will be downloaded from IMDb database to get the 
title types of dfa"""
#getting list of name of the movies of dfa file
titles = [x for x in dfa['Series_Title']]
#titles
#Read csv in dataframe
df = pd.read_csv('C:\\Users\\serta\Downloads\\title.basics.tsv.gz',usecols = ['titleType','originalTitle'],dtype={'titleType':'category','originalTitle':'category'} ,delimiter="\t")
df['titleType'].unique()
df_movies = df[df['titleType']=='movie']
df_movies.head()
df_movies = df_movies[df_movies['originalTitle'].isin(titles)].drop_duplicates(subset = 'originalTitle' )
#df_movies
#getting list of name of the movies of df file
titles_movies = [x for x in df_movies['originalTitle']]
#extracting movies from dfa file
dfa = dfa[dfa['Series_Title'].isin(titles_movies)]
dfa.info()

"""MERGING IMDB AND OSCAR MOVIES DATASET"""
"""extracting Oscar awarded movies"""
oscar_winner_df = oscar_df[oscar_df['winner'] == True]
df = oscar_winner_df[['year_ceremony', 'film', 'winner']]
"""Changing the names of the columns"""
df.columns = ['Ceremony Year', 'Series_Title', 'win']
"""sorting according to series title"""
df.sort_values('Series_Title').head()
"""sorting according to series title"""
dfa = dfa.sort_values(by='Series_Title')
df.head(1)
left = dfa
right = df
"""merging two data frames on Series title column"""
"""merging outer method for taking union of both datasets"""
merged_data = pd.merge(left, right, on='Series_Title', how='outer')
merged_data.head()
#dropping ceremony year column
"""DATATYPE MANIPULATION"""
"""Getting information of missing values and datatypes"""
merged_data.info()
"""filling missing win values with false because they are not awarded"""
merged_data['win'].fillna('False', inplace=True)
merged_data.head(1)
"""getting insight of duplicated rows"""
merged_data[merged_data.duplicated()].head()
"""dropping duplicates"""
merged_data.drop_duplicates(inplace=True)
merged_data.dropna(subset=['Poster_Link'], inplace=True)
merged_data[merged_data['Ceremony Year'].isnull()].head(10)
"""data is missing since the movies were not awarded so there was no ceremony."""
"""filling nan with 1900 ,it is a random year which I pick because there was no Oscar awards in that years"""
merged_data['Ceremony Year'].fillna(1900, inplace=True)
merged_data.info()
"""Replacing 'minutes' and changing datatype from object to float"""
merged_data['Runtime'] = [int(str(i).replace("min", "")) for i in merged_data['Runtime']]
"""filling nan values with 0 to be able to change the datatype from object to integer"""
merged_data['Gross'] = merged_data['Gross'].fillna(0)
merged_data['Gross'] = [int(str(i).replace(",", "")) for i in merged_data['Gross']]
numerical_attributes = ['Ceremony Year', 'Runtime', 'No_of_Votes']
"""there is no day or month on ceremony year column so instead of date format,it will be changed to integer,it will bw dropped later"""
merged_data[numerical_attributes] = merged_data[numerical_attributes].astype('int64')
merged_data['Gross'] = merged_data['Gross'].astype('float64')
"""writing list comprehension for win column instead of for loop which is more declarative and shorter"""
merged_data['win'] = [1 if x == True else 0 for x in merged_data['win']]
merged_data.info()
"""looking for unique values for detecting anormal values"""
merged_data['Released_Year'].unique()
"""finding out the series title of released year with the value of 'PG'"""
print(merged_data[merged_data['Released_Year'] == 'PG'])
"""replacing 'PG' with correct year value after finding out the correct value of released year via internet"""
merged_data['Released_Year'] = merged_data['Released_Year'].str.replace('PG', '1995')
"""Changing datatype from object to datetime"""
merged_data['Released_Year'] = pd.to_datetime(merged_data['Released_Year'])
"""Creating new column by taking year values"""
merged_data['Year_of_release'] = merged_data['Released_Year'].dt.year
merged_data['Year_of_release'].head(1)
"""dropping the released year column"""
merged_data.drop('Released_Year', inplace=True, axis=1)
merged_data.info()
"""Merged data should have 983 entries(since dfa has 983 rows) but it has 990 so it will be examined :"""
"""extracting duplicated movies on series title"""
duplicated_titles = merged_data[merged_data.duplicated(subset='Series_Title')]
print(duplicated_titles)
"""calling duplicated movies from imdb movies dataset"""
dfa[dfa['Series_Title'].isin(['Titanic','Little Women','A Star Is Born','Drishyam','King Kong','Up'])]
"""calling duplicated movies with wrong ceremony year"""
duplicated_titles[duplicated_titles['Ceremony Year'] < duplicated_titles['Year_of_release']]
#Ceremony year of awarded movie cannot be smaller than year of release so 36 and 677 will be removed.
print(merged_data[merged_data['Series_Title'].isin(['Titanic', 'Little Women', 'A Star Is Born', 'Drishyam', 'King Kong', 'Up'])].iloc[:, [1, 15, 16, 17]])
merged_data[merged_data['Series_Title'] == 'King Kong']
#duplicated rows and films that are not in imdb movies will be drop from merged data(Oscar+imdb movies data)
merged_data.drop(index = [35,36,593,676,677,1318,1369],inplace = True)
#Ceremony year of awarded movie cannot be smaller than year.
#getting information of released year range of Oscar df
print(oscar_df['year_film'].nsmallest(2),oscar_df['year_film'].nlargest(2))
#getting information of released year range of merged data
print(merged_data['Year_of_release'].nsmallest(2),merged_data['Year_of_release'].nlargest(2))
"""First Oscar award , honored the best films of 1927 and 1928 so the years smaller than 1927 will be dropped"""
"""Oscar awarded data has movies up to 2019 so 2020 movies will be dropped from the merged data for accurate results"""
index_of_inappropriate_dates = merged_data[merged_data['Year_of_release'].isin([1920,1921,1922,1923,1924,1925,1926,2020])].index
print(index_of_inappropriate_dates)
merged_data = merged_data.drop(index=index_of_inappropriate_dates, axis=0)
merged_data.info()

"""MISSING VALUES IMPUTATION"""
merged_data.isna().sum()
"""Missing value visualization for detecting if the missing values have any relationhip between variables to detect missingness is not random or not"""
"""Visualize missingness"""
import numpy as np
import missingno as msno
msno.matrix(merged_data)
plt.show()
"""calculating percentage of missing values"""
def missing_values_percentage(df) :
    percentage = (df.isna().sum())/len(df.values)*100
    return percentage
print(missing_values_percentage(merged_data))
"""
IMPUTATING GROSS COLUMN
for datatype manipulation ,nan values were imputed with zero before,
now these 0 values will be imputed after having descriptive statistical values of the gross column
"""
merged_data['Gross'].describe()
#Finding out the correlation between gross column and other numerical columns
columns = ['IMDB_Rating','Meta_score','No_of_Votes','Gross']
subset = merged_data[columns]
subset.corr()
#"""according to results,there is a high correlation between gross and number of votes column.Therefore,imputation will be made according to number of votes column
#extracting data in which gross isnot null
data = merged_data[~merged_data['Gross'].isnull()]

#getting linear regression equation between gross and no_o_votes column
from scipy.stats import linregress

xi = data['No_of_Votes']
yi = data['Gross']
# Compute the linear regression
results = linregress(xi,yi)
print(results)
"""According to results,per a vote,gross increases 189.88 points"""
#VISUALIZATION BEFORE IMPUTATION
#Group by numbers of votes
grouped = merged_data.groupby('No_of_Votes')
#Getting mean values of gross
mean_gross_by_votes = grouped['Gross'].mean()
#plotting of mean values of gross vs number of votes
plt.plot(mean_gross_by_votes,'o',alpha = 0.5)
plt.xlabel('No_of_Votes')
plt.ylabel('Gross')
plt.show()

"""Plotting best fit line on the data points of metascores and imdb scores"""
#Scatterplot between no_of_votes and gross
plt.plot(xi,yi,'o',alpha = 0.1)
#plot the best linear fit line on scatter plot
#creating numpy array including min to max values of number of votes
x = np.array([xi.min(),xi.max()])
#creating linear equation according to linear regression slope and intercept values
y = results.intercept+results.slope*x
#plotting the best fit line
plt.plot(x,y,'-',alpha = 0.7)
plt.xlabel('No_of_Votes')
plt.ylabel('Gross')
plt.show()
"""IMPUTATION ACCORDING TO LINEAR REGRESSION RESULTS"""
#getting index number of null gross rows(for datatype manipulation null values were imputed with 0 (zero) before)
gross_null = merged_data[merged_data['Gross']== 0]
#creating list of index numbers
gross_index_lst = gross_null.index
#extracting null values of gross
gross_scores_null = merged_data.loc[gross_index_lst]['Gross']
#extracting no of votes values of null gross values
votes_null = merged_data.loc[gross_index_lst]['No_of_Votes']
#extracting null data
data_null = merged_data.loc[gross_index_lst]
#calculating gross values according to intercept and coefficient value of votes
gross_scores_null = round((5263703.110095441) + ((190.4525132397611)*votes_null))
#imputating null gross rows
merged_data.loc[gross_index_lst,'Gross'] = gross_scores_null
merged_data.loc[gross_index_lst].head()
#The correlation between gross column and votes column after imputation
columns = ['No_of_Votes','Gross']
subset = merged_data[columns]
subset.corr()
#VISUALIZATION AFTER IMPUTATION
#Group by numbers of votes
grouped = merged_data.groupby('No_of_Votes')
#Getting mean values of gross
mean_gross_by_votes = grouped['Gross'].mean()
#plotting of mean values of gross vs number of votes
plt.plot(mean_gross_by_votes,'o',alpha = 0.5)
plt.xlabel('No_of_Votes')
plt.ylabel('Gross')
plt.show()
merged_data['Gross'].describe()
merged_data['Gross'].isnull().sum()

"""IMPUTATING CERTIFICATE COLUMN ACCORDING TO GENRES COLUMN"""
merged_data['Certificate'].value_counts()
"""Defining a function for certificate values imputation"""
import re
mask2 = merged_data['Certificate'].isnull()
def certificate_imp(reg):
    """Finding mode value of certificate according to genres group"""
    mask1 = merged_data['Genre'].str.contains(reg)
    # creating dictionary to take the mode value of certificate values
    x = dict(merged_data[mask1].iloc[:, 2].value_counts())
    # print(list(x.keys())[0])
    result = list(x.keys())[0]
    return result
merged_data[merged_data['Certificate'].isnull()]['Genre'].value_counts()
mask1 = merged_data['Genre'].str.contains(r"^Drama(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"^Drama(?!,)")
# negative loohahead assertion
mask1 = merged_data['Genre'].str.contains(r"(Comedy, Drama)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(^Comedy, Drama)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Drama, War)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(^Drama, War)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Comedy, Drama, Romance)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(^Comedy, Drama, Romance)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Action, Crime, Drama)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(^Action, Crime, Drama)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Drama|Thriller)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(^Drama|Thriller)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Drama, Romance)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(^Drama, Romance)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Comedy, Romance)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(^Comedy, Romance)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Mystery, Thriller)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(^Mystery, Thriller)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Crime, Drama, Film-Noir)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(^Crime, Drama, Film-Noir)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Crime, Drama, Mystery)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(^Crime, Drama, Mystery)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Biography, Drama, History)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(^Biography, Drama, History)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Adventure, Drama, Western)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(^Adventure, Drama, Western(?!,))")
mask1 = merged_data['Genre'].str.contains(r"(Comedy, Drama, War)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(^Comedy, Drama, War)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Crime, Drama)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(^Crime, Drama)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Drama, Music, Romance)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(^Drama, Music, Romance)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Action, Adventure, Drama)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(^Action, Adventure, Drama)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Drama, Film-Noir)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(Drama, Film-Noir)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Drama, History)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(^Drama, History)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Drama, Horror)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(^Drama, Horror)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Crime, Drama, Romance)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(^Crime, Drama, Romance)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Comedy, Drama, Family)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(^Comedy, Drama, Family)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Drama, Horror, Sci-Fi)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(^Drama, Horror, Sci-Fi)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Horror)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(Horror)")
mask1 = merged_data['Genre'].str.contains(r"(Adventure, Comedy)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(Adventure, Comedy)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Animation, Adventure, Family)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(Animation, Adventure, Family)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Comedy, Music, Musical)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(Comedy, Music, Musical)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Drama, Sci-Fi)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(^Drama, Sci-Fi)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Drama, Fantasy, Mystery)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(^Drama, Sci-Fi)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Drama, Western)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(^Drama, Western)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Comedy, Crime)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(^Comedy, Crime)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Family)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(Family)")
mask1 = merged_data['Genre'].str.contains(r"(Crime, Drama, Fantasy)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(Crime, Drama, Fantasy)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Comedy, War)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(Comedy, War)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Drama, Fantasy)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(Drama, Fantasy)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Drama, Romance, War)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(Drama, Romance, War)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Action, Drama, Mystery)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(Action, Drama, Mystery)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Action, Adventure, War)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(Action, Adventure, War)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Drama, Film-Noir, Mystery)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(Drama, Film-Noir, Mystery)(?!,)")
mask1 = merged_data['Genre'].str.contains(r"(Film-Noir, Mystery)(?!,)")
merged_data.loc[mask1 & mask2, 'Certificate'] = certificate_imp(r"(Film-Noir, Mystery)(?!,)")
"""These rows throw 'out of range' error while imputing with certificate_imp function, so I will call them"""
merged_data[merged_data['Certificate'].isnull()]['Genre'].value_counts()
merged_data[merged_data['Genre'].isin(['Action, Adventure, Biography','Action, Adventure, Crime','Comedy, Musical, War','Action, Crime, Comedy','Film-Noir, Mystery'])]
"""Since there are no genres similar to these ones in our data, these nan 
certificate columns will be imputed with approppriate value after searching on 'https://www.imdb.com/' """
"""Imputing  movies' certificate"""
merged_data.loc[[49,241],'Certificate'] = 'A'
merged_data.loc[336,'Certificate'] = 'U'
merged_data.loc[601,'Certificate'] = 'TV-14'
merged_data.loc[1214,'Certificate'] = 'PG'
merged_data.loc[[49,241,336,601,1214]]
merged_data['Certificate'].isnull().sum()
merged_data.info()
merged_data['Certificate'].value_counts()
"""Creating certificate groups by mapping"""
mapping = {'G': 'All_ages_group','U' : 'All_ages_group' , 'PG':'All ages(kids with PG)','TV-PG':'All ages(kids with PG)','UA':'All ages(kids with PG)','U/A':'All ages(kids with PG)','GP':'All ages(kids with PG)','Passed' : 'All ages(kids with PG)','PG-13' : 'Teens', 'TV-14': 'Teens','16' : 'Teens','R' : 'Adult' ,'TV-MA' : 'Adult','A' : 'Adult'}
merged_data['Certificate']= merged_data['Certificate'].replace(mapping)
merged_data['Certificate']=merged_data['Certificate'].replace(mapping)
merged_data['Certificate'].value_counts()
"""IMPUTATION OF METASCORE COLUMN"""
len(merged_data[merged_data['Meta_score'].isnull()])
merged_data['Meta_score'].describe()
"""Finding out the correlation between metascore column and other numerical columns"""
columns = ['IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Gross']
subset = merged_data[columns]
subset.corr()
"""according to results,although it is a small correlation ,only imdb scores have correlation with metascores,
so imputation will be made according to imdb score column."""
"""VISUALIZATION BEFORE IMPUTATION"""
"""Group by imdb scores"""
grouped = merged_data.groupby('IMDB_Rating')
"""Getting mean values of metascores"""
mean_metascore_by_imdb = grouped['Meta_score'].mean()
print(mean_metascore_by_imdb)
"""plotting mean values of metascore vs imdb scores"""
plt.plot(mean_metascore_by_imdb, 'o', alpha=0.5)
plt.xlabel('IMDB_Rating')
plt.ylabel('Meta_score')
plt.title('Metascore vs imdb scores before imputation')
plt.show()
"""extracting data in which metascores isnot null"""
data = merged_data[~merged_data['Meta_score'].isnull()]
"""getting linear regression equation between metascore and imdb scores"""
from scipy.stats import linregress
xi = data['IMDB_Rating']
yi = data['Meta_score']
"""Compute the linear regression"""
results = linregress(xi, yi)
print(results)
"""according to results it means that metascores increases 11.71 points per 1 point of imdb scores"""
"""Plotting best fit line on the data points of metascores and imdb scores"""
"""Scatterplot of imdb scores and metascores"""
plt.plot(xi, yi, 'o', alpha=0.1)
"""plot the best fit linear regression line"""
"""creating numpy array including min to max values of imdb scores"""
x = np.array([xi.min(), xi.max()])
"""creating linear equation according to linear regression slope and intercept values"""
y = results.intercept+results.slope*x
"""plotting best fit line"""
plt.plot(x, y, '-', alpha=0.7)
plt.xlabel('imdb scores')
plt.ylabel('Metascores')
plt.title('Linear equation between metascore and imdb scores')
plt.show()
"""IMPUTATION ACCORDING TO LINEAR REGRESSION RESULTS"""
"""getting index number of null metascores rows"""
metascore_null = merged_data[merged_data['Meta_score'].isnull()]
"""creating list of index numbers"""
metascore_index_lst = metascore_null.index
"""extracting null values of metascores"""
meta_scores_null = merged_data.loc[metascore_index_lst]['Meta_score']
"""extracting null values of imdb scores"""
imdb_rating_null = merged_data.loc[metascore_index_lst]['IMDB_Rating']
"""extracting null data"""
data_null = merged_data.loc[metascore_index_lst]
#calculating metascores values according to intercept and coefficient value of imdb scores
meta_scores_null = round((-16.90239877564008) + ((11.960092936857253)*imdb_rating_null))
"""imputating null metascores rows"""
merged_data.loc[metascore_index_lst,'Meta_score'] = meta_scores_null
merged_data.loc[metascore_index_lst].head()
merged_data.info()
"""VISUALIZATION AFTER IMPUTATION"""
grouped = merged_data.groupby('IMDB_Rating')
mean_metascore_by_imdb = grouped['Meta_score'].mean()
print(mean_metascore_by_imdb)
plt.plot(mean_metascore_by_imdb,'o',alpha = 0.5)
plt.xlabel('IMDB_Rating')
plt.ylabel('Meta_score')
plt.title('Metascore vs imdb scores after imputation')
plt.show()
#The correlation between metascore column imdb scores after imputation
columns = ['IMDB_Rating','Meta_score']
subset = merged_data[columns]
subset.corr()

"""DISTRUBUTION OF NUMERICAL ATTIBUTES"""
"""PLOTTING DISTRIBUTION"""
import seaborn as sns
numerical_columns = ['Runtime','IMDB_Rating','Meta_score','No_of_Votes','Gross','Year_of_release']
for x in numerical_columns:
    sns.histplot(merged_data[x],kde = True,bins = len(merged_data[x].unique()))
    plt.xlabel(x)
    title = print('Distribution of',x)
    plt.title(title)
    plt.show()
merged_data['Gross'].nsmallest()
"""Descriptive analytics"""
merged_data[numerical_columns].describe()
"""Having insight of categorical variables unique values for detecting anormality"""
merged_data['Certificate'].unique()
merged_data['Genre'].unique()
merged_data['Director'].unique()
merged_data['Star1'].unique()
merged_data['Star2'].unique()
merged_data['Star3'].unique()
merged_data['Star4'].unique()
merged_data['Series_Title'].unique()

"""EXPLORATORY ANALYSIS"""

"""Question1; Who are the director's of 5 movies with highest gross values  and what are the genres of these films?"""
#getting rows including  first 5 highest gross values
merged_data_gross_max = merged_data.sort_values(by = 'Gross', ascending=False).head()
#merged_data_gross_max
"""Plotting stripplot"""
sns.stripplot(x = 'Director',y = 'Gross',hue = 'Genre',data = merged_data_gross_max)
plt.xticks(rotation=40)
plt.title('The directors and genres of the top 5 highest-grossing films')
plt.show()

"""Question 2 = Who are the leading actors of the films with the highest 5 imdb points that are Oscar awarded and 
what is the certificate of these films?"""
"""extracting oscar awarded movies"""
awarded = merged_data[merged_data['win']== 1]
awarded = awarded.sort_values(by= 'IMDB_Rating',ascending = False).head()
#awarded
sns.stripplot(x = 'Certificate',y = 'Star1',hue = 'Series_Title',data = awarded)
plt.title('The leading actors and certificates of the movies with the highest top 5 imdb scores')
plt.show()

"""Question 3 : Which words were used mostly in the overview of the top 50 movies with highest number of votes which are Oscar awarded? """

awarded = merged_data[merged_data['win']== 1]
awarded = awarded.sort_values(by= 'No_of_Votes',ascending = False).head(50)


def plot_cloud(wordcloud):
    plt.figure(figsize=(15, 15))
    plt.imshow(wordcloud)

    plt.axis("off");


wordcloud = WordCloud(width=500, height=500, background_color='#40E0D0', colormap="OrRd", random_state=10).generate(
    ' '.join(awarded['Overview']))
plot_cloud(wordcloud)

"""Question 4 : According to descriptive statistics,awarded and not awarded datas have almost similar statistical values of imdb scores.
Are they really similar or not?"""

#group data by win column
merged_data.groupby('win')['IMDB_Rating'].describe()
#oscar awarded data
awarded_data = merged_data[merged_data['win']==1]
#not awarded data
not_oscar_awarded_data = merged_data[merged_data['win']==0]
#imdb scores of awarded movies
imdb_oscar = awarded_data['IMDB_Rating']
#imdb scores of not awarded movies
imdb_not_oscar = not_oscar_awarded_data['IMDB_Rating']
"""Write function for plotting cumulative distribution function """
def ecdf(df):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points
    length = len(df)

    # sorting x array: x
    x = np.sort(df)

    # y-data for the ECDF: y
    y = np.arange(1, length+1) / length

    return x, y

"""Plotting ecdf of imdb scores of awarded and not awarded data"""
x,y = ecdf(imdb_not_oscar)
x_oscar,y_oscar = ecdf(imdb_oscar)
_ = plt.plot(x,y,marker = '*',linestyle = '-',label = 'Not_awarded')
_ = plt.plot(x_oscar,y_oscar,marker = '*',linestyle = '-',label = 'Awarded')

plt.legend(loc = 'lower right')
_ = plt.xlabel('Imdb scores')
_ = plt.ylabel('ECDF')
plt.show()

"""According to ecdf results, awarded movies have higher imdb scores.
Permutation samples of imdb scores will be created to determine if they will overlap with the observed data"""
"""Creating permutation samples for 50 times"""
for _ in range(50):
    # Concatenate two datasets
    df = np.concatenate((imdb_oscar, imdb_not_oscar))

    # Permute the concatenated array: permuted_data
    permuted_df = np.random.permutation(df)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_awarded = permuted_df[:len(imdb_oscar)]
    perm_sample_not_awarded = permuted_df[len(imdb_oscar):]

    # Compute ECDFs
    x_awarded, y_awarded = ecdf(perm_sample_awarded)
    x_not_awarded, y_not_awarded = ecdf(perm_sample_not_awarded)

    # Plot ECDFs of permutation samples
    _ = plt.plot(x_awarded, y_awarded, marker='_',
                 color='red', alpha=0.01)
    _ = plt.plot(x_not_awarded, y_not_awarded, marker='_',
                 color='blue', alpha=0.01)

# Create and plot ECDFs from merged data(original data)
x_org_awarded, y_org_awarded = ecdf(imdb_oscar)
x_not_org_awarded, y_not_org_awarded = ecdf(imdb_not_oscar)
_ = plt.plot(x_org_awarded, y_org_awarded, marker='.', color='red', label='Awarded')
_ = plt.plot(x_not_org_awarded, y_not_org_awarded, marker='.', color='blue', label='Not_awarded')

# Label axes, set margin, and show plot
plt.margins(0.02)
_ = plt.xlabel('imdb')
_ = plt.ylabel('ECDF')
plt.legend(loc='lower right')
plt.title('ECDF of imdb scores awarded ,not awarded and permutation samples of concatenated awarded and not awarded movies')
plt.show()

"""According to graphic, we can see that none of permutation samples overlap on observed data ,they remain between ecdf line of awarded
and not awarded data ,which shows that imdb scores aren't identically distributed between awarded and not awarded data"""

"""Question 5= Who are the leading actors of top 20 movies with highest gross values?"""
#extracting Oscar awarded movies
awarded_data = merged_data[merged_data['win']  == 1]
#extracting leading actor and gross columns
star1_gross= awarded_data[['Star1','Gross']]
#grouping data by actors and getting mean of gross
mean_gross = star1_gross.groupby('Star1').mean()/1000000
#sorting gross values
mean_gross = mean_gross.sort_values('Gross', ascending=False)
#extracting first 20 top gross values
mean_gross_first_twenty = mean_gross.head(20)
#visualization
mean_gross_first_twenty.plot.barh(title = 'Leading actors of the movies with top 20 highest profits ',color = 'c',figsize=(11, 6))
plt.xlabel('Gross in millions')
plt.show()

"""Question 6: What is the distribution of genres of awarded movies?"""
from collections import Counter

# Counter is a sub-class to count hashable objects as key: value pairs (as a dictionary)
# extract awarded movies' genres
dtseries = awarded_data["Genre"]
# print(dtseries)


# creating empty list to collect genres
counts_lst = []
for entry in dtseries:
    # print(entry)
    # extract each genre from groups of genres
    a = entry.split(",")

    # print(a)
    for genre in a:
        # print(genre)
        # strip white space
        genre = genre.strip()
        # print(genre)
        counts_lst.append(genre)

# getting dictionary of genres and counts
Counter(counts_lst)
#creating series of counted list
awarded_genre = pd.Series(counts_lst)
#plotting genres according to counts of each genre
sns.countplot(x = awarded_genre)
plt.xticks(rotation = 50)
plt.title('Genres of awarded movies')

"""PREDICTIVE ANALYSIS"""
#changing index as titles of movies
merged_data = merged_data.set_index('Series_Title')
#dropping Overview column before encoding
merged_data.drop(columns = 'Overview',axis =1,inplace = True)

"""FEATURE ENGINEERING ON CATEGORICAL ATTRIBUTES"""
"""One-Hot encoding the categorical parameters using get_dummies()"""
Genre = merged_data['Genre']
Genre = Genre.str.get_dummies()
Certificate = merged_data['Certificate']
Certificate = Certificate.str.get_dummies()
Star1 = merged_data['Star1']
Star1 = Star1.str.get_dummies()
Star2 = merged_data['Star2']
Star2 = Star2.str.get_dummies()
Star3 = merged_data['Star3']
Star3 = Star3.str.get_dummies()
Star4 = merged_data['Star4']
Star4 = Star4.str.get_dummies()
Director = merged_data['Director']
Director = Director.str.get_dummies()
merged_data_encoded = pd.concat(
    [merged_data.drop(
        ['Genre', 'Certificate', 'Poster_Link', 'Star4', 'Director', 'Star1', 'Star2', 'Star3',
         'Ceremony Year'],
        axis=1
    ),Genre, Certificate,Star1, Star2, Star3, Star4, Director],
    axis=1,)
merged_data_encoded.head()

"""FINDING OUT OUTLIERS"""
"""PLOTTING OUTLIERS"""
for x in ['Runtime','IMDB_Rating','Meta_score','No_of_Votes','Gross']:
    boxplot = merged_data_encoded.boxplot(column=[x],figsize =(3,3))
    plt.show()

#determination of percentage of outlier values of a column
def outliers_percentage(x):
    """finding upper and lower outlier values"""
    #first quantile
    q1 = merged_data_encoded[x].quantile(0.25)
    #third quantile
    q3 = merged_data_encoded[x].quantile(0.75)
    #interquar
    Interquar=q3-q1
    #print(q1)
    #print(q3)
    #print(Interquar)
    #lower outlier value(1.5 times interquar lower than first quantile)
    Lower_outlier_value = q1-(1.5*Interquar)
    #Upper outlier value(1.5 times interquar upper than third quantile)
    Upper_outlier_value = q3+(1.5*Interquar)
    #percentage of outlier values
    percentage = round((len(merged_data_encoded[merged_data_encoded[x] > Upper_outlier_value]) + len(merged_data_encoded[merged_data_encoded[x] < Lower_outlier_value]))/merged_data_encoded.shape[0]*100,2)
    return percentage
outliers_percentage('No_of_Votes')
def sum_per_outliers(lst):
    """calculating total percentage of outliers"""
    #define empty list for percentage of outliers
    percentage_lst = []
    for x in lst:
        #getting percentage of outliers
        percentage = outliers_percentage(x)
        #collecting percentage for each column
        percentage_lst.append(percentage)
        #calculating total percentage of outliers
    return np.sum(percentage_lst)
#total percentage of outliers of all numerical columns
sum_per_outliers(['Runtime','IMDB_Rating','Meta_score','No_of_Votes','Gross'])

"""Since the amount of outliers are  25.15% percent of the data ,outliers will not be removed,
instead they  will be scaled with robust standardization which uses quartile method for scaling.
Before scaling correlation between independent variables will be examined."""

"""EXCLUDE HIGHLY CORRELATED FEATURES"""

columns_to_scale  = ['Runtime', 'IMDB_Rating','Meta_score', 'Gross', 'No_of_Votes','Year_of_release','win']
cor = merged_data_encoded[columns_to_scale].corr(method = 'pearson')
plt.figure(figsize = (10,6))
sns.heatmap(cor, annot = True ,square=True, cmap='RdYlGn',fmt='.4g')

"""The number of votes has high correlation with gross and IMDB column. 
The correlation between the number of votes and the target variable('win') is slightly higher than the correlation between  win and gross earnings.
 On the other hand, the number of votes column has high correlation with imdb scores. For this reason,
 model performance will be evaluated in scaling process by dropping each column separately."""

"""SCALE NUMERICAL COLUMNS"""
"""MODEL ACCURACY BEFORE ROBUST SCALING (AFTER DROPPING GROSS COLUMN)"""
#determining independent and dependent columns for model apply
X = merged_data_encoded.drop(columns = ['win','Gross'],axis = 1)
y = merged_data_encoded['win']
X.head(2)
"""Defining cross validation model accuracy function"""
# importing classification models
from sklearn import model_selection
from sklearn. linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
def models_accuracy_scores(model_lst,independent,dependent):
    #creating empty list for model names
    model_names = []
    #determining predictive models for classification
    models = model_lst
    #looping for each model
    for model_name, model in models:
        #create crossvalidation with 10 splits and 10 repeats with RepeatedstratifiedKfold
        cv = RepeatedStratifiedKFold(n_splits=10, random_state=1)
        #getting validation scores(scoring is accuracy because dependent variable is binary and classification analysis will be applied)
        scores = model_selection.cross_val_score(model, independent, dependent, cv=cv, scoring='accuracy')
        #getting model names for printing
        model_names.append(model_name)
        #rounding average score to 3 decimal places
        average_accuracy = round(scores.mean(),3)
        #printing the results
        print('Average accuracy score of',model_name,'is:',average_accuracy)
"""Getting cross validation accuracy scores with models_accuracy_scores function"""
#determining independent and dependent columns for model apply
X = merged_data_encoded.drop(columns = ['win','Gross'],axis = 1)
y = merged_data_encoded['win']
models = [('LogReg', LogisticRegression(solver = 'liblinear')),
          ('RandomForest',RandomForestClassifier(n_estimators=100)),
          ('DecTree', DecisionTreeClassifier()),
          ('KNN', KNeighborsClassifier()),
          ('SVM', SVC(gamma = 'scale'))]
models_accuracy_scores(models,X,y)
"""ROBUST SCALING"""
"""Getting accuracy scores of different quantile ranges"""
#determining independent and dependent columns for model apply
X = merged_data_encoded.drop(columns = ['win','Gross'],axis = 1)
y = merged_data_encoded['win']
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
#determining independent variables
columns_to_scale  = merged_data_encoded[['Runtime', 'IMDB_Rating','Meta_score', 'No_of_Votes', 'Year_of_release']]
#determining target variable
win = merged_data_encoded['win']
#looping through quantile ranges
for ranges in ((1.0, 99.0),(2.0,98.0),(5.0, 95.0),(10.0, 90.0),(15.0, 85.0),(25.0, 75.0)):
    #instantiate scaler
    scaler = RobustScaler(with_centering=True,
    with_scaling=True,
    quantile_range= ranges,copy=True)
    #instantiate model
    model =KNeighborsClassifier()
    #instantiate pipeline for two steps(first scaling will be done,then model fitting)
    pipeline = Pipeline(steps=[('scaler', scaler), ('model', model)])
    ##instantiate cross validation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    #getting cross validation scores
    scores = cross_val_score(pipeline, columns_to_scale, win, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    #rounding scores numbers to 4 decimal places
    scores = round(np.mean(scores),4)
    print('For quantile range:',ranges,'average score is:',scores)
"""Since accuracy of (15, 85.0) is the highest,scaling will be done with this range."""
"""Apply Robust Scale"""
columns_to_scale  = ['Runtime', 'IMDB_Rating','Meta_score', 'No_of_Votes', 'Year_of_release']
scaler = RobustScaler(with_centering=True,
    with_scaling=True,
    quantile_range=(15.0,85.0),copy=True)
merged_data_encoded[columns_to_scale] =scaler.fit_transform(merged_data_encoded[columns_to_scale])
merged_data_encoded[columns_to_scale].head()
"""Model accuracy testing after robust scaling"""
#determining independent and dependent columns for model apply
X = merged_data_encoded.drop(columns = ['win','Gross'],axis = 1)
y = merged_data_encoded['win']
models = [('LogReg', LogisticRegression(solver = 'liblinear')),
          ('RandomForest',RandomForestClassifier(n_estimators=100)),
          ('DecTree', DecisionTreeClassifier()),
          ('KNN', KNeighborsClassifier()),
          ('SVM', SVC(gamma = 'scale'))]
models_accuracy_scores(models,X,y)
"""# MODEL ACCURACY BEFORE ROBUST SCALING (AFTER DROPPING NO_OF_VOTES COLUMN)"""
#determining independent and dependent columns for model apply
X = merged_data_encoded.drop(columns = ['win','No_of_Votes'],axis = 1)
y = merged_data_encoded['win']
models = [('LogReg', LogisticRegression(solver = 'liblinear')),
          ('RandomForest',RandomForestClassifier(n_estimators=100)),
          ('DecTree', DecisionTreeClassifier()),
          ('KNN', KNeighborsClassifier()),
          ('SVM', SVC(gamma = 'scale'))]
models_accuracy_scores(models,X,y)
# ROBUST SCALING
# Getting accuracy scores of different quantile ranges"""
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

# determining independent variables
columns_to_scale = merged_data_encoded[['Runtime', 'IMDB_Rating', 'Meta_score', 'Gross', 'Year_of_release']]
# determining target variable
win = merged_data_encoded['win']
# looping in quantile ranges

for ranges in ((1.0, 99.0), (2.0, 98.0), (5.0, 95.0), (10.0, 90.0), (15.0, 85.0), (25.0, 75.0)):
    # instaantiate Robust Scaler
    scaler = RobustScaler(with_centering=True,
                          with_scaling=True,
                          quantile_range=ranges, copy=True)
    # instantiate model
    model = KNeighborsClassifier()
    # initiate pipeline for two steps(first scaling will be done,then model fitting)
    pipeline = Pipeline(steps=[('scaler', scaler), ('model', model)])
    ##instantiate cross validation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # getting cross validation scores
    scores = cross_val_score(pipeline, columns_to_scale, win, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # rounding scores numbers to 4 digit
    scores = round(np.mean(scores), 4)

    print('For quantile range:', ranges, 'average score is:', scores)
"""Apply Robust Scaler"""
columns_to_scale  = ['Runtime', 'IMDB_Rating','Meta_score', 'Gross', 'Year_of_release']
scaler = RobustScaler(with_centering=True,
    with_scaling=True,
    quantile_range=(25.0, 75.0),copy=True)
merged_data_encoded[columns_to_scale] = scaler.fit_transform(merged_data_encoded[columns_to_scale])
merged_data_encoded[columns_to_scale].head()
"""Model Performance After Robust Scaling"""
#determining independent and dependent columns for model apply
X = merged_data_encoded.drop(columns = ['win','No_of_Votes'],axis = 1)
y = merged_data_encoded['win']
models = [('LogReg', LogisticRegression(solver = 'liblinear')),
          ('RandomForest',RandomForestClassifier(n_estimators=100)),
          ('DecTree', DecisionTreeClassifier()),
          ('KNN', KNeighborsClassifier()),
          ('SVM', SVC(gamma = 'scale'))]
models_accuracy_scores(models,X,y)
"""After dropping the no_of_votes column, the model performances are higher compared to dropping the gross column.Thus,no_of_votes feature will be dropped."""
merged_data_encoded.drop(columns = 'No_of_Votes',axis = 1,inplace = True)

