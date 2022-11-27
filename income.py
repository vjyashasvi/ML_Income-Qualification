#!/usr/bin/env python
# coding: utf-8

#   # Income Qualification

# Many social programs have a hard time ensuring that the right people are given enough aid. It’s tricky when a program focuses on the poorest segment of the population. This segment of the population can’t provide the necessary income and expense records to prove that they qualify.
# 
# In Latin America, a popular method called Proxy Means Test (PMT) uses an algorithm to verify income qualification. With PMT, agencies use a model that considers a family’s observable household attributes like the material of their walls and ceiling or the assets found in their homes to
# classify them and predict their level of need.
# 
# While this is an improvement, accuracy remains a problem as the region’s population grows and poverty declines.
# 
# The Inter-American Development Bank (IDB)believes that new methods beyond traditional econometrics, based on a dataset of Costa Rican household characteristics, might help improve PMT’s performance.
# 
# Identify the level of income qualification needed for the families in Latin America.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import warnings 
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv('train_inc.csv')
df_test=pd.read_csv('test_inc.csv')
df


# # Identify the output variable.

# In[3]:


df['Target']


# # Understand the type of data.

# In[4]:


df.dtypes


# # Check if there are any biases in your dataset.

# In[5]:


dfn=df.select_dtypes(include=np.number)


# In[6]:


dfn


# In[7]:


dfc=df.select_dtypes(exclude=np.number)
dfc


# In[8]:


def unique_num(var):
    print(var,'\n')
    print(dfc[var].unique())


# In[9]:


l=['dependency','edjefe','edjefa']
for i in range(3):
    unique_num(l[i])


# In[10]:


def map(i):
    
    if i=='yes':
        return(float(1))
    if i=='no':
        return(float(0))
    else:
        return(float(i))


# In[11]:


l=['dependency','edjefe','edjefa']
for i in range(3):
    df[l[i]]=df[l[i]].apply(map)
    df_test[l[i]]=df_test[l[i]].apply(map)


# In[12]:


df[['dependency','edjefe','edjefa']]


# In[13]:


df_test[['dependency','edjefe','edjefa']]


# ### Below is Data dictionary for above object variables
# 
# *dependency, Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)
# 
# *edjefe, years of education of male head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0
# 
# *edjefa, years of education of female head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0

# In[14]:


df.dtypes[df.dtypes=='object']


#  #### Lets identify variable with 0 varinace

# In[15]:


var_df=pd.DataFrame(np.var(df,0),columns=['variance'])
var_df.sort_values(by='variance').head(15)
print('Below are columns with variance 0.')
col=list((var_df[var_df['variance']==0]).index)
print(col)


# In[16]:


df['elimbasu5'].nunique()


# From above it is shown that all values of elimbasu5 is same so there is no variablity in dataset therefor we will drop this variable

# In[ ]:





# In[ ]:





# In[ ]:





# In[17]:


plt.figure(figsize=(19,18))
sns.heatmap(df.corr())


# In[18]:


def unique_nums(var):
    if((df[var].nunique())>=155):
        print(var)
        print(df[var].nunique(),'\n')


# In[19]:


l=list(df.columns)
for i in range(len(l)):
    unique_nums(l[i])
    


# Below is Data dictionary for above object variables
# 
# *ID = Unique ID
# 
# *v2a1, Monthly rent payment
# 
# *meaneduc,average years of education for adults (18+)
# 
# *SQBmeaned, square of the mean years of education of adults (>=18) in the
# household
# 
# *idhogar, Household level identifier

# In[20]:


df.info()


# In[21]:


df.select_dtypes('object').head()


# In[22]:


na_counts=df.isna().sum()
na_counts[na_counts>0]


# In[23]:


# 1. Lets look at v2a1 (total nulls: 6860) : Monthly rent payment 
# why the null values, Lets look at few rows with nulls in v2a1
# Columns related to  Monthly rent payment
# tipovivi1, =1 own and fully paid house
# tipovivi2, "=1 own,  paying in installments"
# tipovivi3, =1 rented
# tipovivi4, =1 precarious 
# tipovivi5, "=1 other(assigned,  borrowed)"


# In[24]:


data=df[df['v2a1'].isna()]
columns=['tipovivi1','tipovivi2','tipovivi3','tipovivi4','tipovivi5']
data[columns]


# In[25]:


own_variables = [x for x in df if x.startswith('tipo')]
own_variables


# In[26]:


df.loc[df['v2a1'].isnull(),own_variables].sum()


# In[27]:


# Plot of the home ownership variables for home missing rent payments
df.loc[df['v2a1'].isnull(), own_variables].sum().plot.bar(figsize = (10, 8),
                                                                        color = 'lime',edgecolor = 'aqua',
                                                               linewidth = 2).set_facecolor('black')
plt.xticks([0, 1, 2, 3, 4],['Owns and Paid Off', 'Owns and Paying', 'Rented', 'Precarious', 'Other'],
          rotation = 20)
plt.title('Home Ownership Status for Households Missing Rent Payments', size = 18);


# In[28]:


#Looking at the above data it makes sense that when the house is fully paid, there will be no monthly rent payment.
#Lets add 0 for all the null values.

df['v2a1']= df['v2a1'].fillna(0)
df_test['v2a1']= df_test['v2a1'].fillna(0)

df[['v2a1']].isnull().sum()
   


# In[29]:


# 2. Lets look at v18q1 (total nulls: 7342) : number of tablets household owns 
# why the null values, Lets look at few rows with nulls in v18q1
# Columns related to  number of tablets household owns 
# v18q, owns a tablet


# In[30]:


#Since this is a household variable, it only makes sense to look at it on a household level, 
# so we'll only select the rows for the head of household.

# Heads of household
heads = df.loc[df['parentesco1'] == 1].copy()
heads.groupby('v18q')['v18q1'].apply(lambda x: x.isnull().sum())


# In[31]:


h = df.loc[df['parentesco1'] == 1].copy()
h.groupby('v18q')['v18q1'].apply(lambda x: x.isnull().sum())


# In[32]:


plt.figure(figsize = (8, 6))
col='v18q1'
df[col].value_counts().sort_index().plot.bar(color = 'blue',
                                             edgecolor = 'aqua',
                                             linewidth = 2)
plt.xlabel(f'{col}'); plt.title(f'{col} Value Counts'); plt.ylabel('Count')
plt.show();


# In[33]:


#Looking at the above data it makes sense that when owns a tablet column is 0, there will be no number of tablets household owns.
#Lets add 0 for all the null values.
df['v18q1']=df['v18q1'].fillna(0)
df_test['v18q1']=df_test['v18q1'].fillna(0)

df[['v18q1']].isnull().sum()


# In[34]:


# 3. Lets look at rez_esc    (total nulls: 7928) : Years behind in school  
# why the null values, Lets look at few rows with nulls in rez_esc
# Columns related to Years behind in school 
# Age in years

# Lets look at the data with not null values first.
df[df['rez_esc'].notnull()]['age'].describe()


# In[35]:


#From the above , we see that when min age is 7 and max age is 17 for Years, then the 'behind in school' column has a value.
#Lets confirm
df.loc[df['rez_esc'].isnull()]['age'].describe()


# In[36]:


df.loc[df['rez_esc'].isnull()]['rez_esc'].nunique()


# In[37]:


df[df['rez_esc'].notnull()]['age'].nunique()


# In[38]:


df[(df['age']>=7) & (df['age']<=17) & (df['rez_esc'].isnull())]


# In[39]:


#there is only one member in household for the member with age 10 and who is 'behind in school'.


# In[40]:


df['rez_esc']=df['rez_esc'].fillna(0)
df_test['rez_esc']=df_test['rez_esc'].fillna(0)

df[['rez_esc']].isnull().sum()


# In[41]:


df_test[['rez_esc']].isnull().sum()


# In[42]:


#Lets look at meaneduc   (total nulls: 5) : average years of education for adults (18+)  
# why the null values, Lets look at few rows with nulls in meaneduc
# Columns related to average years of education for adults (18+)  
# edjefe, years of education of male head of household, based on the interaction of escolari (years of education),
#    head of household and gender, yes=1 and no=0
# edjefa, years of education of female head of household, based on the interaction of escolari (years of education), 
#    head of household and gender, yes=1 and no=0 
# instlevel1, =1 no level of education
# instlevel2, =1 incomplete primary 


# In[43]:


data=df[df['meaneduc'].isnull()]
columns=['edjefe','edjefa','instlevel1','instlevel2']
data[columns][data[columns]['instlevel1']>0]


# In[44]:


#from the above, we find that meaneduc is null when no level of education is 0
#Lets fix the data
df['meaneduc']=df['meaneduc'].fillna(0)
df_test['meaneduc']=df_test['meaneduc'].fillna(0)
df_test['meaneduc'].isnull().sum()    


# In[45]:


#Lets look at SQBmeaned  (total nulls: 5) : square of the mean years of education of adults (>=18) in the household 142  
# why the null values, Lets look at few rows with nulls in SQBmeaned
# Columns related to average years of education for adults (18+)  
# edjefe, years of education of male head of household, based on the interaction of escolari (years of education),
#    head of household and gender, yes=1 and no=0
# edjefa, years of education of female head of household, based on the interaction of escolari (years of education), 
#    head of household and gender, yes=1 and no=0 
# instlevel1, =1 no level of education
# instlevel2, =1 incomplete primary 


# In[46]:


data = df[df['SQBmeaned'].isnull()].head()

columns=['edjefe','edjefa','instlevel1','instlevel2']
data[columns][data[columns]['instlevel1']>0].describe()


# In[47]:


#from the above, we find that SQBmeaned is null when no level of education is 0
#Lets fix the data
df['SQBmeaned']=df['SQBmeaned'].fillna(0)
df_test['SQBmeaned']=df_test['SQBmeaned'].fillna(0)
df[['SQBmeaned']].isnull().sum()


# In[48]:


#Lets look at the overall data
null_counts = df.isnull().sum()
null_counts[null_counts > 0].sort_values(ascending=False)


# In[49]:


df[['idhogar','Target']]


# In[50]:


all_equal= df.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
all_equal


# In[51]:


# Households where targets are not all equal
not_equal = all_equal[all_equal != True]
print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))


# In[52]:


not_equal.index


# In[53]:


#Lets check one household
df[df['idhogar'] == not_equal.index[0]][['idhogar', 'parentesco1', 'Target']]


# In[54]:


#Lets use Target value of the parent record (head of the household) and update rest. But before that lets check
# if all families has a head. 

households_head = df.groupby('idhogar')['parentesco1'].sum()

# Find households without a head
households_no_head = df.loc[df['idhogar'].isin(households_head[households_head == 0].index), :]

print('There are {} households without a head.'.format(households_no_head['idhogar'].nunique()))


# In[55]:


# Find households without a head and where Target value are different
households_no_head_equal = households_no_head.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
print('{} Households with no head have different Target value.'.format(sum(households_no_head_equal == False)))


# In[56]:


#Lets fix the data
#Set poverty level of the members and the head of the house within a family.
# Iterate through each household
for household in not_equal.index:
    # Find the correct label (for the head of household)
    true_target = int(df[(df['idhogar'] == household) & (df['parentesco1'] == 1.0)]['Target'])
    
    # Set the correct label for all members in the household
    df.loc[df['idhogar'] == household, 'Target'] = true_target
    
    
# Groupby the household and figure out the number of unique values
all_equal = df.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

# Households where targets are not all equal
not_equal = all_equal[all_equal != True]
print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))


# ## Lets check for any bias in the dataset

# In[57]:


#Lets look at the dataset and plot head of household and Target
# 1 = extreme poverty 2 = moderate poverty 3 = vulnerable households 4 = non vulnerable households 
target_counts = heads['Target'].value_counts().sort_index()
target_counts


# In[58]:


target_counts.plot.bar(figsize = (8, 6),linewidth = 2,edgecolor = 'cyan',title="Target vs Total_Count")


# In[59]:


# extreme poverty is the smallest count in the train dataset. The dataset is biased.


# Lets look at the Squared Variables
# ‘SQBescolari’
# ‘SQBage’
# ‘SQBhogar_total’
# ‘SQBedjefe’
# ‘SQBhogar_nin’
# ‘SQBovercrowding’
# ‘SQBdependency’
# ‘SQBmeaned’
# ‘agesq’

# In[60]:


#Lets remove them
print(df.shape)
cols=['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 
        'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']


df_test=df_test.drop(columns = cols)
df=df.drop(columns = cols)

print(df.shape)


# In[61]:


id_ = ['Id', 'idhogar', 'Target']

ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 
            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 
            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 
            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 
            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 
            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 
            'instlevel9', 'mobilephone']

ind_ordered = ['rez_esc', 'escolari', 'age']

hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 
           'paredpreb','pisocemento', 'pareddes', 'paredmad',
           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 
           'pisonatur', 'pisonotiene', 'pisomadera',
           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 
           'abastaguadentro', 'abastaguafuera', 'abastaguano',
            'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 
           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',
           'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 
           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 
           'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',
           'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 
           'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 
           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',
           'lugar4', 'lugar5', 'lugar6', 'area1', 'area2']

hh_ordered = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 
              'r4t3', 'v18q1', 'tamhog','tamviv','hhsize','hogar_nin',
              'hogar_adul','hogar_mayor','hogar_total',  'bedrooms', 'qmobilephone']

hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding']


# In[62]:


#Check for redundant household variables
heads = df.loc[df['parentesco1'] == 1, :]
heads = heads[id_ + hh_bool + hh_cont + hh_ordered]
heads.shape


# In[63]:


# Create correlation matrix
corr_matrix = heads.corr()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]

to_drop


# In[64]:


corr_matrix.loc[corr_matrix['tamhog'].abs() > 0.9, corr_matrix['tamhog'].abs() > 0.9]


# In[65]:


sns.heatmap(corr_matrix.loc[corr_matrix['tamhog'].abs() > 0.9, corr_matrix['tamhog'].abs() > 0.9],
            annot=True, cmap = 'gist_rainbow', fmt='.3f');


# In[66]:


# There are several variables here having to do with the size of the house:
# r4t3, Total persons in the household
# tamhog, size of the household
# tamviv, number of persons living in the household
# hhsize, household size
# hogar_total, # of total individuals in the household
# These variables are all highly correlated with one another.


# In[67]:


cols=['tamhog', 'hogar_total', 'r4t3']
df_test=df_test.drop(columns = cols) 
df=df.drop(columns = cols)

df.shape


# In[68]:


#Check for redundant Individual variables
ind = df[id_ + ind_bool + ind_ordered]
ind.shape


# In[69]:


# Create correlation matrix
corr_matrix = ind.corr()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]

to_drop


# In[70]:


# This is simply the opposite of male! We can remove the male flag.
df_test=df_test.drop(columns = 'male')
df=df.drop(columns = 'male')

df.shape


# In[71]:


#lets check area1 and area2 also
# area1, =1 zona urbana 
# area2, =2 zona rural 
#area2 redundant because we have a column indicating if the house is in a urban zone

df_test=df_test.drop(columns = 'area2')
df=df.drop(columns = 'area2')

df.shape


# In[72]:


#Finally lets delete 'Id', 'idhogar'
cols=['Id','idhogar']
df_test=df_test.drop(columns = cols)
df=df.drop(columns = cols)

df.shape


# #  Predict the accuracy using random forest classifier.

# In[73]:


x_features=df.iloc[:,0:-1]
y_features=df.iloc[:,-1]
print(x_features.shape)
print(y_features.shape)


# In[74]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,classification_report

x_train,x_test,y_train,y_test=train_test_split(x_features,y_features,test_size=0.2,random_state=1)
rmclassifier = RandomForestClassifier()


# In[75]:


rmclassifier.fit(x_train,y_train)


# In[76]:


y_predict = rmclassifier.predict(x_test)


# In[77]:


print(accuracy_score(y_test,y_predict))
print(confusion_matrix(y_test,y_predict))
print(classification_report(y_test,y_predict))


# In[78]:


y_predict_testdata = rmclassifier.predict(df_test)


# In[79]:


y_predict_testdata


# In[ ]:




