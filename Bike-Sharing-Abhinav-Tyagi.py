#!/usr/bin/env python
# coding: utf-8

# <div align="center">
#     <h1> <b><u>BIKE SHARING ASSIGNMENT SUBMISSION</u></b></h1></div>
# <div align="right">    
# <h2><i>by</i>: <b>Abhinav Tyagi</b></h2>
# <h2><i>for</i>: <b>UPGRAD</b></h2>
# </div>

# <h3><b><u><center>Target</center></u></b></h3>
# After surviving the corona associated business crash, the American bike sharing company <b>BoomBikes</b> has hired us to predict forthcoming demand of the bikes to deploy the precious and now diminished resources where the returns can be highest.
# 
# 
# <h3><b><u>Required Outcome</u></b></h3>
# The company wants to know:
# <ol>
#     <li><b><i>Variables significant in predicting the demand for shared bikes</i></b></li>
# <li><b><i>How well those variables describe the bike demands</i></b></li>
# </ol>
# <div align="right"><i>let's give them what they want...</i></div>

# <hr>
# <center><div style="color:#41368F"><h1><b><u>Final Report</u></b></h1></div></center>
# 
# 
# <table>
#   <tr>
#       <td><center><b>Train R-Squared</b></center></td>
#     <td><i><div style="color:green">0.824</div></i></td>
#   </tr>
#     
#   <tr>
#       <td><center><b>Train R-Squared Adjusted</b></center></td>
#     <td><i><div style="color:green">0.821</div></i></td>
#   </tr>
#   <tr>
#       <td><center><b>Test R-Squared</b></center></td>
#     <td><i><div style="color:green">0.820</div></i></td>
#   </tr>
#     <tr>
#       <td><center><b>Test R-Squared Adjusted</b></center></td>
#     <td><i><div style="color:green">0.812</div></i></td>
#   </tr>
#     
# </table>
# 
# <br>
# <center><b>Top 3 Predictor Variables</b><center>
# <table>
#     <tr>
#         <th>Variable</th>
#         <th>Relation</th>
#     </tr>
#     
#   <tr>
#       <td><center><b>temp (Temperature)</b></center></td>
#     <td><i><div style="color:green">Per Unit Increase in yr yields bike bookings raised by 0.563615 times.</div></i></td>
#   </tr>
#     
#   <tr>
#       <td><center><b>weathersit_3 (Weather Situation 3)</b></center></td>
#     <td><i><div style="color:green">Per Unit Increase in weathersit_3 yields bike bookings decreased by -0.306992 times.</div></i></td>
#   </tr>
#   <tr>
#       <td><center><b>yr (Year)</b></center></td>
#     <td><i><div style="color:green">Per Unit Increase in yr yields bike bookings raised by 0.230846 times.</div></i></td>
#   </tr>
#     
# </table>
#     
# <center><div style="color:#77AAFC"><h3><u>Other Variables Of Importance</u></h3></div></center>
#     
# <table>
#     <tr>
#         <th>Variable</th>
#         <th>Relation</th>
#     </tr>
#     
#   <tr>
#       <td><center><b>season_4</b></center></td>
#     <td><i><div style="color:green">Per Unit Increase in season_4 yields bike bookings raised by 0.128744 times.</div></i></td>
#   </tr>
#     
#   <tr>
#       <td><center><b>windspeed</b></center></td>
#     <td><i><div style="color:green">Per Unit Increase in windspeed yields bike bookings decreased by -0.155191 times.</div></i></td>
#   </tr>
# 
#     
# </table>
#     <center><div style="color:#3647B5"><h2>Suggestions:</h2></div></center>
#     <center><i>Most important variables are temp, weathersit_3 and year. Followed by season_4 and windspeed to predict maximum bike bookings.</i></center>
# <hr>

# <b><h4>Data Loading and Basic Analysis</b></h4>

# In[1]:


#importing all the standard libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#to make our prediction model we would be using scikit learn
# RFE And Linear Regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.metrics import r2_score


# In[2]:


bike = pd.DataFrame(pd.read_csv("datasets/day.csv"))


# In[3]:


bike.describe()


# In[4]:


bike.head()


# In[5]:


bike.info()


# In[6]:


print(bike.shape)


# <div style="color:#41368F;"><h4><u><b>Initial Conclusions</b></u>:</h4></div>
# <ol>
#     <li>DataSet has,<br> <i>Rows : <u>730</u></i> <br><i> Columns : <u>16</u></i></li>
# <li>dteday is the Only Date Column</li>
# <li>Except dteday, everyother column is a continuous value, ie, int or float type.</li>
# </ol>

# <h2><u>Data Inspection and Correction</u></h2>

# <div style="color:#41368F;"><h3><i>1. Null Values</i></h3></div>

# In[7]:


print(round(100*(bike.isnull().sum()/len(bike)), 2).sort_values(ascending=False)) #Column Wise
print(round((bike.isnull().sum(axis=1)/len(bike))*100,2).sort_values(ascending=False)) #Axis switched over to rows.


# <h3> <div style="color:#41368F" align="center"><b>Conclusion</b></div></h3>
# <div style="color:#3647B5;" align="center"><u><b>No Null Values Found</b></u></div>

# <div style="color:#41368F;"><h3><i>2. Duplicates</i></h3></div>

# In[8]:


duplicates = bike.copy()
duplicates.drop_duplicates(subset=None, inplace=True)


# <div style="color:#77AAFC"><i>(If Duplicate Rows Are Present, They Are Now Dropped)</i></div>

# In[9]:


#Checking the Shape
if (duplicates.shape == bike.shape):
    print("No Change In Data After Running The Copy Of Original Data Set through Drop Duplicates Method. \nTherefore, Original Data is Free of Duplicates and Can Be Used As It Is!")
else:
    print("Shapes Dont Match, Original Data Rejected, Duplicate Needs to be Used.")


# <h3> <div style="color:#41368F" align="center"><b>Conclusion</b></div></h3>
# <div style="color:#3647B5;" align="center"><u><b>No Duplicates Found. Using Original Intact Data.</b></u></div>
# <hr>

# <div style="color:#41368F;"><h1><b>Data Cleaning</b></h1></div>

# In[10]:


# Copying data frame into a new variable without copying the column with unique values.

dummyBike = bike.iloc[:,1:16]
for column in dummyBike:
    print (dummyBike[column].value_counts(ascending=False),'\n\n\n\n') #Space Added For Greater Clarity


# <h3> <div style="color:#41368F" align="center"><b>Conclusion</b></div></h3>
# <div style="color:#3647B5;" align="center"><u><b>Dataset is quite clean. No Junk Values Found.</b></u></div>
# <hr>
# 

# <div style="color:#41368F;"><h1><b>Inconsequential Column Removal</b></h1></div>
# 
# <i>Removing the Following Columns:</i>
# <ol>
# <li><b>casual</b> <i>This column represents booking by a category of customer. A mismatch with our objective of find total count of bikes without a categorical breakdown.</i></li>
# <li><b>registered</b> <i>Again, column representing booking based on a category of people. Irrelevant</i> </li>
# <li><b>dteday</b> <i>Column representing date type. A presence of seperate columns based on day, mm, year, makes this unnecessary and therefore can be removed.</i></li>
# <li><b>instant</b> <i>Indexing variable. Removing.</i></li>
# </ol>

# In[11]:


print(bike.columns) #inpecting our columns once again.


# In[12]:


#creating a new bike variable to leave the original 
#dataset intact for future reference and cross checking and comparisions

cleanBike = bike[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed','cnt']]
cleanBike.info() #checking our new variable


# <div style="color:#41368F;"><h1><b>Encoding Categorical Variables Into Dummy Representations</b></h1></div>
# 
# <b>Categorical Columns Are:</b>
# <ol>
# <li><i>season</i></li>    
# <li><i>weekday</i></li>    
# <li><i>mnth</i></li>    
# <li><i>weathersit</i></li>        
# </ol>
# 

# In[13]:


cleanBike['weekday'] = cleanBike['weekday'].astype('category')
cleanBike['season'] = cleanBike['season'].astype('category')
cleanBike['mnth'] = cleanBike['mnth'].astype('category')
cleanBike['weathersit'] = cleanBike['weathersit'].astype('category')

cleanBike.info()


# In[14]:


cleanBike=pd.get_dummies(cleanBike, drop_first=True)
cleanBike.info()


# In[15]:


print(cleanBike.shape)


# <hr>

# <div style="color:#41368F;"><h2> Data Preparation Before Feeding to Scikit Learn.</h2></div>
# 
# <i>We will go with a 70/30 split between train/test data.</i>
# In layman's terms it would simply mean we would take our entire data and make a dataset containing 70 percent of it for training our model and then using the rest of the 30 percent data that our training model is blind to, to find the accuracy of our model.

# In[16]:


np.random.seed(0)
df_train, df_test = train_test_split(cleanBike, train_size = 0.70, test_size = 0.30, random_state = 333)


# In[17]:


print(df_train.shape)
df_train.info()


# In[18]:


print(df_test.shape)
df_test.info()


# <div style="color:#41368F;"><h2>EDA: <u>Exploratory Data Analysis On df_train Dataset</u></h2></div>

# In[19]:


df_train.info()


# In[20]:


print (df_train.columns)


# <div style="color:#77AAFC"><h4> Storing all the Continuous Data in a Single Variable Dataset</h4></div>

# In[21]:


conBike = df_train[['temp','atemp','hum','windspeed','cnt']]
sns.pairplot(conBike, diag_kind='kde')
plt.show()


# <center><div style="color:#77AAFC"><u><h3>Conclusions</h3></u></div></center>
# <center>A Linear Correlation Between cnt, temp and atemp can be observered</center>
# 

# <div style="color:#77AAFC"><h2>Categorical Variable Representation</h2></div>

# In[22]:


df_train.info()


# In[23]:


plt.figure(figsize=(30,20))
plt.subplot(3,2,1)
sns.boxplot(x='season', y='cnt', data=bike)
plt.subplot(3,2,2)
sns.boxplot(x='mnth',y='cnt',data=bike)
plt.subplot(3,2,3)
sns.boxplot(x='weathersit', y='cnt', data=bike)
plt.subplot(3,2,4)
sns.boxplot(x='holiday',y='cnt',data=bike)
plt.subplot(3,2,5)
sns.boxplot(x='weekday',y='cnt',data=bike)
plt.subplot(3,2,6)
sns.boxplot(x='workingday', y='cnt', data=bike)
plt.show()


# <center><div style="color:#77AAFC"><u><h3>Conclusions</h3></u></div></center>
# 
# <b>Number of categorical variables: 6</b>
# 
# <table>
#   <tr>
#     <th>Categorical Variable</th>
#     <th>Effect On Dependent Variable</th>
#   </tr>
#   <tr>
#       <td><center><div style="color:green">mnth</div></center></td>
#     <td><i>10% Bookings at 4000 per month seen in month 5th, 6th, 7th, 8th, and 9th. Making mnth a good predictor for the dependent variable</i></td>
#   </tr>
#     
#   <tr>
#       <td><center><div style="color:red">weekday</div></center></td>
#     <td><i>Trend of booking on each day of the week aren't well seperated, therefore inconsequential for the dependent variable.</i></td>
#   </tr>
#   <tr>
#       <td><center><div style="color:green">workingday</div></center></td>
#     <td><i>69% Bookings received on workingday, putting the median at 5000 bookings. Making this an excellent predictor.</i></td>
#   </tr>
# 
#   <tr>
#     <td><center><div style="color:red">holiday</div></center></td>
#     <td><i>97.6% Bookings received on holidays, indicating a data bias as the holidays are marked incorrectly. Can't be used therefore, as a predictor</i></td>
#   </tr>
#   <tr>
#       <td><center><div style="color:green">weathersit</div></center></td>
#     <td><i>67% Bookings observed during weathersit1, putting the median at 5000 bookings. Good predictor therefore.</i></td>
#   </tr>
#   <tr>
#       <td><center><div style="color:green">season</div></center></td>
#     <td><i>32% Bookings Observed On Season 3, 27% in season2 and 25% in season4. Indicating it being a good predictor.</i></td>
#   </tr>
# </table>

# <div style="color:#77AAFC"><h2>Correlation Matrix</h2></div>

# In[24]:


plt.figure(figsize=(25,20))
sns.heatmap(cleanBike.corr(), annot=True, cmap="RdGy")


# ## Rescaling

# In[25]:


scaler = MinMaxScaler()
print(df_train.columns)


# In[26]:


#After examination lets apply scaler() to all the continuous variables

continuous_variables = ['cnt','temp','windspeed','hum','atemp']
df_train[continuous_variables]=scaler.fit_transform(df_train[continuous_variables])


# In[27]:


df_train.head()


# In[28]:


df_train.describe()


# <div style="color:#41368F"><h2>LINEAR MODELLING</h2></div>

# In[29]:


y_train = df_train.pop('cnt')
X_train = df_train


# <div style="color:#41368F"><h3>Recursive Feature Elimination</h3></div>

# In[30]:


lm = LinearRegression()
lm.fit(X_train, y_train)
rfe = RFE(lm,n_features_to_select=15)
rfe = rfe.fit(X_train, y_train)


# In[31]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[32]:


column = X_train.columns[rfe.support_]


# In[33]:


print (column)


# In[34]:


X_train_rfe=X_train[column]


# <hr>
# <center><div style="color:#41368F"><h2>Modelling With 'STATS MODEL'</h2></div></center>

# In[35]:


# Examining VIF
vif = pd.DataFrame()
vif['Features'] = X_train_rfe.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe.values, i) for i in range(X_train_rfe.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
print(vif)


# In[36]:


X_train_lm1 = sm.add_constant(X_train_rfe)
linearRegression = sm.OLS(y_train, X_train_lm1).fit()
linearRegression.params


# In[37]:


print (linearRegression.summary())


# <center><div style="color:#77AAFC"><h3><u>Another Model</u></h3></div></center>

# In[38]:


X_train_new = X_train_rfe.drop(["atemp"], axis = 1)


# <center><div style="color:#77AAFC"><h3><u>Examining VIF</u></h3></div></center>

# In[39]:


vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
print(vif)


# In[40]:


X_train_lm2 = sm.add_constant(X_train_new)
linearRegression2 = sm.OLS(y_train, X_train_lm2).fit()
linearRegression2.params


# In[41]:


print(linearRegression2.summary())


# <center><div style="color:#77AAFC"><h2>Model 3</h2></div></center>
# <center>
# 
# <ol>
# <li>Removing hum as very high VIF value.</li>    
# <li>Retaining temp because of impact of temperature on bookings</li>    
# <ol>
#     
# </center>

# In[42]:


X_train_new = X_train_new.drop(["hum"], axis = 1)


# <center><div style="color:#77AAFC"><h3><u>Examining VIF</u></h3></div></center>

# In[43]:


vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
print(vif)


# In[44]:


X_train_lm3 = sm.add_constant(X_train_new)
#Make a fitted model
linearRegression3 = sm.OLS(y_train, X_train_lm3).fit()


# In[45]:


linearRegression3.params


# In[46]:


print(linearRegression3.summary())


# <center><div style="color:#41368F"><h2>4th Model</h2></div></center>
# <center>
# 
# <ol>
# <li>Removing season3 as very high VIF value.</li>    
# <li>Retaining temp because of impact of temperature on bookings</li>    
# <ol>
#     
# </center>

# In[47]:


X_train_new = X_train_new.drop(["season_3"], axis = 1)


# <center><div style="color:#77AAFC"><h3><u>Examining VIF</u></h3></div></center>

# In[48]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[49]:


X_train_lm4 = sm.add_constant(X_train_new)
linearRegression4 = sm.OLS(y_train, X_train_lm4).fit()
linearRegression4.params


# In[50]:


print (linearRegression4.summary())


# <center><div style="color:#41368F"><h2>5th Model</h2></div></center>
# <center>
# 
# <ol>
# <li>Removing mnth_10 as very high p-value.</li>        
# <ol>
#     
# </center>

# In[51]:


X_train_new = X_train_new.drop(["mnth_10"], axis = 1)


# <center><div style="color:#77AAFC"><h3><u>Examining VIF</u></h3></div></center>

# In[52]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[53]:


X_train_lm5 = sm.add_constant(X_train_new)
linearRegression5 = sm.OLS(y_train, X_train_lm5).fit()
linearRegression5.params
print(linearRegression5.summary())


# <center><div style="color:#41368F"><h2>6th Model</h2></div></center>
# <center>
# 
# <ol>
# <li>Removing mnth_3 as very high p-value.</li>        
# <ol>
#     
# </center>

# In[54]:


X_train_new = X_train_new.drop(["mnth_3"], axis = 1)


# <center><div style="color:#77AAFC"><h3><u>Examining VIF</u></h3></div></center>

# In[55]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[56]:


X_train_lm6 = sm.add_constant(X_train_new)
linearRegression6 = sm.OLS(y_train, X_train_lm6).fit()


# In[57]:


linearRegression6.params


# In[58]:


print(linearRegression6.summary())


# <center><div style="color:#77AAFC"><u><h3>Conclusions</h3></u></div></center>
# <center><b>Observable Distinctions of model:</b>
#     p-values : <i>Significant</i>
#     Multicollinearity Between Predictors : <i>Low</i>
#     <br>
#     <b>Inference : Model is Very Good. Chosen As Our Final Model</b>
# </center>
# 

# <div style="color:#41368F"><h2><u>Final Observations <i>(linearRegression6)</i></u></h2></div>
# 
# <div style="color:#3647B5"><h4><b>Coefficient Values</b></h4></div>
# <table>
#   <tr>
#     <th>Variable</th>
#     <th>Coefficient Values</th>
#     <th>Interpretations</th>
#   </tr>
#     
#     
#   <tr>
#       <td><center>yr</center></td>
#     <td><i><div style="color:green">0.230846</div></i></td>
#     <td><i><div style="color:#3647B5">Per Unit Increase in yr yields bike bookings raised by 0.230846 times.</div></i></td>
#   </tr>
#     
#   <tr>
#       <td><center>const</center></td>
#     <td><i><div style="color:green">0.084143</div></i></td>
#       <td><i><div style="color:#3647B5">Per Unit Increase in yr yields bike bookings raised by 0.084143 times.</div></i></td>
#   </tr>
#     
#     
#     
#   <tr>
#       <td><center>workingday</center></td>
#     <td><i><div style="color:green">0.043203</div></i></td>
#       <td><i><div style="color:#3647B5">Per Unit Increase in yr yields bike bookings raised by 0.043203 times.</div></i></td>
#   </tr>
#   <tr>
#       <td><center>temp</center></td>
#     <td><i><div style="color:green">0.563615</div></i></td>
#       <td><i><div style="color:#3647B5">Per Unit Increase in yr yields bike bookings raised by 0.563615 times.</div></i></td>
#   </tr>
# 
#   <tr>
#     <td><center>windspeed</center></td>
#     <td><i><div style="color:green">-0.155191</div></i></td>
#       <td><i><div style="color:#3647B5">Per Unit Increase in windspeed yields bike bookings decreased by -0.155191 times.</div></i></td>
#   </tr>
#   <tr>
#       <td><center>season_2</center></td>
#     <td><i><div style="color:green">0.082706</div></i></td>
#       <td><i><div style="color:#3647B5">Per Unit Increase in season_2 yields bike bookings raised by 0.082706 times.</div></i></td>
#   </tr>
#      <tr>
#       <td><center>season_4</center></td>
#     <td><i><div style="color:green">0.128744</div></i></td>
#          <td><i><div style="color:#3647B5">Per Unit Increase in season_4 yields bike bookings raised by 0.128744 times.</div></i></td>
#   </tr>
#     <tr>
#       <td><center>mnth_9</center></td>
#     <td><i><div style="color:green">0.094743</div></i></td>
#         <td><i><div style="color:#3647B5">Per Unit Increase in mnth_9 yields bike bookings raised by 0.094743 times.</div></i></td>
#   </tr>
#     <tr>
#       <td><center>weekday_6</center></td>
#     <td><i><div style="color:green">0.056909</div></i></td>
#         <td><i><div style="color:#3647B5">Per Unit Increase in weekday_6 yields bike bookings raised by 0.056909 times.</div></i></td>
#   </tr> 
#     <tr>
#       <td><center>weathersit_2</center></td>
#     <td><i><div style="color:green">-0.074807</div></i></td>
#         <td><i><div style="color:#3647B5">Per Unit Increase in yr yields bike bookings decreased by -0.074807 times.</div></i></td>
#   </tr> 
#     <tr>
#       <td><center>weathersit_3</center></td>
#     <td><i><div style="color:green">-0.306992</div></i></td>
#         <td><i><div style="color:#3647B5">Per Unit Increase in weathersit_3 yields bike bookings decreased by -0.306992 times.</div></i></td>
#   </tr>
# </table>
# 
# <div style="color:#77AAFC"><h3>Conclusions</h3></div>
# No cofficients = 0,
# Therefore, All Null Hypothesis stands rejected.
# 
# 
# <br><br>
# 
# <div style="color:#3647B5"><h3><b>F Statistics</b></h3></div></center>
# 
# <table>
#   <tr>
#       <td><center>F-statistic</center></td>
#     <td><i><div style="color:green">233.8</div></i></td>
#   </tr>
#     
#   <tr>
#       <td><center>Prob (F-statistic)</center></td>
#     <td><i><div style="color:green">3.77e-181</div></i></td>
#   </tr>
# 
# </table>
# 
# A higher F-Stastics value corresponds to a more signficant Model.
# 
# Here, with an F-statistic of 233.8 and p-Value of 0.00, it is clear that model is significant.
# 
# <br><br>
# 
# <div style="color:#3647B5"><h3><b>Best fitted surface equation</b></h3></div></center>
# 
# <i>cnt = 0.084143 + (**yr** × 0.230846) + (**workingday** × 0.043203) + (**temp** × 0.563615) − (**windspeed** × 0.155191) + (**season2** × 0.082706) + (**season4** ×0.128744) + (**mnth9** × 0.094743) + (**weekday6** ×0.056909) − (**weathersit2** × 0.074807) − (**weathersit3** × 0.306992)</i>
# 
# 

# 

# In[59]:


y_train_pred = linearRegression6.predict(X_train_lm6)
res = y_train-y_train_pred
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((res), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)


# <center><div style="color:#3647B5"><h3><b><u>Conclusions:</u></b></h3></div></center>
# <center>Residuals are normally distributed. <br><i>Linear Regression Stands Validated.</i></center>

# In[60]:


cleanBike=cleanBike[[ 'temp', 'atemp', 'hum', 'windspeed','cnt']]

sns.pairplot(bike_num, diag_kind='kde')
plt.show()


# <center><div style="color:#3647B5"><h3><b><u>Conclusions</u></b></h3></div></center>
# <center>Linear Relationship between temp and atemp variables observed with predictor <i>cnt</i></i></center>
# 
# <center><b><div style="color:#77AAFC"><h3>No Observable MultiCollinearity between the Predictor Variables</h3></div></b></center>

# In[ ]:


#Another VIF Exam
vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# <center><div style="color:#3647B5"><h3><b><u>Conclusions</u></b></h3></div></center>
# <center><i>Absence of Multicollinearity between the predictor variables, as values are consistently below 5</i></center>

# <center><div style="color:#41368F"><h1><b><u>Final Prediction with Our Chosen Model</u></b></h1></div></center>

# In[ ]:


num_vars = ['temp', 'atemp', 'hum', 'windspeed','cnt']
df_test[num_vars] = scaler.transform(df_test[num_vars])
df_test.head()


# In[ ]:


df_test.describe()


# <center><div style="color:#3647B5"><h3><b><u>Splitting Into X_test and y_test</u></b></h3></div></center>

# In[ ]:


y_test = df_test.pop('cnt')
X_test = df_test
x_test.info()


# In[ ]:


column1 = X_train_new.columns
X_test=X_test[col1]
X_test_lm6 = sm.add_constant(X_test)
X_test_lm6.info()


# In[ ]:


y_pred=linearRegression6.predict(x_test_lm6)


# <center><div style="color:#41368F"><h1><b><u>Evaluating Model</u></b></h1></div></center>

# In[ ]:


fig = plt.figure()
plt.scatter(y_test, y_pred, alpha = .6)
fig.suptitle('y_test VS y_pred', fontsize = 21)
plt.xlabel('y_test',fontsize = 19)
plt.ylabel('y_pred', fontsize=17)
plt.show()


# <center><div style="color:#41368F"><h1><b><u>R-Squared Value of test</u></b></h1></div></center>

# In[ ]:


r2_score(y_test, y_pred)


# In[ ]:


#Adjusted R-Sqaured Value

r2= 0.8203092200749708
X_test.shape


# In[ ]:


n=X_test.shape[0]
p=X_test.shape[1]

adjusted_r2 = 1-(1-r2)+(n-1)/(n-p-1)
print (adjusted_r2)


# <hr>
# <center><div style="color:#41368F"><h1><b><u>Report</u></b></h1></div></center>
# 
# 
# <table>
#   <tr>
#       <td><center><b>Train R-Squared</b></center></td>
#     <td><i><div style="color:green">0.824</div></i></td>
#   </tr>
#     
#   <tr>
#       <td><center><b>Train R-Squared Adjusted</b></center></td>
#     <td><i><div style="color:green">0.821</div></i></td>
#   </tr>
#   <tr>
#       <td><center><b>Test R-Squared</b></center></td>
#     <td><i><div style="color:green">0.820</div></i></td>
#   </tr>
#     <tr>
#       <td><center><b>Test R-Squared Adjusted</b></center></td>
#     <td><i><div style="color:green">0.812</div></i></td>
#   </tr>
#     
# </table>
# 
# <br>
# <center><b>Top 3 Predictor Variables</b><center>
# <table>
#     <tr>
#         <th>Variable</th>
#         <th>Relation</th>
#     </tr>
#     
#   <tr>
#       <td><center><b>temp (Temperature)</b></center></td>
#     <td><i><div style="color:green">Per Unit Increase in yr yields bike bookings raised by 0.563615 times.</div></i></td>
#   </tr>
#     
#   <tr>
#       <td><center><b>weathersit_3 (Weather Situation 3)</b></center></td>
#     <td><i><div style="color:green">Per Unit Increase in weathersit_3 yields bike bookings decreased by -0.306992 times.</div></i></td>
#   </tr>
#   <tr>
#       <td><center><b>yr (Year)</b></center></td>
#     <td><i><div style="color:green">Per Unit Increase in yr yields bike bookings raised by 0.230846 times.</div></i></td>
#   </tr>
#     
# </table>
#     
# <center><div style="color:#77AAFC"><h3><u>Other Variables Of Importance</u></h3></div></center>
#     
# <table>
#     <tr>
#         <th>Variable</th>
#         <th>Relation</th>
#     </tr>
#     
#   <tr>
#       <td><center><b>season_4</b></center></td>
#     <td><i><div style="color:green">Per Unit Increase in season_4 yields bike bookings raised by 0.128744 times.</div></i></td>
#   </tr>
#     
#   <tr>
#       <td><center><b>windspeed</b></center></td>
#     <td><i><div style="color:green">Per Unit Increase in windspeed yields bike bookings decreased by -0.155191 times.</div></i></td>
#   </tr>
# 
#     
# </table>
#     <center><div style="color:#3647B5"><h2>Suggestions:</h2></div></center>
#     <center><i>Most important variables are temp, weathersit_3 and year. Followed by season_4 and windspeed to predict maximum bike bookings.</i></center>
# <hr>

# <center><div style="color:#41368F"><h1>THANK YOU!</h1></div></center>
