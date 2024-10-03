#!/usr/bin/env python
# coding: utf-8

# # LEVEL-1 

# ## Task-1: Data Exploration and Preprocessing
# ### What is data exploration and Data Pre-Processing
# #### -Data exploration refers to the initial step in data analysis in which data analysts use data visualization and statistical techniques to describe dataset characterizations, such as size, quantity, and accuracy, in order to better understand the nature of the data.
# #### -Data Preprocessing-refers to the cleaning, transforming, and integrating of data in order to make it ready for analysis. The goal of data preprocessing is to improve the quality of the data

# ## 1-Reading CSV File. 

# In[1]:


import pandas as pd 
csv_1=pd.read_csv("D:\Internship\dataset.csv")
csv_1


# #### -No. of rows =9551
# #### -No of columns = 21

# ## 2- Checking no  of Missing Values in each Column 

# In[2]:


missing_values = csv_1.isnull().sum()

# Display the result
print("Missing values in each column:")
print(missing_values)


# ### -Hence there is a missing value in column cuisines
# ###  Cousines - number of missing values =9
# #### So we can ignore it or reolace it by not specified

# In[3]:


csv_1['Cuisines'].fillna('Not Specified', inplace= True)


# In[4]:


# lets check it again

csv_1.isnull().sum()


# ### Now there is no missing values in the data set 

# In[5]:


## check for duplicate

dup = csv_1.duplicated().sum()
print(f'Number of duplicate rows are {dup}')


#  

# In[6]:


csv_1.head()


# In[7]:


csv_1.tail()


# ### To find no of rows and columns 

# In[8]:


csv_1.shape


# # 3-  Now we have to perform data conversion if needed. Analyze the distribution of target variable [,Aggregate Rating'] and identify any class imbalance

# In[9]:


csv_1.info()


# ### So , no need to any data conversion 

# In[10]:


# Targhet variable [" Aggregate Rating"]

target = "Aggregate rating"

# Descriptive statistics

print(csv_1[target].describe())


# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


plt.figure(figsize=(8, 5))
sns.boxplot(x=csv_1[target])
plt.title('Box Plot')
plt.xlabel('Aggregate rating')
plt.show()


# In[13]:


# Histogram
plt.figure(figsize=(8, 5))
sns.histplot(x=csv_1[target],bins =30,kde =True , color ='Skyblue')
plt.title('Histogram')
plt.xlabel('Aggregate rating')
plt.ylabel('frequency')
plt.show()


# ### No class imbalance 

# # Level 1-Task 2

# ## Task : Descriptive Analysis 
# ### Calculate basic statistical measure (mean ,median ,SD etc) for numerical columns

# In[14]:


csv_1.describe()


# ### 2-Explore the distribution of categorical variable like "Country Code", "City","Cuisines" . Identify the top city and Cuisines whit highest number of restaurents.

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


# explore the distribution of "country code"

plt.figure(figsize=(8, 5))
sns.countplot(x= 'Country Code', data=csv_1,palette ='cividis')

plt.title('Distribution of restaurent by  country code')
plt.xlabel('Country Code')
plt.ylabel('Number of restaurents')
plt.show()


# ### The majority of restaurents are located  om Country Code 1 followed by second  highest concentration in country code 216

# In[17]:


top_countries = csv_1["Country Code"].value_counts().head()
print('Top 5 countries with the highest number of restaurants: ')
print(top_countries)


# In[18]:


plt.figure(figsize=(15, 6))

# Correct method name value_counts and correct variable naming for order
sns.countplot(x='City', data=csv_1, order=csv_1['City'].value_counts().head(20).index, palette='Set2')

plt.title('Distribution of restaurants by city')
plt.xlabel('City')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[19]:


plt.figure(figsize=(15, 6))

# Correct method name value_counts and correct variable naming for order
Cuisines_count = csv_1['Cuisines'].value_counts()
Cuisines_count.head(20).plot(kind='bar', color=sns.color_palette("Set2"))

plt.title('Tp 20 cisines with highest number of restaurents')
plt.xlabel('Cuisines')
plt.ylabel('number of restaurents')
plt.xticks(rotation=45)
plt.show()


# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming cuisines_count is a Series with the count of cuisines
cuisines_count = csv_1['Cuisines'].value_counts()  # Replace 'Cuisine' with the actual column name if different

plt.figure(figsize=(15, 6))
cuisines_count.head(20).plot(kind='bar', color=sns.color_palette('Set2'))

plt.title('Top 20 Cuisines Distribution')
plt.xlabel('Cuisine')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# ## Top cuisines and Cities 

# In[21]:


top_cities = csv_1['City'].value_counts().head(10)
print('topo 10  cities with highest number of restaurents :')
print(top_cities)


# In[22]:


top_cuisines = cuisines_count.head(10)
print('top 10 cuisines with highest number of restaurents:')
print(top_cuisines)


# # Task-3 

#  ## Geo-Spatial Analysis
#  
#  ### 1-Visualise the locations of the restaurents on map using latitude and longitude information.

# In[23]:


# Location of restaurents on a map using latitude and longitudesinformation
#import tyhe necessary libraries

from shapely.geometry import point
import geopandas as gpd
from geopandas import GeoDataFrame


# In[24]:


from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt

# Assuming csv_1 is your DataFrame with 'Latitude' and 'Longitude' columns
# Replace 'Latitude' and 'Longitude' with the actual column names if different

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(
    csv_1, geometry=gpd.points_from_xy(csv_1['Longitude'], csv_1['Latitude'])
)

# Plot the locations of restaurants
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

fig, ax = plt.subplots(figsize=(15, 10))
world.plot(ax=ax, color='lightgrey')

gdf.plot(ax=ax, markersize=10, color='red', alpha=0.5)
plt.title('Restaurant Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# In[25]:


pip install shapely


# In[26]:


pip install geopandas


# In[27]:


# Create Point geomtry from latitude and longitude using Shapely
gdf = gpd.GeoDataFrame(csv_1, geometry=gpd.points_from_xy(csv_1.Longitude, csv_1.Latitude))

# Create a base map of the world using Geopandas 
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowers'))

# Create a map that fits the screen and plots the restaurant locations
gdf.plot(ax=world.plot("continent",legend = True, figsize=(18, 15)), marker= '0', color'red', markersize= 15)

plt.show()


# In[ ]:


plt.show


# ## 2-Analyze thedistribution of restaurents across different cities or countries. determine if there is any correlation between the restaurents location and its rating 

# In[ ]:


plt.figure(figsize=(8, 5))
sns.countplot(y = csv_1['City'],order = csv_1.City.value_counts().head(10).index,palette='Set2')


plt.xlabel('No of restaurents')
plt.ylabel('No of cities')
plt.title('Distribution of restaurents across cities')
plt.show()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'csv_1' is your DataFrame
sns.countplot(y=csv_1['City'], order=csv_1['City'].value_counts().head(10).index, palette='Set2')

correlation_matrix = csv_1[['Latitude', 'Longitude', 'Aggregate rating']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

plt.title('Correlation between restaurants located at rating')
plt.show()


# ###  OBSERVATION:-
# ##### -The Restaurents dataset has various attributes such as Restaurents Ids,name,city, country,types of cuisines.
# ##### -There are 9551 rows and 21 columns
# ##### -In this dataset9 missing values from "Cuisines"column.so it can be replaced by non specified.
# ##### -In dataset no duplicates are present.
# ##### -Top cuisines are "North india","Chinese","Fast Food"
# ##### -USA and India has most number of restsaurents

# # Level 2: Task-1 

# ##  Task-1: Table Booking and Online Delivery
#  ### -Determine the percentage of restaurants that offer table booking and online delivery.
#  ### -Compare the average ratings of restaurants with table booking and those without.
#  ### -Analyze the availability of online delivery among restaurants with different price ranges 
# 
# ## Task-2: Price Range Analysis
#  ### -Determine the most common price range among all the restaurants.
#  ### -Calculate the average rating for each price range.
#  ### -Identify the color that represents the highest average rating among different price ranges.  
#  
#  ##  Task-3: Feature Engineering
#  ### -Extract additional features from the existing columns, such as the length of the restaurant name or address.
#  ### -Create new features like "Has Table Booking" or "Has Online Delivery" by encoding categorical variables 

# In[28]:


# Importing the warnings

import warnings
warnings.filterwarnings("ignore")


# In[29]:


# Importing libraries which we are going to use in EDA.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;sns.set(color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Level 2-Task 1:-
# 
# ### Task: Table Booking and Online Delivery
# 
# #### Determine the percentage of restaurents that offer table booking and online delivery. 

# In[30]:


# Print column names to check for discrepancies
print(csv_1.columns)

# Strip any leading/trailing spaces from column names
csv_1.columns = csv_1.columns.str.strip()

# Convert column names to lowercase for consistency (optional)
csv_1.columns = csv_1.columns.str.lower()

# Use the correct column name (adjust based on your actual column names)
try:
    print(csv_1["has table booking"].value_counts())
except KeyError:
    print("The column 'Has Table Booking' does not exist in the DataFrame.")


# In[31]:


# Print the list of columns to debug
print(csv_1.columns)

# Clean the column names
csv_1.columns = csv_1.columns.str.strip()  # Strip leading/trailing spaces
csv_1.columns = csv_1.columns.str.lower()  # Convert to lowercase for uniformity

# Print the cleaned column names to verify
print(csv_1.columns)

# Use the correct column name based on cleaned columns
try:
    print(csv_1["has online delivery"].value_counts())  # Adjust based on actual column name
except KeyError:
    print("The column 'Has Online delivery' does not exist in the DataFrame.")


# In[32]:


print("Table Booking : ",round((1158/(8393+1158)) *100, 2),"%")
print("Online Delivery : ",round((2451/(7100+2451)) *100, 2),"%")


# ### 2- Compare the avarage reting of restaurentswith table booking and those without  

# In[34]:


# Step 1: Print the column names to see if 'Has Table Booking' exists and is correctly named
print(csv_1.columns)

# Step 2: Strip any leading or trailing whitespace from the column names
csv_1.columns = csv_1.columns.str.strip()

# Step 3: Try to access the column again
print(csv_1.columns)  # Print again to verify the changes

# Now filter the DataFrame
if 'Has Table Booking' in csv_1.columns:
    csv_with_table_booking = csv_1[csv_1['Has Table Booking'] == 'Yes']
    csv_without_table_booking = csv_1[csv_1['Has Table Booking'] == 'No']
else:
    print("The column 'Has Table Booking' does not exist in the DataFrame.")
    


# In[36]:


csv_1.columns = csv_1.columns.str.strip()

# Print the column names to verify
print(csv_1.columns)

# Check if 'Has Table Booking' column exists
if 'Has Table Booking' in csv_1.columns:
    # Filter the DataFrame for rows with 'Yes' and 'No' in the 'Has Table Booking' column
    csv_with_table_booking = csv_1[csv_1['Has Table Booking'] == 'Yes']
    csv_without_table_booking = csv_1[csv_1['Has Table Booking'] == 'No']
    
    # Check if 'Aggregate rating' column exists
    if 'Aggregate rating' in csv_1.columns:
        # Calculate and print the average ratings
        print("Average Ratings : ")
        print("with table booking : ", round(csv_with_table_booking["Aggregate rating"].mean(), 2))
        print("without table booking : ", round(csv_without_table_booking["Aggregate rating"].mean(), 2))
    else:
        print("The column 'Aggregate rating' does not exist in the DataFrame.")
else:
    print("The column 'Has Table Booking' does not exist in the DataFrame.")


# ## 3-Analyze the availability of online delivery amongewataurents with different price range 

# In[37]:


import matplotlib.pyplot as plt


# In[41]:


print(csv_1.columns)

# Step 2: Strip any leading or trailing whitespace from the column names
csv_1.columns = csv_1.columns.str.strip()

# Step 3: Check if 'Price range' and 'Has Online delivery' columns exist
if 'Price range' in csv_1.columns and 'Has Online delivery' in csv_1.columns:
    # Perform the grouping and plotting
    Online_Delivery_by_price_range = csv_1.groupby('Price range')['Has Online delivery'].value_counts(normalize=True).unstack() * 100
    Online_Delivery_by_price_range.plot(kind='bar', stacked=True, colormap='viridis', figsize=(10, 6))

    plt.title('Online Delivery Availability by Price Range')
    plt.xlabel('Price Range')
    plt.ylabel('Percentage')
    plt.legend(bbox_to_anchor=(1.05, 1), title='Online Delivery')
    plt.show()
else:
    print("The necessary columns do not exist in the DataFrame. Please check the column names.")


# In[42]:


# Taking only those restaurents with online delivery availible

Online_delivery_Yes = csv_1[csv_1['Has Online Delivery'] == 'Yes']

# Group by price renge and calculate the percentage of restaurents wth online delivery
Online_delivery_counts = Online_delivery_Yes.groupby(['Price Range' , 'Has Online Delivery']).size().unstack()
Online_delivery_counts.plot(kind='bar',stacked=True,colormap='cividis',figsize=(10,6))

plt.title('Online Delivery Availability by price range')
plt.xlabel('Price Range')
plt.ylabel('no of restaurents ')
plt.xticks(rotation = 0)
plt.legend(title='Online Delivery',bbox_to_anchor(1.05,1))
#plt.legend(title='Online Delivery', bbox_to_anchor=(1.05, 1),loc='upper left')
plt.show()


# In[44]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming csv_1 is your DataFrame

# Step 1: Print the column names to verify
print(csv_1.columns)

# Step 2: Strip any leading or trailing whitespace from the column names
csv_1.columns = csv_1.columns.str.strip()

# Step 3: Check if 'Price range' and 'Has Online delivery' columns exist
if 'Price range' in csv_1.columns and 'Has Online delivery' in csv_1.columns:
    # Perform the grouping and plotting
    Online_Delivery_by_price_range = csv_1.groupby('Price range')['Has Online delivery'].value_counts(normalize=True).unstack() * 100
    Online_Delivery_by_price_range.plot(kind='bar', stacked=True, colormap='viridis', figsize=(10, 6))

    plt.title('Online Delivery Availability by Price Range')
    plt.xlabel('Price Range')
    plt.ylabel('Percentage')
    plt.legend(bbox_to_anchor=(1.05, 1), title='Online Delivery')
    plt.show()
else:
    print("The necessary columns do not exist in the DataFrame. Please check the column names.")


#  #### -From the above 1st graph we can see that most of the  restaurent do not have theonline delivery services.In price renge 1 less then 20% are available in  price renge2 around 40% arw available,In price range 3 it look like 30% are available .and in price range of 4 only 10%are available.
# 
# #### From the above 2nd graph we analyze the people used to by from the price range2 and verey less number of people buy food from price range 4 may be because of its costliest in  price compare to others 

#  ###  Level 2=Task 2
#  ### Task: Price Range Analysis
#  ### 1-Determine the most common price range among all the restaurants.
#   

# In[46]:


import pandas as pd 
csv_1=pd.read_csv("D:\Internship\dataset.csv")
csv_1


# In[47]:


csv_1["Price range"].value_counts()


# In[48]:


most_common = csv_1["Price range"].mode()[0]
print("Most common Price among all restaurents : ", most_common )


# ###  2- Calculate the  avarage rating of each price range,
# ####                           & identify the color that represent then highest avarage ratingamong different price range. 

# In[49]:


Avarage_Rating_by_price_range = csv_1.groupby('Price range')['Aggregate rating'].mean().round(2)

print("Avarage Rating for each  price range :")
print(Avarage_Rating_by_price_range)


# In[50]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;sns.set(color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[53]:


# find the price range with highest avarage rating
highest_avg_rating_color = Avarage_Rating_by_price_range.idxmax()
plt.bar(Avarage_Rating_by_price_range.index, Avarage_Rating_by_price_range,color='skyblue' ,width=0.5)
plt.bar(highest_avg_rating_color , Avarage_Rating_by_price_range[highest_avg_rating_color] , color='green' , width=0.5 )

plt.xlabel('Price Range')
plt.ylabel(' Avarage Rating')
plt.title('Avarage rating by prive range')

plt.show()


# ##### Price range 4 get the highest avarage rating,which is 3.82 followed by price range 3,2,1. 

# # Level 2-Task 3:-
# ### 1. Extract additional features from the existing columns, such as the length of the restaurant name or address.

# In[54]:


csv_1['Restaurant Name Length'] = csv_1['Restaurant Name'].apply(lambda x: len(str(x)))
csv_1['Address Length'] = csv_1['Address'].apply(lambda x: len(str(x)))


# In[55]:


csv_1[['Restaurant Name' ,'Restaurant Name Length','Address' , 'Address Length']]


# ##### 2-Create new features like "Has Table Booking" or "Has Online Delivery" by encoding categorical variables  

# In[56]:


csv_1['Has Table Booking'] = csv_1['Has Table booking'].apply(lambda x:  1 if x == 'Yes' else 0)
csv_1['Has Online Delivery'] = csv_1['Has Online delivery'].apply(lambda x: 1 if x == 'Yes' else 0)


# In[57]:


csv_1[['Has Table Booking' ,'Has Table booking','Has Online Delivery', 'Has Online delivery']]


# #### -Two new column added  'Restaurant Name Length' and 'Adress Length' from the length of n the restaurant name or adress.
# #### -And also two new binary column added by encoding categorical variable 'Has Table Booking' and 'Has Online Delivery'.

# ## Observation:-
# 
# #### -Percentage of restaurants offer table booking is 12.12% and percentage of restaurants offer online delivery  is 25.66%.
# #### -Avarage rating with table booking is 3.44 and without table booking is 2.56.
# 
# #### -Most of the restaurants have do not online delivery service. In price range1 less then 20%  are available in price range 2 around 40% are available. in price range 3 it looklike 30% are available in price range 4 only 10% are available.
# 
# #### -People mostly buy from price range 2 and very less number of people buy food from  price range 4 may be because of its costliest in price compare to others.
# 
# #### - Most common price range among all restaurants are1.
# 
# #### - Price range 4 get the highest avarage rating  which is 3.83 followed by price range 3,2,1.

# # Level-3
#  
# ##  Task 1: Predictive Modeling
# 
#  ###  Build a regression model to predict the aggregate rating of a restaurant based on available features.
#  ### Split the dataset into training and testing sets and evaluate the model's performance using appropriate metrics.
#  ### Experiment with different algorithms (e.g., linear regression, decision trees, random forest) and compare their performance 
# 
# 
# ## Task 2 : Customer Preference Analysis
#  ### Analyze the relationship between the type of cuisine and the restaurant's rating.
#  ### Identify the most popular cuisines among customers based on the number of votes.
#  ### Determine if there are any specific cuisines that tend to receive higher ratings.
# 
# ## Task 3: Data Visualization
#  
#  ### Create visualizations to represent the distribution of ratings using different charts (histogram, bar plot, etc.).
#  ### Compare the average ratings of different cuisines or cities using appropriate visualizations.
#  ### Visualize the relationship between various features and the target variable to gain insights.
#  

# In[58]:


# Importing the warnings

import warnings
warnings.filterwarnings("ignore")


# In[59]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;sns.set(color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# ###  Level 3-  Task 1: Predictive Modeling
# ### 1. Build a regression model to predict the aggregate rating of a restaurant based on available features.
# ####               &Split the dataset into training and testing sets and evaluate the model's performance using appropriate metrics.¶  

# In[60]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[61]:


# Convert categorical variable to numeric

# csv_1 = pd.get_dummies(csv_1, columns=['Has Table booking' , 'Has Online delivery'], drop_first=True)

csv_1 = pd.get_dummies(csv_1, columns=['Has Table booking', 'Has Online delivery'], drop_first=True)


# In[62]:


# # Select features and target variable
x = csv_1[['Average Cost for two', 'Votes', 'Price range', 'Has Table booking_Yes', 'Has Online delivery_Yes']]
y = csv_1['Aggregate rating']


# In[63]:


# Split the dataset into training and test sets( 80% training and 20% testing)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)


# In[64]:


# Initialize and train the  Linear regression Model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict ratings on the testing set

y_pred = model.predict(x_test)

# Evaluate the models performance
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print("Model: Linear Regression")
print("Mean Squared Error (MSE):",mse)
print("R-squared (R2) Score:",r2)


# ### 2-Experiment with different algorithms (e.g., linear regression, decision trees, random forest) and compare their performance

# In[65]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[66]:


# Initialize and train  different regression model

models = {
    'Linear REgression': LinearRegression(),
    
    'Decision Tree':DecisionTreeRegressor(random_state=42),
    'Random Forest' : RandomForestRegressor(random_state=42)
    
}

# Evaluate models

results = {}
for name, model in models.items():
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    results[name] = {'MSE':mse, 'R2 Score' : r2}
        
        # Diaplay results
#result_csv_1 = csv_1.DataFrame(results) 
results_csv_1 = pd.DataFrame(results)
print(results_csv_1)


# ##### -Linear Regression produced an MSE of 1.6765 and an R-squared  value is 0.2634.
# #####  -Decision tree yielded an MSE of 0.2074 and an R-squared value of 0.9089
# ##### - Random forest  displayed the most promising results with the lowest MSE of approximately  0.1337 and the highest R squred of about 0.9413

#  ###  Level -3:Task 2
#  #### Task: Customer Preference Analysis
#  ####  1. Analyze the relationship between the type of cuisine and the restaurant's rating.

# In[67]:


cuisines = csv_1['Cuisines']


# In[68]:


cuisines.value_counts().head(10)


# In[69]:


# Get the top 10 most common cuisines

top_10_cuisines = cuisines.value_counts().head(10).index


# In[70]:


# Create a dataframe with cuisines ypes and currosponding Rating

cuisine_rating = pd.DataFrame({'Cuisine':cuisines,'Rating': csv_1['Aggregate rating']})


# In[71]:


# Filter cuisines rating dataframe to include only  the top 10 cuisines 
cuisine_ratings_top_10 = cuisine_rating[cuisine_rating['Cuisine'].isin(top_10_cuisines)]


# In[72]:


# Plot the relationship between top 20 cusines types nd rating

plt.figure(figsize=(12,6))
sns.boxplot(x='Cuisine', y='Rating',data=cuisine_ratings_top_10,palette='viridis')
plt.title('Relationship between top 10 cuisines types and Ratings')
plt.xlabel('Cuisine Type')
plt.ylabel('Rating')
plt.xticks(rotation=45)
plt.show()


# ### 2- Identify the most popular cuisines among customers based on the number of votes.

# In[73]:


# Create  a dataframe with the cuisine types and currospondng votes
cuisine_votes = pd.DataFrame({'Cuisine':cuisines,'Votes': csv_1['Votes']})


# In[74]:


# Groupby cuisind and sum the votes for each cuisines
cusine_votes_sum = cuisine_votes.groupby('Cuisine')['Votes'].sum()


# In[75]:


import pandas as pd

# Example: Replace this with your actual data loading code
# csv_1 = pd.read_csv('your_data.csv')

# Check column names to ensure they are correct
print(csv_1.columns)

# Remove any leading/trailing whitespace from column names
csv_1.columns = csv_1.columns.str.strip()

# Step 1: Group by Cuisines and sum the votes
cuisine_votes_sum = csv_1.groupby('Cuisines')['Votes'].sum().reset_index()
cuisine_votes_sum.columns = ['Cuisine', 'Total Votes']

# Step 2: Sort cuisines based on the total votes in descending order
popular_cuisines = cuisine_votes_sum.sort_values(by='Total Votes', ascending=False)

# Step 3: Display the sorted DataFrame
print(popular_cuisines)


# In[76]:


poular_cuisines = cuisine_votes_sum.sort_values(by='Total Votes', ascending=False)


# In[77]:


print(popular_cuisines.head(10))


# In[78]:


# Plotting the bar plot
plt.figure(figsize=(10,6))

popular_cuisines.head(10).plot(kind='bar',color='skyblue')

plt.title('Top 10 most popular cuisines based on number of votes')
plt.xlabel('Cuisine')
plt.ylabel('No of votes')
plt.xticks(rotation=45)
plt.show()


# ### 3-Determine if there are any specific cuisines that tend to receive higher ratings 

# In[79]:


#Create a DataFrame with cuisine types and currosponding ratings

cuisine_rating = pd.DataFrame({'Cuisine' : cuisines, 'Rating' : csv_1['Aggregate rating']})


# In[80]:


# CAlcutlate thge avarage rating for each cuisines

avarage_rating_by_cuisine  = cuisine_rating.groupby('Cuisine')['Rating'].mean()


# In[81]:


# Sort cuisine based on the avarage rating  in descending order
sorted_cuisines_by_rating = avarage_rating_by_cuisine.sort_values(ascending=False)


# In[82]:


# Display the top 10 cuisine with highest avarage rating

print("Top 10 cuisine with highest avarage rating :")
print(sorted_cuisines_by_rating.head(10))


# In[83]:


# Plot the graph

plt.figure(figsize=(12,6))

sorted_cuisines_by_rating.head(10).plot(kind='barh',color='skyblue')

plt.title('Top 10 most popular cuisines based on number of votes')
plt.xlabel(' top 10 Cuisine with  Highest avarage rating')
plt.ylabel('Avarage Rating')

plt.show()


# In[ ]:




