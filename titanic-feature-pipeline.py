import os
import modal
#import great_expectations as ge
import hopsworks
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

project = hopsworks.login()
fs = project.get_feature_store()

data_raw = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")
data1 = data_raw.copy(deep = True)

#complete missing age with median
data1['Age'].fillna(data1['Age'].median(), inplace = True)

#complete embarked with mode
data1['Embarked'].fillna(data1['Embarked'].mode()[0], inplace = True)

#complete missing fare with median
data1['Fare'].fillna(data1['Fare'].median(), inplace = True)
    
#delete the cabin feature/column and others previously stated to exclude in train dataset
drop_column = ['PassengerId','Cabin', 'Ticket']
data1.drop(drop_column, axis=1, inplace = True)

###CREATE: Feature Engineering for train and test/validation dataset

#Discrete variables
data1['FamilySize'] = data1 ['SibSp'] + data1['Parch'] + 1

data1['IsAlone'] = 1 #initialize to yes/1 is alone
data1['IsAlone'].loc[data1['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1

#quick and dirty code split title from name: http://www.pythonforbeginners.com/dictionary/python-split
data1['Title'] = data1['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]


#Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut
#Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
data1['FareBin'] = pd.qcut(data1['Fare'], 4)

#Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
data1['AgeBin'] = pd.cut(data1['Age'].astype(int), 5)

#cleanup rare title names
#print(data1['Title'].value_counts())
stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
title_names = (data1['Title'].value_counts() < stat_min) #this will create a true false series with title name as index

#apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)


#code categorical data
label = LabelEncoder()

data1['Sex_Code'] = label.fit_transform(data1['Sex'])
data1['Embarked_Code'] = label.fit_transform(data1['Embarked'])
data1['Title_Code'] = label.fit_transform(data1['Title'])
data1['AgeBin_Code'] = label.fit_transform(data1['AgeBin'])
data1['FareBin_Code'] = label.fit_transform(data1['FareBin'])


#define y variable aka targets/outcome
Target = ['Survived']

#define x variables for original w/bin features to remove continuous variables
data1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code', 'Survived']

final_data = data1[data1_x_bin]


titanic_fg = fs.get_or_create_feature_group(
    name="titanic_modal",
    version=1,
    primary_key=data1_x_bin, 
    description="Titanic")
titanic_fg.insert(final_data, write_options={"wait_for_job" : False})


