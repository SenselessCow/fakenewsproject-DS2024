import dataprocessing
import sklearn.model_selection as ms
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import pandas as pd 


# Function to map types to new groups
def map_to_group(type_value):
    if type_value in ['fake', 'conspiracy']:
        return 'GroupFake'
    elif type_value == 'reliable':
        return 'GroupReliable'
    else:
        return 'GroupOmitted'

# Ville lave en funktion som kunne tilføje en boolean værdi til hver artikel der angiver om 
# domainet har udgivet reliable informationer før. Lige nu overskriver boolean værdierne domain 
# parameterne, men de burde faktisk bare lave en ny parameter for hvert row.
# PROBLEM! ! ! ! ! Af en eller anden grund er det ikke muligt at iterate alle X's domains, og
# derfor er jeg stuck. jeg har også prøvet at lave alt dette med pandas .apply, men jeg var for dum..  
def domain_to_boolean(X,reliables):
    for j in range (len(X['domain'])):
        i=0
        # print(X.loc[j, "domain"])
        if reliables.dtype.type is np.str_:
            if X.loc[j, "domain"] != (reliables):
                X.loc[j, "trusted"] = 0
            else:
                X.loc[j, "trusted"] = 1
        else:
            while X.loc[j, "domain"] != (reliables[i]):
                i=i+1
                if i+1 == len(reliables):
                    X.loc[j, "trusted"] = 0
            X.loc[j, "trusted"] = 1
            j=j+1
        
    return X

# def domain_to_boolean(domain,reliables):
#     i=0
#     while domain != (reliables[i]):
#         i=i+1
#         if i+1 == len(reliables):
#             return 0
#     return 1

# Returns list containining one of each trusted domains from a given dataset
def trusted_sites(a):
    b=[]
    for i in range (len(a['content'])):
        if a['GroupedType'][i] == 'GroupReliable':
            b.append(a['domain'][i])
    result = np.unique(b, return_counts=False)
    print("reliable sites found = " + str(result))   
    return result[0]

raw_data = dataprocessing.get_data()
clean_data = raw_data.copy()
clean_data['content'] = raw_data['content'].apply(dataprocessing.text_preprocessing)


# Apply the function to create a new 'GroupedType' column
clean_data['GroupedType'] = clean_data['type'].apply(map_to_group)

# Make a list with all unique trusted websites                      #Ibler update
reliable_sites = trusted_sites(clean_data)

# All groups percentage
all_groups_counts = clean_data['GroupedType'].value_counts()
all_groups_percentage = all_groups_counts / all_groups_counts.sum() * 100

# 'GroupFake' and 'GroupReliable' only percentage
filtered_counts = clean_data[clean_data['GroupedType'].isin(['GroupFake', 'GroupReliable'])]['GroupedType'].value_counts()
filtered_percentage = filtered_counts / filtered_counts.sum() * 100

# Combine data into a single DataFrame for plotting
comparison_df = pd.DataFrame({'AllGroups': all_groups_percentage, 'FakeVsReliable': filtered_percentage}).fillna(0)


#IBLERS UPDATES - 
X = clean_data
# y = clean_data['GroupedType']

#Split dataen op, den klassiske måde virkede ikke for det resulterede i at man kun kunne iterere columns men ikke rows i vores pandas dataframe..
#Med denn måde at fordele det på bliver fordelingen lidt forskellig hver gang, men det virker. Vi skal fikse det tho
# X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=0)

# msk = np.random.rand(len(X)) < 0.8
# X_trainv = X[msk]
# X_testv = X[~msk]
# msk = np.random.rand(len(X_trainv)) < 0.5
# X_train = X_trainv[msk].reset_index()
# y_train = X_trainv[~msk].reset_index()
# msk = np.random.rand(len(X_testv)) < 0.5
# X_test = X_testv[msk].reset_index()
# y_test = X_testv[~msk].reset_index()
# del X_trainv
# del X_testv

indexlist = []
# print(indexlist[0])
for a in range (len(X)):
    indexlist.append(a) 
vX_train, vX_test, vy_train, vy_test = ms.train_test_split(indexlist, indexlist, test_size=0.2, random_state=0)
# print(vX_train)
# print(len(X))
# print(len(vX_train))
# print(len(vy_train))
# print(len(vX_test))
# print(len(vy_test))
# print(X.iloc[3])
X_train = X.iloc[vX_train].reset_index()
y_train = X.iloc[vy_train].reset_index()
X_test = X.iloc[vX_test].reset_index()
y_test = X.iloc[vy_test].reset_index()
# print(X_train)

# X_train = X_train['domain']
# y_train = y_train['GroupedType']
y_test =y_test['GroupedType']

# X_test = X_test['domain'].apply(domain_to_boolean,args=(reliable_sites))
X_train = domain_to_boolean(X_train, reliable_sites)
y_train = domain_to_boolean(y_train, reliable_sites)
# print(X_test)

feature_cols = ['trusted']
# you want all rows, and the feature_cols' columns
X_train = X_train.loc[:, feature_cols]
print(X_train)
y_train = y_train.loc[:, feature_cols]
print(y_train)

#Mange data skal behandles før dette vil virke:
model = LinearRegression()
# print(X_train)
# print(y_train)
reg = model.fit(X_train, y_train)     #MEGET LANGSOM LINJE
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
 
# DO NOT INSERT OR CHANGE ANYTHING BELOW
print("LinearRegression MSE: ", mse)
print("LinearRegression accuracy: ", acc)