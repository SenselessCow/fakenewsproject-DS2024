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

# Adds a boolean indicator to x and y -test to show wether the article is true or not. 
def map_to_authenticity(grouptype):
    if grouptype in ['GroupFake', 'GroupOmitted']:
        return 0
    elif grouptype == 'GroupReliable':
        return 1
    
#If a single column only contains 2 different values, this will turn all entries of the most common one (most likely one to be a fake article)
#into "1"'s and all the most uncommon ones into "0"'s yet to be tested on a dataset containing any true articles.
def interpret_commons_as_bool(y_pred):
    y_pred_uniques = np.unique(y_pred, return_counts=True)
    if np.size(y_pred_uniques[0]) > 1:
        if y_pred_uniques[1][0] > y_pred_uniques[1][1]:
            for i in range (np.size(y_pred)):
                if y_pred[i] == y_pred_uniques[0][0]:
                    y_pred[i] = 0
                else:
                    y_pred[i] = 1
        else:
            for i in range (np.size(y_pred)):
                if y_pred[i] == y_pred_uniques[0][0]:
                    y_pred[i] = 1
                else:
                    y_pred[i] = 0
    else:
        for i in range (np.size(y_pred)):
            y_pred[i] = 0
    return y_pred

#Vi sætter en boolean værdi til at indikere om artiklens hjemmeside har et "ry" for at lave true artikler. Dette gemmes i parameter kolennen "trusted"
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

# Returns list containining one of each trusted domains from a given dataset. Now its called on the entire corpus, but it should only
#be run on our training data..
def trusted_sites(a):
    b=[]
    for i in range (len(a['content'])):
        if a['GroupedType'][i] == 'GroupReliable':
            b.append(a['domain'][i])
    result = np.unique(b, return_counts=False)
    print("reliable domains found in cleaned data = " + str(result))   
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

#Jeg har strugglet rigtigt meget med at fordele dataen... Dette var en af mine implementationer, men dette ville resultere i at man var nødt 
#til at croppe dataen så jeg valgte at abandon det, ville bare vise det så derfor har jeg beholdt det xD
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

#Vi laver en liste af indeks som tilsvarer mængden af indeks i vores corpus. Vi tager næst en fordeling af disse index til hver af vores dataset
#Herefter smider vi dataen fra de tilhørende indeks ind i datasettene. Den normale måde fungerer åbenbart ikke på store 2d panda df's så 
#vidt jeg forstår xd..
indexlist = []
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

#laver dem boolean
X_test['trusted'] = X_test['GroupedType'].apply(map_to_authenticity)
y_test['trusted'] = y_test['GroupedType'].apply(map_to_authenticity)

# print(X_test)

# X_test = X_test['domain'].apply(domain_to_boolean,args=(reliable_sites))
X_train = domain_to_boolean(X_train, reliable_sites)
y_train = domain_to_boolean(y_train, reliable_sites)
# print(X_test)

#Definer hvilke features vi faktisk skal bruge til modellen.
test_col = ['trusted']
feature_cols = ['trusted']
X_train = X_train.loc[:, feature_cols]
y_train = y_train.loc[:, feature_cols]
X_test = X_test.loc[:, test_col]
y_test = y_test.loc[:, test_col]

model = LinearRegression()
reg = model.fit(X_train, y_train)     #MEGET LANGSOM LINJE
y_pred = reg.predict(X_test)

#Laver y_pred til binary ved at tælle hvilken værdi der er flest af, og så lave alle kopier af den værdi til 0 og den anden til 1
y_pred = interpret_commons_as_bool(y_pred)
# print(y_pred)

mse = mean_squared_error(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
 
# DO NOT INSERT OR CHANGE ANYTHING BELOW
print("LinearRegression MSE: ", mse)
print("LinearRegression accuracy: ", acc)