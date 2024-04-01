import dataprocessing
import sklearn.model_selection as ms
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
import re
from collections import Counter
# from collections import Counter


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

def data_stats(clean_data):
     # All groups percentage
    all_groups_counts = clean_data['GroupedType'].value_counts()
    all_groups_percentage = all_groups_counts / all_groups_counts.sum() * 100

    # 'GroupFake' and 'GroupReliable' only percentage
    filtered_counts = clean_data[clean_data['GroupedType'].isin(['GroupFake', 'GroupReliable'])]['GroupedType'].value_counts()
    filtered_percentage = filtered_counts / filtered_counts.sum() * 100

    # Combine data into a single DataFrame for plotting
    comparison_df = pd.DataFrame({'AllGroups': all_groups_percentage, 'FakeVsReliable': filtered_percentage}).fillna(0)
    print(comparison_df)
    return

#Define two ranges for a word count to fall under to be either true or false
# def word_count_range(true_words,false_words,l_A,l_B):
#     wa = 25  #Common word amount
#     max_true = true_words.most_common(wa)
#     max_false = false_words.most_common(wa)
#     list_avg_A = []
#     list_avg_B = []
#     # print("max_true= ",max_true,"\nmax_false= ",max_false)
    
#     #Bare et forslag til hvad man kunne gøre, jeg tror det kan gøres meget smartere og bedre sikkert:
#     true_deviance_range_lst = []  # liste af lister hvor [[ord,min,max][ord,min,max]...] hvor vi specificere en range som et ord kan være i for at være true
#     fake_deviance_range_lst = []                                                                                                                 #eller fake
#     for i in range(wa):
#         # print(max_true[i][0])
#         for j in range(wa):
#             if max_true[i][0] == max_false[j][0]:
#                 avrg_tru_word_pr_art = max_true[i][1]/l_A
#                 avrg_false_word_pr_art = max_false[j][1]/l_B
#                 mean = (avrg_tru_word_pr_art+avrg_false_word_pr_art)/2
#                 difference = abs(avrg_tru_word_pr_art-mean)
#                 true_deviance_range_lst.append([max_true[i][0],avrg_tru_word_pr_art-difference,avrg_tru_word_pr_art+difference])
#     # print("true word deviance list:",true_deviance_range_lst,"\n")
#     for i in range(wa):
#         for j in range(wa):
#             if max_false[i][0] == max_true[j][0]:
#                 avrg_false_word_pr_art = max_false[i][1]/l_B
#                 avrg_tru_word_pr_art = max_true[j][1]/l_A
#                 mean = (avrg_false_word_pr_art+avrg_tru_word_pr_art)/2
#                 difference = abs(avrg_false_word_pr_art-mean)
#                 fake_deviance_range_lst.append([max_false[i][0],avrg_false_word_pr_art-difference,avrg_false_word_pr_art+difference])
#     # print("fake word deviance list:",fake_deviance_range_lst,"\n")
    
#     # for i in range(len(true_deviance_range_lst)):
#         #print("For an article to be true it must contain at least",true_deviance_range_lst[i][1]," instances of  the word ", true_deviance_range_lst[i][0]," and at most ",true_deviance_range_lst[i][2]," instances.\n")
#     # print(max_true[i][0]," ",max_true[i][1]/len(true_df['trusted']),"true")      print alle de true og fake ord og deres avrg per artikel
#     # print(max_false[i][0]," ",max_false[i][1]/len(fake_df['trusted']),"fake")
#     return [true_deviance_range_lst,fake_deviance_range_lst]

# If it in training finds a website that has made a credible article before, it will mark every article in the test set made by the website 
# as credible 
# def linear_model1(clean_data):
#     print("linear_model1 run:")
    
#     # Optional line makes it so we can see all columns every time we print a dataframe
#     # pd.set_option('display.max_columns', None)

#     # Apply the function to create a new 'GroupedType' column
#     clean_data['GroupedType'] = clean_data['type'].apply(map_to_group)

#     # Make a list with all unique trusted websites                      #Ibler update
#     reliable_sites = trusted_sites(clean_data)

#     # Stats on distribution of reliable fake and omitted articles
#     data_stats(clean_data)

#     #Vi laver en liste af indeks som tilsvarer mængden af indeks i vores corpus. Vi tager næst en fordeling af disse index til hver af vores dataset
#     #Herefter smider vi dataen fra de tilhørende indeks ind i datasettene. Den normale måde fungerer åbenbart ikke på store 2d panda df's så 
#     #vidt jeg forstår xd..
#     indexlist = []
#     for a in range (len(clean_data)):
#         indexlist.append(a) 
#     vX_train, vX_test, vy_train, vy_test = ms.train_test_split(indexlist, indexlist, test_size=0.2, random_state=0)
#     X_train = clean_data.iloc[vX_train].reset_index()
#     y_train = clean_data.iloc[vy_train].reset_index()
#     X_test = clean_data.iloc[vX_test].reset_index()
#     y_test = clean_data.iloc[vy_test].reset_index()
#     # pd.set_option('display.max_columns', None)
#     # print(X_train)

#     # Laver dem boolean
#     X_test['trusted'] = X_test['GroupedType'].apply(map_to_authenticity)
#     y_test['trusted'] = y_test['GroupedType'].apply(map_to_authenticity)
#     # print(X_test)

#     # X_test = X_test['domain'].apply(domain_to_boolean,args=(reliable_sites))
#     X_train = domain_to_boolean(X_train, reliable_sites)
#     y_train = domain_to_boolean(y_train, reliable_sites)
#     # print(X_test)

#     # Definer hvilke features vi faktisk skal bruge til modellen.
#     test_col = ['trusted']
#     feature_cols = ['trusted']
#     X_train = X_train.loc[:, feature_cols]
#     y_train = y_train.loc[:, feature_cols]
#     X_test = X_test.loc[:, test_col]
#     y_test = y_test.loc[:, test_col]

#     model = LinearRegression()
#     reg = model.fit(X_train, y_train)     #MEGET LANGSOM LINJE
#     y_pred = reg.predict(X_test)

#     # Laver y_pred til binary ved at tælle hvilken værdi der er flest af, og så lave alle kopier af den værdi til 0 og den anden til 1
#     y_pred = interpret_commons_as_bool(y_pred)
#     # print(y_pred)

#     mse = mean_squared_error(y_test, y_pred)
#     acc = accuracy_score(y_test, y_pred)
    
#     print("LinearRegression MSE: ", mse)
#     print("LinearRegression accuracy: ", acc,"\n")
#     return

#Tager pandas df og finder ud af hvilke ord i hvilken mængde kan være med til at afgøre om en artikel er fake eller reliable 
def simpleModel1(cleaned_data):
    cleaned_data['GroupedType'] = cleaned_data['type'].apply(map_to_group)
    cleaned_data['trusted'] = cleaned_data['GroupedType'].apply(map_to_authenticity)
    true_df = pd.DataFrame()
    fake_df = pd.DataFrame()
    for i in range(len(cleaned_data)):
        if len(fake_df) < 1:
            if cleaned_data.iloc[i]['trusted'] == 1:
                true_df = cleaned_data.iloc[[i]]
            else:
                fake_df = cleaned_data.iloc[[i]]
        else:
            if cleaned_data.iloc[i]['trusted'] == 1:
                true_df = true_df._append(cleaned_data.iloc[[i]], ignore_index=True)
            elif cleaned_data.iloc[i]['trusted'] == 0:
                fake_df = fake_df._append(cleaned_data.iloc[[i]], ignore_index=True)
    l_A = len(true_df)
    l_B = len(fake_df)
    # print("l_A: ",l_A," l_B: ",l_B," datA: ",dataA," datB: ",fake_df)
    f, g, true_words = dataprocessing.count_words(true_df, 'content')
    h, j, false_words = dataprocessing.count_words(fake_df, 'content')

    # true_deviance_range_lst, fake_deviance_range_lst = word_count_range(true_words,false_words,l_A,l_B)
    # print("true word deviance list:",true_deviance_range_lst,"\n")
    # print("fake word deviance list:",fake_deviance_range_lst,"\n")
    def funktion(X):
            # print(X)
            wa = 25  #Common word amount
            max_true = true_words.most_common(wa)
            max_false = false_words.most_common(wa)
            count = Counter(X)
            if len(count) < wa:
                wa = len(count)
            else:
                wa = wa
            X_common = count.most_common(wa)
            # print(X_common)
            # print("max_true= ",max_true,"\nmax_false= ",max_false)
            fake_distance_lst = []
            true_distance_lst = []
            for i in range(wa):
                #print(max_true[i][0])
                for j in range(wa):
                    if X_common[i][0] == max_false[j][0] and X_common[i][0] == max_true[j][0]:
                        fake_distance_lst.append([X_common[i][0],abs(X_common[i][1]/len(X)-max_false[j][1]/l_B)])
                        true_distance_lst.append([X_common[i][0],abs(X_common[i][1]/len(X)-max_true[j][1]/l_A)])
                    elif X_common[i][0] == max_false[j][0]:
                        fake_distance_lst.append([X_common[i][0],abs(X_common[i][1]/len(X)-max_false[j][1]/l_B)])
                    elif X_common[i][0] == max_true[j][0]:
                        true_distance_lst.append([X_common[i][0],abs(X_common[i][1]/len(X)-max_true[j][1]/l_A)])
            
            n_true_list = true_distance_lst
            n_fake_list = fake_distance_lst
            
            real_score = 0
            fake_score = 0
            for i in true_distance_lst:
                for j in fake_distance_lst:
                    if i[0] == j[0]:
                        n_true_list.remove(i)
                        n_fake_list.remove(j)
                        if i[1] < j[1]:
                            real_score = real_score + 1
                        elif i[1] > j[1]:
                            fake_score = fake_score + 1
            real_score = real_score + len(n_true_list)
            fake_score = fake_score + len(n_fake_list)
            print("real_score: ",real_score," fake_score: " ,fake_score)    
            if real_score > fake_score:
                return 1 
            else:
                return 0


    #Train test split
    indexlist = []
    for a in range (len(cleaned_data)):
        indexlist.append(a) 
    vX_train, vX_test, vy_train, vy_test = ms.train_test_split(indexlist, indexlist, test_size=0.2, random_state=0)
    X_train = cleaned_data.iloc[vX_train].reset_index()
    y_train = cleaned_data.iloc[vy_train].reset_index()
    X_test = cleaned_data.iloc[vX_test].reset_index()
    y_test = cleaned_data.iloc[vy_test].reset_index()

    # Definerer i ['trusted'] om artiklen er fake(0) eller  reliable(1) med boolena 0 og 1
    X_test['trusted'] = X_test['GroupedType'].apply(map_to_authenticity)
    y_test['trusted'] = y_test['GroupedType'].apply(map_to_authenticity)

    #Find distance from wordsCount to true or false wordcounts and predict if its a true or false article
    X_train['trusted'] = X_train['content'].apply(funktion)
    y_train['trusted'] = y_train['content'].apply(funktion)
    print(X_train)
    print(y_train)
    
    # article_contains_words()
    grid = LogisticRegression(max_iter=500)
    reg = grid.fit(X_train, y_train)
    print(grid.best_params_)
    print(grid.best_score_)
    y_pred = reg.predict(X_test)

    # Laver y_pred til binary ved at tælle hvilken værdi der er flest af, og så lave alle kopier af den værdi til 0 og den anden til 1
    y_pred = interpret_commons_as_bool(y_pred)
    # print(y_pred)

    mse = mean_squared_error(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    print("LinearRegression MSE: ", mse)
    print("LinearRegression accuracy: ", acc,"\n")
    return

# Include to run models without main.py
raw_data = dataprocessing.get_data("news_sample.csv")
clean_data = raw_data.copy()
clean_data['content'] = raw_data['content'].apply(dataprocessing.text_preprocessing)
# linear_model1(clean_data)
# logistic_model2(clean_data)
simpleModel1(clean_data)
# iblersfunktion(clean_data)