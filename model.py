import dataprocessing
import sklearn.model_selection as ms
import pandas as pd 

raw_data = dataprocessing.get_data()
cleaned_data = raw_data.copy()
reliable_data = raw_data.copy()
print(len(raw_data.content))

cleaned_data['content'] = raw_data['content'].apply(dataprocessing.text_preprocessing)
#print(cleaned_data)
print(reliable_data)
# i=0
# while i <= (len(reliable_data.content)):
#     if reliable_data['type'][i] != 'reliable':
#         reliable_data = reliable_data.droplevel(i)
#     else:
#         i = i + 1
# reliable_data = reliable_data.drop("id",axis=1)
# reliable_data = reliable_data.drop(reliable_data['type'][reliable_data['type'] != 'reliable'], inplace = True)
reliable_data = reliable_data.drop(reliable_data[~(reliable_data['type'] != 'reliable')])
print(reliable_data)


# reliable_data['content'] = raw_data['content'].apply(only_relaible)

# def only_reliable(x):
#     if x["type"] != "reliable":
#         reliable_data.drop(reliable_data[reliable_data["type"]])
        
#     return

# reliable_data = []
# for i in range (len(raw_data.content)):
#     print(raw_data["type"][i])
#     if raw_data["type"][i] == "reliable":
#         reliable_data.append(raw_data["content"][i])
# print(reliable_data)

X_train, X_test, y_train, y_test = ms.train_test_split(cleaned_data, reliable_data, test_size=0.2, random_state=0)

