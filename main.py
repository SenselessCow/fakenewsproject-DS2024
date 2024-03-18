import dataprocessing

raw_data = dataprocessing.get_data()
cleaned_data = raw_data.copy()

cleaned_data['content'] = raw_data['content'].apply(dataprocessing.text_preprocessing)

print(cleaned_data['content'][0:5])