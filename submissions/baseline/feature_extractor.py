
import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import string

# remove dashes and apostrophes from punctuation marks 
punct = string.punctuation.replace('-', '').replace("'",'')
# regex to match intra-word dashes and intra-word apostrophes
my_regex = re.compile(r"(\b[-']\b)|[\W_]")

def clean_string(string, punct=punct, my_regex=my_regex, to_lower=False):
    if to_lower:
        string = string.lower()
    # remove formatting
    str = re.sub('\s+', ' ', string)
     # remove punctuation
    str = ''.join(l for l in str if l not in punct)
    # remove dashes that are not intra-word
    str = my_regex.sub(lambda x: (x.group(1) if x.group(1) else ' '), str)
    # strip extra white space
    str = re.sub(' +',' ',str)
    # strip leading and trailing white space
    str = str.strip()
    return str

class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        
        #### NLP BASICS ####
        X['name'] = X['name'].fillna('non available')
        X['short_description'] = X['short_description'].fillna('non available')
        names = X['name'].tolist()
        description = X["short_description"].tolist()
        
        cleaned_project_names = []
        
        for idx, doc in enumerate(names):
            # clean
            doc = clean_string(doc, punct, my_regex, to_lower=True)
            # tokenize (split based on whitespace)
            tokens = doc.split(' ')
            # remove digits
            tokens = [''.join([elt for elt in token if not elt.isdigit()]) for token in tokens]
            # remove tokens shorter than 3 characters in size
            tokens = [token for token in tokens if len(token) > 1]
            # remove tokens exceeding 25 characters in size
            tokens = [token for token in tokens if len(token) <= 25]
            cleaned_project_names.append(tokens)
            
        cleaned_description = []

        for idx, doc in enumerate(description):
            doc = clean_string(doc, punct, my_regex, to_lower=True)
            tokens = doc.split(' ')
            tokens = [''.join([elt for elt in token if not elt.isdigit()]) for token in tokens]
            tokens = [token for token in tokens if len(token)>3]
            tokens = [token for token in tokens if len(token)<=25]
            cleaned_description.append(tokens)
    
        cleaned_total = cleaned_project_names.copy()

        for a in cleaned_description:
            cleaned_total.append(a)
            
        ##### Train Word2vec 
        self.model = Word2Vec(cleaned_project_names, min_count=1, size=100, workers=8)
        
        return self

    def transform(self, X):
        
        data = X.copy()
        
        #### SIMPLE TRANSFORMATION #### 
        
        ### Date features
        data['launched_date'] = pd.to_datetime(data['created_at'], format='%Y-%m-%d %H:%M:%S')
        data['deadline_date'] = pd.to_datetime(data['deadline'], format='%Y-%m-%d %H:%M:%S')
        data['length'] = data['deadline_date'] - data['launched_date']
        data['length'] = [d.days for d in data['length']]
        data['year'] = [d.year for d in data['launched_date']]
        data['month'] = [d.month for d in data['launched_date']]
        data['day'] = [d.day for d in data['launched_date']]
        
        #Drop NaN in name and description and reset index
        data = data.dropna(subset=['name','short_description'])
        data.index = np.arange(0, len(data))
        
        # Length of name and description
        data['name_length'] = [len(name) for name in data['name']]
        data['description_length'] = [len(desc) for desc in data['name']]
        data['word_number_name'] = [len(name.split(' ')) for name in data['name']]
        data['word_number_desc'] = [len(desc.split(' ')) for desc in data['short_description']]

        # Ponctuation in name
        data['question'] = (data.name.str[-1] == '?').astype(int)
        data['exclamation'] = (data.name.str[-1] == '!').astype(int)

        # Create dummies for categorical features
        main_category = pd.get_dummies(data['main_category'],prefix='mc')
        category = pd.get_dummies(data['category'], prefix = 'cat')
        country = pd.get_dummies(data['country'], prefix = 'country')
        currency = pd.get_dummies(data['currency'], prefix = 'currency')
        data= pd.concat([data, main_category, category, country, currency], axis=1)
        
        
        #Switch to binary features
        data.description_available = data.description_available.astype(int)
        data.image_available = data.image_available.astype(int)
        data.disable_communication = data.disable_communication.astype(int)
        
        # Drop several features
        names = data['name'].tolist()
        description = data["short_description"].tolist()
    
        features_to_drop = [ 'name', 'short_description', 'country', 'city',
       'state_location', 'main_category', 'category', 'created_at',
       'launched_at', 'state_changed_at', 'deadline', 'currency', 
       'creator_name', 'creator_id', 'project_we_love', 'goal',  
       'launched_date','deadline_date', 'static_usd_rate', 'usd_type']
        data.drop(features_to_drop, axis=1, inplace=True)
  
        #### NLP BASICS ####
        cleaned_project_names = []
      
        for idx, doc in enumerate(names):
            # clean
            doc = clean_string(doc, punct, my_regex, to_lower=True)
            # tokenize (split based on whitespace)
            tokens = doc.split(' ')
            # remove digits
            tokens = [''.join([elt for elt in token if not elt.isdigit()]) for token in tokens]
            # remove tokens shorter than 3 characters in size
            tokens = [token for token in tokens if len(token) > 1]
            # remove tokens exceeding 25 characters in size
            tokens = [token for token in tokens if len(token) <= 25]
            cleaned_project_names.append(tokens)
            
        cleaned_description = []

        for idx, doc in enumerate(description):
            doc = clean_string(doc, punct, my_regex, to_lower=True)
            tokens = doc.split(' ')
            tokens = [''.join([elt for elt in token if not elt.isdigit()]) for token in tokens]
            tokens = [token for token in tokens if len(token)>3]
            tokens = [token for token in tokens if len(token)<=25]
            cleaned_description.append(tokens)
            
        name_matrix = np.zeros((len(cleaned_project_names), 100), dtype="float32")

        for i in range(len(cleaned_project_names)):
            try:
                name_matrix[i,]= self.model.wv[cleaned_project_names[i]].sum(0) / len(cleaned_project_names[i]) 
            except:
                pass
            
        description_matrix = np.zeros((len(cleaned_description),100),dtype="float32")
       
        for i in range(len(cleaned_description)):
            try:
                description_matrix[i,]=self.model.wv[cleaned_description[i]].sum(0)/len(cleaned_description[i]) 
            except:
                pass
       
                
       
        name_embeddings = pd.DataFrame(name_matrix)
        name_embeddings = name_embeddings.add_prefix('name_')
        description_embeddings = pd.DataFrame(description_matrix)
        description_embeddings = description_embeddings.add_prefix('desc_')
        
        data = pd.concat([data, name_embeddings], axis=1)
        data = pd.concat([data, description_embeddings], axis = 1)
        
        return data
