import pandas as pd
import string
import re
import os
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

#review data file loading
df = pd.read_csv(r'Project_WoC_7.0_Fake_Review_Detection\checkpoint_1\fakeReviewData.csv')

# Check the data structure
print(df.head(), df.info(),"\n--------")
print("Rows containing NULL values",df.isnull().sum())
print ("\n--------")
print ("Duplicate value containing rows", df.duplicated().sum())
print ("\n--------")
print("Data Scan Completed") 
os.system("pause")
os.system('cls')

# Data cleaning
df = df.drop_duplicates()
print("\nDuplicate Data Removed")

df = df.dropna(subset=['text_'])
print("\nRemoved Rows containing NULL values")

df = df[df['text_'].str.len() > 10]
print("\nRemoved Rows containing short reviews\n")

print(df.info())
print("\nDuplicates:", df.duplicated().sum())

os.system("pause")
os.system('cls')

# Data Normalization
def normalize_text(text): # function to make lowercse and remove numbers,special char,punctuation
    text = text.lower() 
    text = re.sub(r'\d+', '', text) 
    text = re.sub(r'[^\w\s]', ' ', text)
    return text

df['nrm_text'] = df['text_'].apply(normalize_text) # aplly funtion to all text

print("\n",df[['text_', 'nrm_text']].head()) # comparision

print("\nData Normalized\n")

os.system("pause")
os.system('cls')

# tokanization of data
import nltk
nltk.download('punkt')
nltk.download('punkt_tab') # pretrained data

from nltk.tokenize import word_tokenize

df['tokens'] = df['nrm_text'].apply(word_tokenize)
print("\ntokanization Complete")
print("\n",df[['text_', 'tokens']].head()) # comparision

os.system("pause")
os.system('cls')

# Stopword removal
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))  # Load English stopwords
df['filt_tokens'] = df['tokens'].apply(lambda tokens: [word for word in tokens if word not in stop_words])

print("\nStopwords Removed")    
print("\n",df[['tokens', 'filt_tokens']].head()) # comparision

os.system("pause")
os.system('cls')

# Lemmatization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
df['lemma_tokens'] = df['filt_tokens'].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])

print("\nLemmatized tokens")    
print("\n",df[['filt_tokens','lemma_tokens']].head()) # comparision
os.system("pause")
os.system('cls')

#Vectorize output
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
df['final_text'] = df['lemma_tokens'].apply(lambda tokens: ' '.join(tokens))  # Convert list to text
print("\n",df[['lemma_tokens','final_text']].head()) # comparision
os.system("pause")
os.system('cls')

df.to_csv(r"Project_WoC_7.0_Fake_Review_Detection\checkpoint_1\preprocessed_reviews.csv", index=False)
print("pre done")

X = vectorizer.fit_transform(tqdm(df['final_text']))  # Convert text to numerical vectors
vec_matrix =pd.DataFrame.sparse.from_spmatrix(X,columns=vectorizer.get_feature_names_out())

vec_matrix.to_csv(r'Project_WoC_7.0_Fake_Review_Detection\checkpoint_1\tfidf_matrix.csv', index=False)
print("pre processed data ready")
