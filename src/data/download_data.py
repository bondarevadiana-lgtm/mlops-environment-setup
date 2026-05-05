from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import os


os.makedirs('data/raw', exist_ok=True)

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

df = pd.DataFrame({'text': data.data, 'target': data.target})

df.to_csv('data/raw/newsgroups.csv', index=False)
print("Dataset saved to data/raw/newsgroups.csv")