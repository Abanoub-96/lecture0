import pandas as pd 
import numpy as np
import requests 
import urllib , urllib3
import csv , re , sklearn
import requests.exceptions 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import arabic_reshaper 
import pyarabic.araby as araby
from sklearn import svm 
from sklearn import cross_decomposition 
from sklearn import datasets

colnames = ['link','class']
raheem=[] # list for raheem related links
unrelated=[] # list for unrelated links 
working_links=[] #list for refined links (excluding broken links)
urls= pd.read_csv('Dataset.csv',delimiter=",",usecols=colnames)
links=urls.link.tolist()  #convert links into a list
for link in links:        #errors handling   
    try:
        r = requests.get(link)
        r.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xxx
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,requests.exceptions.TooManyRedirects):
        print ("Down")
    except requests.exceptions.HTTPError:
        print ("4xx, 5xx")
    else:
        working_links.append(link) #appends all working links to working_list
        print ("All good!")  # Proceed to do stuff with `r` 

    
for i in working_links:       #extracting raheem related and non related to append to lists  
    x=requests.get(i)
    html_text= x.text
    if "رحيم" in html_text:
        raheem.append(i)            #appending link to raheem list
    else:
        unrelated.append(i)         #appending link to unrelated link
    
# spliting data from both lists for training and testing 80% and 20%
X_train, X_test, y_train, y_test = train_test_split(raheem, unrelated, test_size = 0.20,random_state=0)

# Build model for predicting using trained data
clf= svm.SVC(kernel='linear',C=1).fit(X_train, y_train)
# performance measurment 
clf.score(X_test,y_test)

scores= cross_val_score(clf,raheem,unrelated,cv=5)
print (scores) 
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))




