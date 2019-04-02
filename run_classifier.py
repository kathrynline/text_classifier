# -*- coding: utf-8 -*-
"""
AUTHOR: Emily Linebarger 
PURPOSE: Select a machine learning model to fit module/intervention/code from an activity description 
    for Global Fund budgets and PU/DRs. 
DATE: February 2019 
"""

#Import your libraries 
import numpy as np
import matplotlib
import pandas as pd
import sklearn
import string
import os
import sys

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from googletrans import Translator
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import re

pd.options.display.float_format = '{:20,.2f}'.format

translator = Translator()
#translation = translator.translate('buenos dias').text
#translation = translator.translate('안녕하세요.')
## <Translated src=ko dest=en text=Good evening. pronunciation=Good evening.>
#translator.translate('안녕하세요.', dest='ja')
## <Translated src=ko dest=ja text=こんにちは。 pronunciation=Kon'nichiwa.>
#translator.translate('veritas lux mea', src='la')
## <Translated src=la dest=en text=The truth is my light pronunciation=The truth is my light>

if sys.platform == "win32":
    j = "J:/"
else:
    j = "homes/j/"

mappingDir = j + "Project/Evaluation/GF/resource_tracking/modular_framework_mapping/"
nlpDir = j + "Project/Evaluation/GF/resource_tracking/modular_framework_mapping/nlp/"   

repo_loc = "C:/Users/elineb/Documents/text_classifier/" 
os.chdir(j + "Project/Evaluation/GF/resource_tracking/modular_framework_mapping/nlp") #This is where your output will print. 

#---------------------------------------------------------
# To-do list for this code: 
#   verify that you have the same model fitting for all languages
#   Add disease as an independent variable in the model. 
#   Run some descriptive statistics on the types and counts of codes that are in the training data right now. 
#   Make an array of the vectorization of activity description. We really only want this as one variable. 
# Need to fix the input training data; we've had some corruption of the formatting. 
#---------------------------------------------------------

#Write your lists of stopwords 
stopwords_french = list(stopwords.words('french'))
stopwords_spanish = list(stopwords.words('spanish'))
stopwords_english = list(stopwords.words('english'))
stopwords_all = stopwords_french + stopwords_spanish + stopwords_english

#Read in your replacement acronym lists. and subset to only the acronym and the original language translation. 
# You could play with this and try different methodologies to see if one works better. 
acronyms_esp = pd.read_csv(repo_loc + "acronyms_spanish.csv", encoding = "latin-1")
acronyms_esp = acronyms_esp[['acronym', 'host_translation']]
acronyms_fr = pd.read_csv(repo_loc + "acronyms_french.csv", encoding = "latin-1")
acronyms_fr = acronyms_fr[['acronym', 'host_translation']]
acronyms_eng = pd.read_csv(repo_loc + "acronyms_english.csv", encoding = "latin-1")
acronyms_eng = acronyms_eng[['acronym', 'host_translation']]

#Read in your pre-prepared training data
handcoded_all = pd.read_csv(nlpDir + "nlp_training_handcoded_all.csv", encoding = "latin-1")
teeny_test = handcoded_all[1:100] #Make a tiny dataset you can test code with
modular_framework = pd.read_csv(mappingDir + "all_interventions.csv", encoding = "latin-1")

#Read in the full list of indicator codes so you can see how comprehensive the training data you have is. 

def review_training_data(dataset, label):
    print("Data review for dataset: " + label)
    dataset = dataset.applymap(str)
    
    #Review what percentage of the modular framework your training data is covering
    dataset_codes = dataset.code.unique()
    dataset_codes = pd.Series(dataset_codes)
    mf_codes = modular_framework.code.unique()
    mf_codes = pd.Series(mf_codes)
    
    missing_codes = mf_codes.isin(dataset_codes)
    print("Percentage of codes in modular framework covered by this training data")
    print("True means this code is covered.")
    print(missing_codes.value_counts())
    
    #What is the representation of the codes you do have? Are things underweighted? 
    print("What is the distribution of the codes in the data? Are codes weighted unevenly?")
    print(dataset.code.value_counts())
    
    #What is the distribution of the languages, and diseases, in the data? 
    print("What is the distribution of the languages and diseases in the data?")
    print(dataset.lang.value_counts())
    print(dataset.disease.value_counts())

def test_models(label, dataset, stopWords, translate, models, balanceData):
    #-------------------------------------------------------------------------
    # Common natural language processing prep before running machine learning 
    #-------------------------------------------------------------------------
    dataset = dataset.applymap(str)
    print("Number of observations for dataset " + label + str(dataset.shape))
    print("")
    
     #Fix acronyms - first split by language 
    eng = dataset[dataset['lang'] == 'english']
    esp = dataset[dataset['lang'] == 'spanish']
    fr = dataset[dataset['lang'] == 'french']
    
    #Then replace acronym with the host language translation 
    # EMILY NEED TO ADD IN ENGLISH HERE 
    esp_acronym_corrected = []
    for activity in esp['sda_activity']:
        for index, row in acronyms_esp.iterrows():
            activity = re.sub(row['acronym'], row['host_translation'], activity)
        esp_acronym_corrected.append(activity)
    esp['sda_activity'] = esp_acronym_corrected
    
    fr_acronym_corrected = []
    for activity in fr['sda_activity']:
        for index, row in acronyms_fr.iterrows():
            activity = re.sub(row['acronym'], row['host_translation'], activity)
        fr_acronym_corrected.append(activity)
    fr['sda_activity'] = fr_acronym_corrected
    
    eng_acronym_corrected = []
    for activity in eng['sda_activity']:
        for index, row in acronyms_eng.iterrows():
            activity = re.sub(row['acronym'], row['host_translation'], activity)
        eng_acronym_corrected.append(activity)
    eng['sda_activity'] = eng_acronym_corrected
        
    corrected_acronyms = eng.append(fr, ignore_index = True)
    corrected_acronyms = corrected_acronyms.append(esp, ignore_index = True)
    dataset = corrected_acronyms
    
    #Create a list of activities to work with (prep for vectorization)
    activities = list(dataset['sda_activity'])
    
    #Remove numbers and punctuation
    new_activities = []
    translator1 = str.maketrans('', '', string.digits) #Remove numbers 
    translator2 = str.maketrans('', '', string.punctuation) #Remove punctuation 
    translator3 = Translator() #Translate to English, only if translate option is turned on
    for activity in activities:
        activity = activity.translate(translator1)
        activity = activity.translate(translator2)
        if (translate == True):
            activity = translator3.translate(activity, dest="en").text
        new_activities.append(activity)
    
    activities = new_activities 
    
    #Tokenize - split strings into sentences or words and vectorize data (store words as numbers so it's easier for a computer to understand) using Bag-Of-Words 
    vectorizer = CountVectorizer(stop_words=stopWords) #Remove stop words, or common words like 'the' and 'a' that don't mean anything. 
    vectorizer.fit(activities) 
    
    #What do your vocabulary and dictionary look like? 
    #print(vectorizer.vocabulary_) 
    #print(vectorizer.get_feature_names())
    dictionary_length = len(vectorizer.get_feature_names())
    data_length = len(activities)
   
    #For each row in activity description, create a vector and append it to the dataset. 
    activity_vectors = np.zeros([data_length, dictionary_length])
    for activity in activities:
        vector = vectorizer.transform([activity])
        vector_arr = vector.toarray()
        np.concatenate((activity_vectors, vector_arr))
    
    #-------------------------------------------------------------------------
    # From vectorized data, run some analysis to make sure it's still comparable
    #   to training data, and store dictionary for reference.     
    #-------------------------------------------------------------------------
    #Store the dictionary 
    dictionary = vectorizer.get_feature_names()
    activity_df = pd.DataFrame(activity_vectors.reshape(data_length, dictionary_length), columns=dictionary)
    #activity_df.shape #Make sure this is exactly the same as the dataset before, and append with original dataset so you can match with disease. 
    
    dataset = dataset.reset_index()
    activity_df = activity_df.reset_index()
    disease_col = dataset.disease
    activity_df = activity_df.join(disease_col, lsuffix = "left")
    activity_df['disease'] = activity_df['disease'].map({'hiv':1, 'tb':2, 'malaria':3, 'rssh':4, 'hiv/tb':5}) #Save this encoding for later! 
    
    #How are we getting NAs at this point?? 
    #-------------------------------------------------------------------------
    # Test different machine learning models using 5-fold cross-validation. 
    #   Pick the model with the highest accuracy. 
    #-------------------------------------------------------------------------
    #Separate out a validation dataset
    #X = activity_df.values #These are your independent variables - disease and the vectorized activity description. 
    X = activity_df.values
    dependent_vars = dataset[['code']]
    Y = dependent_vars.values #These are your dependent variables - just 'code'. 
    
    #Balance data with random over-sampling
    #First see how your data looks beforehand. Can you save this somewhere?
    
    ros = RandomOverSampler(random_state=0)
    X_resampled, Y_resampled = ros.fit_resample(X, Y)
    #print(sorted(Counter(Y_resampled).items()))
    
    #Set up the training data using the resampled data 
    validation_size = 0.20 #Save 20% of your data for validation
    seed = 7 #Set a random seed of 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X_resampled, Y_resampled, test_size=validation_size, random_state=seed)
    
    seed = 7 #Pick a random seed. We'll want to reset this every time to make sure the data is always split in the best way. 
    scoring = 'neg_log_loss' #We want to pick the model that minimizes log loss.  
    
    #print(X_train)
    #print(Y_train)
    # evaluate each model in turn
    results = []
    names = []
    print("Results for training dataset " + label)
    for name, model in models:
        kfold = model_selection.KFold(n_splits=5, random_state=seed) #Set up 5-fold cross-validation with n_splits here. 
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print("Log loss score: " + msg)
            
        model.fit(X_train, Y_train)
        predicted = model.predict(X_validation)
      
        print("Accuracy score for " + label + ": " + accuracy_score(Y_validation, predicted)) #96.1% accuracy on french data, 98.4% accuracy on English data 
        #print(confusion_matrix(Y_validation, predicted))
        print(classification_report(Y_validation, predicted))  

    print("")

#Run the function above on all the training datasets you specify. 
if __name__ == '__main__':   
    
    #What models do you want to test? 
    models = []
    #models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
#    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
#    models.append(('NB', GaussianNB()))
    #models.append(('SVM', SVC(gamma='auto')))     
    models.append(('RFC', RandomForestClassifier())) 
    
    #What training datasets do you want to run? 
    training_datasets = []
    training_datasets.append(("Hand-coded data, all languages", handcoded_all, stopwords_all, True, True))
    #training_datasets.append(("Teeny untranslated test", teeny_test, stopwords_all, True, True))
    
    #Print your results to a file 
    orig_stdout = sys.stdout
    f = open('data_review.txt', 'w')
    sys.stdout = f
    
    review_training_data(handcoded_all, "Handcoded All")
    
    sys.stdout = orig_stdout
    f.close() 
    
    #Print your results to a file 
    orig_stdout = sys.stdout
    f = open('model_testing.txt', 'w')
    sys.stdout = f
    
    for label, data, stopWords, translate, balanceData in training_datasets:
        test_models(label, data, stopWords, translate, models, balanceData)
    
    sys.stdout = orig_stdout
    f.close() 

