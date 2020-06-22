# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 13:53:44 2020

@author: User
"""

import requests
import MySQLdb
import nltk
from nltk.corpus import wordnet
#from nltk import word_tokenize
import string
from bs4 import BeautifulSoup
import threading
from multiprocessing import Pool, Queue
from functools import partial
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from requests_html import HTMLSession
from requests_html import AsyncHTMLSession
#from nltk.stem import WordNetLemmatizer 
#SQL connection data to connect and save the data in
HOST = "localhost"
USERNAME = "MPW20AMJ01"
PASSWORD = "miniproject"
DATABASE = "reviews"

# Open database connection
#db = MySQLdb.connect(HOST, USERNAME, PASSWORD, DATABASE)

#Search Query
# baseURL = "https://www.amazon.in/s?k="
# URL = baseURL+searchQuery
# #Setting header info
header={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36'}
# #Getting content of web page
# search_response=requests.get(URL,headers=header)
# #Status Code for Success = 200
#search_response.status_code
# insert request cookies within{}
cookie={}
def getAmazonSearch(search_query):
    url="https://www.amazon.in/s?k="+search_query
    # options = Options()
    # options.add_argument("--headless")
    driver = webdriver.Chrome(r"C:/Program Files (x86)/Google/Chrome/Application/Driver/chromedriver.exe")
    driver.implicitly_wait(10)
    driver.get(url)

    #print(url)
    page=requests.get(url,cookies=cookie,headers=header)
    
    if page.status_code==200:
        #print (driver.page_source)
        return driver.page_source
    else:
        return "Error"

def Searchasin(asin, index):
    url="https://www.amazon.in/dp/"+asin[index]
    session = HTMLSession()
    r = session.get(url)
    #print(url)
    page=requests.get(url,cookies=cookie,headers=header)
    if page.status_code==200:
        return r.content
    else:
        return "Error"

def Searchreviews(review_link, index):
    #url="https://www.amazon.in"+review_link
    url="https://www.amazon.in"+review_link[index]+'&pageNumber='+str(0)

    session = HTMLSession()
    r = session.get(url)
    
    #print(url)
    page=requests.get(url,cookies=cookie,headers=header)
    if page.status_code==200:
        return r.content
    else:
        return "Error"
    
#def reduce_lengthening(text):
 #   pattern = re.compile(r"(.)\1{2,}")
  #  return pattern.sub(r"\1\1", text)

def find_noun_object(nouns, aspect):
    for noun in nouns:
        if noun.text == aspect:
            return noun
        

def find_relevant_nouns(nouns, initial_nouns, threshold):
    open_set = initial_nouns
    #print(open_set)
    closed_set = set()

    while open_set:
        candidate = open_set.pop()
        #print(int(candidate))
        
        for noun in nouns:
            if candidate.similarity(noun) >= threshold and noun not in closed_set:
                open_set.add(noun)
        closed_set.add(candidate)
        
    return closed_set

def get_nouns_in_aspects(nouns, aspects, threshold, general_aspects):
    aspects_dict = {}
    for aspect in aspects:
        obj = find_noun_object(nouns, aspect)
        if obj is not None:
            initial_nouns = {obj}
            final_nouns = find_relevant_nouns(nouns, initial_nouns, threshold)
            for key in general_aspects.keys():
                if key not in aspects_dict.keys():
                    aspects_dict[key] = {noun.text for noun in final_nouns}
    return aspects_dict

if __name__ == '__main__':
    searchQuery = input("Enter a Search Query : ")
    
    product_names=[]
    reviewWithProdName = dict()
    response=getAmazonSearch('titan+men+watches')
    soup=BeautifulSoup(response)
    for i in soup.findAll("span",{"class":"a-size-base-plus a-color-base a-text-normal"}): # the tag which is common for all the names of products
        product_names.append(i.text.replace("-"," "))
        for i in product_names:
            if i not in reviewWithProdName:
                reviewWithProdName[i.replace("\'","")] = []

    data_asin=[]
    response=getAmazonSearch('titan+men+watches')
    soup=BeautifulSoup(response)
    for i in soup.findAll("div",{'class':"sg-col-4-of-24 sg-col-4-of-12 sg-col-4-of-36 s-result-item s-asin sg-col-4-of-28 sg-col-4-of-16 sg-col sg-col-4-of-20 sg-col-4-of-32"}):
        data_asin.append(i['data-asin'])

    
    
    threadPoolResponses1 = []       #collect the threadpool responses
    
    # A thread pool basically spawns the given number of processes on the called function
    # So the "Searchasin" function can be called concurrently with 10 links at the same time
    # Since each thread in the pool has separate memory space, to collect all the responses, use "pool.imap_unordered"
    # "pool.imap_unordered" takes the partial function and an iterable which can be used to iterate the data
    print("data_as_in", data_asin)
    func = partial(Searchasin, data_asin)
    with Pool(processes=15) as pool:
        for result in pool.imap_unordered(func, range(len(data_asin))):
            threadPoolResponses1.append(result)
        pool.terminate()
        pool.join()

    """ 
    This is the same loop you wrote, but instead of calling the "Searchasin" function again and again in a loop,
    all responses are alread there in "threadPoolResponses1".
    """
    link=[]
    #print("responses:",threadPoolResponses1)
    for i in threadPoolResponses1:
        response=i
        soup=BeautifulSoup(response)
        for i in soup.findAll("a",{'data-hook':"see-all-reviews-link-foot"}):
            link.append(i['href'])
    #print(link)
    threadPoolResponses2 = []               #collect the threadpool responses
    #print("link", link)
    func = partial(Searchreviews, link)
    with Pool(processes=15) as pool:
        for result in pool.imap_unordered(func, range(len(link))):
            threadPoolResponses2.append(result)
        pool.terminate()
        pool.join()

    reviews = []
    newR = []
    for i in threadPoolResponses2:
        response=i
        newR = []
        soup=BeautifulSoup(response)
        for i in soup.findAll("span",{'data-hook':"review-body"}):
            if i.text not in newR:
                newR.append(i.text)
        if newR not in reviews:
            reviews.append(newR)

    #print("reviews:",reviews)
    
    j = 0
    for i in reviewWithProdName.keys():
        i = i.replace("`","")
        if(j<len(reviews)):
            if reviews[j] not in reviewWithProdName[i]:
                ins = list( i[1:len(i)-1] for i in reviews[j])
                reviewWithProdName[i].extend(ins)
        j = j+1

    #print("reviewWithProductName after:", reviewWithProdName)

    # Prepare SQL query to INSERT a record into the database.
    db = MySQLdb.connect(HOST, USERNAME, PASSWORD, DATABASE)
    # prepare a cursor object using cursor() method
    cursor = db.cursor()
    # Prepare SQL query to INSERT a record into the database.
    for key in reviewWithProdName.keys():
        for i in range(0,len(reviewWithProdName[key])):
            #print("key",key)
            #print("review:",reviewWithProdName[key][i])
            reviewWithProdName[key][i] = reviewWithProdName[key][i].replace('\'','')
            reviewWithProdName[key][i] = "".join(i for i in reviewWithProdName[key][i] if ord(i)<128)
            query = "INSERT INTO productreviews(ProductName,ProductReviews) VALUES('"+ key +"' ,'"+ reviewWithProdName[key][i] +"');"
            #print("query", query)
            cursor.execute(query)
    query = "SELECT * FROM productreviews"
    cursor.execute(query)
    result = cursor.fetchall() 
    final_result = [list(i) for i in result]
    
    dictRev = dict()
    dictRev1 = dict()
    for subList in final_result:
        if subList[0] not in dictRev.keys():
            dictRev[subList[0]] = []
            dictRev1[subList[0]] = []
        dictRev[subList[0]].append(subList[1])
    print(dictRev["All Black Analog Dial Mens Watch 1698NM01"][0])
    
    
    #tokenizing the sentences
    for key in dictRev.keys():
        for subList in range(0,len(dictRev[key])):
            dictRev[key][subList] = nltk.sent_tokenize(dictRev[key][subList]) #.split()
     
    print(dictRev["All Black Analog Dial Mens Watch 1698NM01"][3][0])
    
    #POS Tagging and Getting O/P as (Adj, Adj, base form of noun)
    import sys
    sys.path.append(".")
    adjPair = []
    from try1 import AdjNounExtractor
    ap = AdjNounExtractor()
    for key in dictRev.keys():
        adjPair = []
        for rev in dictRev[key]:
            for val in rev:
                
                #ap = list(ap)
                adjPair.append(ap.extract(val))
        dictRev1[key].append(adjPair)
        
    print(dictRev1["All Black Analog Dial Mens Watch 1698NM01"][0][1][0])
    
    #Extracting the nouns from the POS tags and counting the number of occurences
    nouns_count = dict()
    for key in dictRev1.keys():
        if key not in nouns_count.keys():
            nouns_count[key] = dict()
        for i in dictRev1[key]:
            for j in i:
                for k in j:
                    w = k[2].lower()
                    if w not in nouns_count[key].keys():
                        nouns_count[key][w] = 1
                    else:
                        nouns_count[key][w] = nouns_count[key][w]+1
                        
    #Sorting the Nouns according the count of occurences
    r = dict()
    for key in nouns_count.keys():
        if key not in r.keys():
            r[key] = []
        results = sorted(nouns_count[key].items(), key=lambda x: x[1], reverse=True) 
        r[key].extend(results)
    r["All Black Analog Dial Mens Watch 1698NM01"][0]   
            
    ###Finding relevant nouns###########

    #print("finding relevant nouns")
    #Nouns which are relaed to the reviews
    # Load spaCy model
    import spacy
    nlp = spacy.load('en_core_web_lg')                
    # Use spaCy model to get tokens
    nouns = []
    # Remove docs that have more than one token
    s = dict()
    for key in r.keys():
        if key not in s.keys():
            s[key] = []
        for t in r[key]:
            s[key].append(t[0])
    s["All Black Analog Dial Mens Watch 1698NM01"]  

    #Converting each review to a spacy doc     
    docs = dict()
    for key in s.keys():
        if key not in docs.keys():
            docs[key] = []
        docs[key].extend([nlp(noun) for noun in s[key]])
    #Converting each spacy doc to token     
    tkn = dict()
    for key in docs.keys():
        if key not in tkn.keys():
            tkn[key] = []
        tkn[key].extend([doc[0] for doc in docs[key] if len(doc) == 1])
    len(tkn["All Black Analog Dial Mens Watch 1698NM01"])
    
    #Checking if the tokens are there in the vocab of the spacy model
    tokensVocab = dict()
    for key in tkn.keys():
        if key not in tokensVocab.keys():
            tokensVocab[key] = []
        for token in tkn[key]:
            if not token.is_oov:
                tokensVocab[key].append(token)
    len(tokensVocab["All Black Analog Dial Mens Watch 1698NM01"][0])
    
    #Converting each words to vectors
    vectors = dict()
    for key in tokensVocab.keys():
        if key not in vectors.keys():
            vectors[key] = dict()
        for token in tokensVocab[key]:
            if token not in vectors[key].keys():
                vectors[key][str(token)] = []
            vectors[key][str(token)].extend(token.vector)
    vectors["All Black Analog Dial Mens Watch 1698NM01"]
    
    
    #Dynamic Aspects for the product in consideration
    general_aspects = {
    'Look': {'looking', 'design','color','colour','belt','dial','look','looks','style'},
    'Product' : {'watch','it','product'},
    'Cost': {'price', 'value','deal','offer','sale','cost','purchase'},
    'Quality'  : {'brand','quality','finishing'}
}
    #Common Aspects : Keys of the general_aspects dict
    general_aspects.keys()
    indProdAspect = dict()
    from functools import reduce
    aspects = reduce(lambda x, y: x.union(y), general_aspects.values())
    threshold = 0.7
    for key in tokensVocab.keys():
        if key not in indProdAspect.keys():
            indProdAspect[key] = []
        indProdAspect[key].extend(get_nouns_in_aspects(tokensVocab[key], aspects, threshold,general_aspects))
    indProdAspect["All Black Analog Dial Mens Watch 1698NM01"][0]
    
    
    #Sentiment to each aspect word based on the adjectives
    from textblob import TextBlob
    dictRev1["All Black Analog Dial Mens Watch 1698NM01"][0][-1][-1][-1]
       
    sentimentDict = dict()
    txt = " "
    for key in dictRev1.keys():
        txt = " "
        if key not in sentimentDict.keys():
            sentimentDict[key] = dict()
        for val in dictRev1[key]:
            if len(val)!=0:
                for j in val:
                    for k in j:
                        w = k[2].lower()
                        for v in general_aspects.keys():
                            value = list(general_aspects[v])
                            for v1 in list(value):
                                if w == v1:
                                    w = w.replace(w, v)
                            
                        if w in indProdAspect[key]:
                            if w not in sentimentDict[key].keys():
                                sentimentDict[key][w] = []
                            txt = k[0]+" "+k[1]
                            pol = TextBlob(txt).sentiment.polarity
                            sentimentDict[key][w].append(pol)
                            
    #Average polaroty : Obtaining an Average Polarity
    avgPol = dict()            
    for key in sentimentDict.keys():
        if key not in avgPol.keys():
            avgPol[key] = dict()
        for k in sentimentDict[key].keys():
            if k not in avgPol[key].keys():
                avgPol[key][k]= int()
            avgPolScore = (sum(sentimentDict[key][k]))/len(sentimentDict[key][k])
            avgPol[key][k] = avgPolScore
            
    aspectTerms = list(general_aspects.keys())
    
    #Flattening the score
    max_scores = {aspect: -1.0 for aspect in aspectTerms}
    min_scores = {aspect: 1.0 for aspect in aspectTerms}
    
    rating = dict()
    for key in avgPol.keys():
        if key not in rating.keys():
            rating[key] = dict()
        for k in avgPol[key].keys():
            if k not in rating[key].keys():
                rating[key][k] = int()
            min_scores[k] = min(min_scores[k], avgPol[key][k])
            max_scores[k] = max(max_scores[k], avgPol[key][k])
            
    #Scaling      
    diff = {aspect: max_scores[aspect] - min_scores[aspect] for aspect in aspectTerms}
    for key in rating.keys():
        for k in rating[key].keys():
            if diff[k]>0.0:
                rating[key][k] = (avgPol[key][k] - min_scores[k]) / diff[k] * 5
                rating[key][k] = round(rating[key][k],2)
    rating["All Black Analog Dial Mens Watch 1698NM01"]       
    # Prepare SQL query to INSERT a record into the database.
    #Inserting into Database, the ratings of each aspect
    db = MySQLdb.connect(HOST, USERNAME, PASSWORD, DATABASE)
    # prepare a cursor object using cursor() method
    cursor = db.cursor()
    # Prepare SQL query to INSERT a record into the database.
    l = []
    for key in rating.keys():
        ProductName = key
        #asp = rating[key].keys()
        
        Cost = 'null'
        Quality = 'null'
        Look = 'null'
        Product = 'null'
        
        if 'Product' in rating[key].keys():
            Product = rating[key]['Product']
        if 'Cost' in rating[key].keys():
            Cost = rating[key]['Cost']
        if 'Look' in rating[key].keys():
            Look = rating[key]['Look']
        if 'Quality' in rating[key].keys():
            Quality = rating[key]['Quality']
        #print(Quality)
        
        query = "INSERT into ratingsTable(ProductName, Cost, Quality, Look, Product) values ('"+key+ "','" + str(Cost)+"','" + str(Quality) +"','" + str(Look) +"','"+str(Product)+"');"
        print(query)
        cursor.execute(query)
    #QUERY FOR THE VENDOR
    query = "SELECT ProductName, COST FROM ratingsTable WHERE COST>3"
    cursor.execute(query)
    result = cursor.fetchall() 
    
    result = [list(i) for i in result]
    result[0]
    l = [i[0] for i in result]   
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
        #ax = fig.add_axes([0,0,1,1])
#langs = ['C', 'C++', 'Java', 'Python', 'PHP']
    y = [i[1] for i in result]
    ax.bar(l,y)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.show()
        
    db.commit()
            #query = "INSERT INTO productreviews(ProductName,ProductReviews) VALUES('"+ key +"' ,'"+ reviewWithProdName[key][i] +"');"
            #print("query", query)
            #cursor.execute(query)
    
            
            
  
    

    """
    for i in dict.keys():
    name = i
    aspect = i.keys()

    product = 'null'
    cost = 'null'
    look = 'null'
    quality = 'null'

    if 'Product' in aspect.keys():
        product = aspect['Product']
    if 'Look' in aspect.keys():
        product = aspect['Look']
    if 'Quality' in aspect.keys():
        product = aspect['Quality']
    if 'Cost' in aspect.keys():
        product = aspect['Cost']
    
    query = "INSERT into tablename(Product, Quality, Look, Cost) values ('"+product + "','" + quality +"','" + look "','" + cost + ";"
    print(query)
    """
    """
    
    #Removal of Punctuation
    l = []
    for key in dictRev.keys():
        for value in dictRev[key]:
            l = []
            for word in value:
                for ch in word: 
                    if ch in set(string.punctuation):
                        word = word.replace(ch, " ")
                
                if word not in list(string.punctuation):
                    l.append(word)
            dictRev1[key].append(l)
    print(dictRev1["All Black Analog Dial Mens Watch 1698NM01"][2])
   
   
    
    #removal of stop words
    l1 = []
    for key in dictRev1.keys():
        for value in dictRev1[key]:
            l1 = []
            for word in value:
                #for w in word:
                if word.lower() not in list(stopwords.words("english")):
                    l1.append(word.lower())
            dictRev2[key].append(l1)
    print(dictRev2["All Black Analog Dial Mens Watch 1698NM01"][2])
    
           
    
    #Converting each review into a string
    #import re
    from textblob import TextBlob
    for k in dictRev2.keys():
        l2 = []
        for v in dictRev2[k]:
            s = " "
            for word in v:
                b = TextBlob(word)
                b = str(b.correct())
                s = s+" "+b
            l2.extend(s)
        if l2 not in dictRev3[k]:
            dictRev3[k].extend(l2)
    print(dictRev3["All Black Analog Dial Mens Watch 1698NM01"][0].strip())
    """
    """
    from nltk import FreqDist
    fd = dict()
    for key in dictRev3.keys():
        if key not in fd.keys():
            fd[key]= []
        for val in dictRev3[key]:
            fq = FreqDist(val.split())
        fd[key].append(dict(fq))
    ############# PREPROCESSING IS COMPLETE ###################

    import spacy
    nlp = spacy.load("en_core_web_sm")
    print("################## ASPECT TERM EXTRACTION #########################")
    #req_tag = ['NN']
    aspectPairs = []
    i = 0
    for key in dictRev3.keys():
        for rev in dictRev3[key]:
            rev = rev.strip()
            doc = nlp(rev)
            aspectPairs = []
            for j, token in enumerate(doc):
                i += 1
                if token.pos_ not in ('NOUN','PROPN'):
                    continue
                for k in range(j+1, len(doc)):
                    if doc[k].pos_ == 'ADJ' or doc[k].pos_ == 'ADV':
                        aspectPairs.append([token.lemma_, doc[k]])
            dictRev4[key].extend(aspectPairs)
                #print(extracted_words)
    print(dictRev4["All Black Analog Dial Mens Watch 1698NM01"])
    
    aspectDict = dict()
    for key in dictRev4.keys():
        if key not in aspectDict.keys():
            aspectDict[key] = dict()
        for l in dictRev4[key]:
            if l[0] not in aspectDict[key]:
                aspectDict[key][l[0]] = []
            aspectDict[key][l[0]].append(l[1])   
    aspectDict["All Black Analog Dial Mens Watch 1698NM01"]

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

    sid_obj = SentimentIntensityAnalyzer() 
    sumPol = 0
    sentimentScore = dict()
    for key in aspectDict.keys():
        if key not in sentimentScore.keys():
            sentimentScore[key] = dict()
        for k in aspectDict[key]:
            if k not in sentimentScore[key].keys():
                sentimentScore[key][k] = []
            for word in aspectDict[key][k]:
                #print(word)
                sentiment_dict = sid_obj.polarity_scores(str(word))
                #word = TextBlob(str(word))
                #pol = word.sentiment.polarity
                sumPol = sumPol+sentiment_dict['compound']
            sentimentScore[key][k] = int(sumPol//len(aspectDict[key][k]))
          
    #Filtering the top 5 aspects based on the sentiment score as in the above dictionary
    from collections import Counter 
    top5Dict = dict()
    for key in sentimentScore.keys():
        myDict = Counter(sentimentScore[key])
        top5 = myDict.most_common(5)
        # Prepare SQL query to INSERT a record into the database.
        db = MySQLdb.connect(HOST, USERNAME, PASSWORD, DATABASE)
        # prepare a cursor object using cursor() method
        cursor = db.cursor()
        query = "INSERT INTO sentimentscore(productName, aspect1, aspect2, aspect3, aspect4, aspect5) VALUES('"+ key +"' ,'"+ +"');"
            
    
    # Prepare SQL query to INSERT a record into the database.
    for key in sentimentScore.keys():
        for k in sentimentScore[key]:
                       
            
            #print("query", query)
            cursor.execute(query)
    """       

            
    #print(aspectDict["All Black Analog Dial Mens Watch 1698NM01"]) 
 
    
   