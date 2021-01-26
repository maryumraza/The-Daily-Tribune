import pickle
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import time
import re
from newspaper import Article






# SVM
with open('best_svc.pickle', 'rb') as data:
    svc_model = pickle.load(data)
    
# TFIDF 
with open('tfidf.pickle', 'rb') as data:
    tfidf = pickle.load(data)



punctuation_signs = list("?:!.,;")
stop_words = list(stopwords.words('english'))
  


category_codes = {
    'business': 0,
    'entertainment': 1,
    'politics': 2,
    'sports': 3,
    'tech': 4
}






def create_features_from_text(text):
    
    # Dataframe creation
    lemmatized_text_list = []
    df = pd.DataFrame(columns=['Content'])
    df.loc[0] = text
    df['Content_Parsed_1'] = df['Content'].str.replace("\r", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("\n", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("    ", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace('"', '')
    df['Content_Parsed_2'] = df['Content_Parsed_1'].str.lower()
    df['Content_Parsed_3'] = df['Content_Parsed_2']
    for punct_sign in punctuation_signs:
        df['Content_Parsed_3'] = df['Content_Parsed_3'].str.replace(punct_sign, '')
    df['Content_Parsed_4'] = df['Content_Parsed_3'].str.replace("'s", "")
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_list = []
    text = df.loc[0]['Content_Parsed_4']
    text_words = text.split(" ")
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
    lemmatized_text = " ".join(lemmatized_list)    
    lemmatized_text_list.append(lemmatized_text)
    df['Content_Parsed_5'] = lemmatized_text_list
    df['Content_Parsed_6'] = df['Content_Parsed_5']
    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        df['Content_Parsed_6'] = df['Content_Parsed_6'].str.replace(regex_stopword, '')
    df = df['Content_Parsed_6']
    #df = df.rename(columns={'Content_Parsed_6': 'Content_Parsed'})
    
    # TF-IDF
    features = tfidf.transform(df).toarray()
    
    return features






def get_category_name(category_id):
    for category, id_ in category_codes.items():    
        if id_ == category_id:
            return category





        
def predict_from_text(text):
    
    # Predict using the input model
    prediction_svc = svc_model.predict(create_features_from_text(text))[0]
    prediction_svc_proba = svc_model.predict_proba(create_features_from_text(text))[0]
    
    # Return result
    category_svc = get_category_name(prediction_svc)
    percent = prediction_svc_proba.max()*100
    # print(percent)
    if percent >= 70:
        return category_svc
       
    else: 
        return 'other'
    

    



def get_news_theguardian():
    
    # url definition
    url = "https://www.theguardian.com/uk"
    
    # Request
    r1 = requests.get(url)
    r1.status_code

    # We'll save in coverpage the cover page content
    coverpage = r1.content

    # Soup creation
    soup1 = BeautifulSoup(coverpage, 'html5lib')

    # News identification
    coverpage_news = soup1.find_all('h3', class_='fc-item__title')
    len(coverpage_news)
    
    number_of_articles = 20

    # Empty lists for content, links and titles
    news_contents = []
    list_links = []
    list_titles = []
    list_images = []

    for n in np.arange(0, number_of_articles):

        # We need to ignore "live" pages since they are not articles
        if "live" in coverpage_news[n].find('a')['href']:  
            continue

        link = coverpage_news[n].find('a')['href']
    
        title = coverpage_news[n].find('a').get_text()
   
        
        
        
        # Reading the content (it is divided in paragraphs)
        article = requests.get(link)

        article_link = Article(link)
        article_link.download()
        article_link.parse()
        image = article_link.top_image


        


        article_content = article.content
        soup_article = BeautifulSoup(article_content, 'html5lib')
        body = soup_article.find_all('div', class_='content__article-body from-content-api js-article__body')
        if len(body) == 0:

            continue

        x = body[0].find_all('p')
    
        
        # Unifying the paragraphs
        list_paragraphs = []
        for p in np.arange(0, len(x)):
            paragraph = x[p].get_text()
            list_paragraphs.append(paragraph)
            final_article = " ".join(list_paragraphs)

        news_contents.append(final_article)
        
        # Getting the link of the article
        
        list_links.append(link)

        # Getting the title
        
        list_titles.append(title)

        list_images.append(image)


    
    df_features = pd.DataFrame(
        {'Article Title': list_titles,
         'Article Link': list_links,
         'Image' : list_images,
         'Newspaper': 'The Guardian',
         'Content' : news_contents})

    
    return df_features








def get_news_dailymail():
    
    # url definition
    url = "https://www.dailymail.co.uk"
    
    # Request
    r1 = requests.get(url)
    r1.status_code

    # We'll save in coverpage the cover page content
    coverpage = r1.content

    # Soup creation
    soup1 = BeautifulSoup(coverpage, 'html5lib')

    # News identification
    coverpage_news = soup1.find_all('h2', class_='linkro-darkred')
    len(coverpage_news)
    
    number_of_articles = 20

    # Empty lists for content, links and titles
    news_contents = []
    list_links = []
    list_titles = []
    list_images = []

    for n in np.arange(0, number_of_articles):

        # Getting the link of the article
        link = url + coverpage_news[n].find('a')['href']
        list_links.append(link)

        # Getting the title
        title = coverpage_news[n].find('a').get_text()
        list_titles.append(title)

        # Reading the content (it is divided in paragraphs)
        article = requests.get(link)

        article_link = Article(link)
        article_link.download()
        article_link.parse()
        image = article_link.top_image
        list_images.append(image)



        article_content = article.content
        soup_article = BeautifulSoup(article_content, 'html5lib')
        body = soup_article.find_all('p', class_='mol-para-with-font')

        # Unifying the paragraphs
        list_paragraphs = []
        for p in np.arange(0, len(body)):
            paragraph = body[p].get_text()
            list_paragraphs.append(paragraph)
            final_article = " ".join(list_paragraphs)
            
        # Removing special characters
        final_article = re.sub("\\xa0", "", final_article)
        
        news_contents.append(final_article)
        

    
    df_features = pd.DataFrame(
        {'Article Title': list_titles,
         'Article Link': list_links,
         'Image' : list_images,
         'Newspaper': 'Daily Mail',
         'Content' : news_contents})
    
    return df_features









def get_news_dawn():
    
    # url definition
    url = "https://www.dawn.com/"
    
    r1 = requests.get(url)
    # print(r1.status_code)

    # We'll save in coverpage the cover page content
    coverpage = r1.content

    # Soup creation
    soup1 = BeautifulSoup(coverpage, 'html5lib')

    # News identification
    coverpage_news = soup1.find_all('h2', class_='story__title')
    
    
    number_of_articles = 5
    
    # Empty lists for content, links and titles
    news_contents = []
    list_links = []
    list_titles = []
    list_images = []



    for n in range(0, number_of_articles):
        
        # Getting the link of the article
        link = coverpage_news[n].find('a')['href']
        list_links.append(link)

        # print(link)
    
        # Getting the title
        title = coverpage_news[n].find('a').get_text()
        list_titles.append(title)


    
    
        # Reading the content (it is divided in paragraphs)
        article = requests.get(link)


        article_link = Article(link)
        article_link.download()
        article_link.parse()
        image = article_link.top_image
        list_images.append(image)

       

        article_content = article.content
        soup_article = BeautifulSoup(article_content, 'html5lib')
        x = soup_article.find_all('p')
    
        # Unifying the paragraphs
        list_paragraphs = []
        for p in np.arange(0, len(x)):
            paragraph = x[p].get_text()
            list_paragraphs.append(paragraph)
            final_article = " ".join(list_paragraphs)
        
        news_contents.append(final_article)
    
   
    df_features = pd.DataFrame(
        {'Article Title': list_titles,
         'Article Link': list_links,
         'Image' : list_images,
         'Newspaper': 'Dawn', 
         'Content' : news_contents})
    
    
    return df_features







raw_html = 'C:\\Users\\AA\\Documents\\UPDATED_WEB\\laila work\\politics.html'

_data = ''





def write_webpage_as_html(filepath = raw_html, data = ''):
    if data is '':
        data = _data
    
    with open(filepath, 'wb') as fobj:
        fobj.write(data)
        
        




def display_news(guardian_features, daily_features, dawn_features):


    df_politics = pd.DataFrame(columns = ['Article Title', 'Article Link', 'Image', 'Newspaper', 'Content'])
    df_business = pd.DataFrame(columns = ['Article Title', 'Article Link', 'Image', 'Newspaper', 'Content'])
    df_sports = pd.DataFrame(columns = ['Article Title', 'Article Link', 'Image', 'Newspaper', 'Content'])
    df_entertainment = pd.DataFrame(columns = ['Article Title', 'Article Link', 'Image', 'Newspaper', 'Content'])
    df_tech = pd.DataFrame(columns = ['Article Title', 'Article Link', 'Image', 'Newspaper', 'Content'])
    df_other = pd.DataFrame(columns = ['Article Title', 'Article Link', 'Image', 'Newspaper', 'Content'])


    for y in (guardian_features, daily_features, dawn_features):
        for index, row in y.iterrows():
            text = row['Content']
            category = predict_from_text(text)

            description = text[0:130]+'...'



            if category == 'politics':
                df_politics = df_politics.append({'Article Title': row['Article Title'], 'Article Link': row['Article Link'], 'Image' : row['Image'], 'Newspaper': row['Newspaper'],  'Content' : description}, ignore_index=True)



            elif category == 'business':
                df_business = df_business.append({'Article Title': row['Article Title'], 'Article Link': row['Article Link'], 'Image' : row['Image'], 'Newspaper': row['Newspaper'],  'Content' : description}, ignore_index=True)

                


            elif category == 'sports':
                df_sports = df_sports.append({'Article Title': row['Article Title'], 'Article Link': row['Article Link'], 'Image' : row['Image'], 'Newspaper': row['Newspaper'],  'Content' : description}, ignore_index=True)




            elif category == 'entertainment':
                df_entertainment = df_entertainment.append({'Article Title': row['Article Title'], 'Article Link': row['Article Link'], 'Image' : row['Image'], 'Newspaper': row['Newspaper'],  'Content' : description}, ignore_index=True)




            elif category == 'tech':
                df_tech = df_tech.append({'Article Title': row['Article Title'], 'Article Link': row['Article Link'], 'Image' : row['Image'], 'Newspaper': row['Newspaper'],  'Content' : description}, ignore_index=True)




            else:
                df_other = df_other.append({'Article Title': row['Article Title'], 'Article Link': row['Article Link'], 'Image' : row['Image'], 'Newspaper': row['Newspaper'],  'Content' : description}, ignore_index=True)





                
        
    political_list = [[row['Article Title'] for index,row in df_politics.iterrows()] , [row['Article Link'] for index,row in df_politics.iterrows()], [row['Image'] for index,row in df_politics.iterrows()], [row['Content'] for index,row in df_politics.iterrows()]]
    business_list = [[row['Article Title'] for index,row in df_business.iterrows()] , [row['Article Link'] for index,row in df_business.iterrows()], [row['Image'] for index,row in df_business.iterrows()], [row['Content'] for index,row in df_business.iterrows()]]
    sports_list = [[row['Article Title'] for index,row in df_sports.iterrows()] , [row['Article Link'] for index,row in df_sports.iterrows()], [row['Image'] for index,row in df_sports.iterrows()], [row['Content'] for index,row in df_sports.iterrows()]]
    enter_list = [[row['Article Title'] for index,row in df_entertainment.iterrows()] , [row['Article Link'] for index,row in df_entertainment.iterrows()], [row['Image'] for index,row in df_entertainment.iterrows()], [row['Content'] for index,row in df_entertainment.iterrows()]]
    tech_list = [[row['Article Title'] for index,row in df_tech.iterrows()] , [row['Article Link'] for index,row in df_tech.iterrows()], [row['Image'] for index,row in df_tech.iterrows()], [row['Content'] for index,row in df_tech.iterrows()]]
    other_list = [[row['Article Title'] for index,row in df_other.iterrows()] , [row['Article Link'] for index,row in df_other.iterrows()], [row['Image'] for index,row in df_other.iterrows()], [row['Content'] for index,row in df_other.iterrows()]]


    


    return political_list, business_list, sports_list, enter_list, tech_list, other_list






    
guardian_features = get_news_theguardian()
daily_features  = get_news_dailymail()
dawn_features = get_news_dawn()


political_list, business_list, sports_list, enter_list, tech_list, other_list = display_news(guardian_features, daily_features, dawn_features)








