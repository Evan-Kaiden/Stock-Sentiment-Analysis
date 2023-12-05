from tqdm import tqdm

import requests
from bs4 import BeautifulSoup

import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import softmax

import matplotlib.pyplot as plt

import datetime

# create a tokenizer object
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

# fetch the pretrained model 
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

tickers = ['SPY']

def compound_score(pos, neg, neu):
    # normalize the values to ensure they sum up to 1
    total = pos + neg + neu
    pos = pos / total
    neg = neg / total
    neu = neu / total
    
    # calculate the compound score
    compound = (pos - neg) / (pos + neg + neu)
    return compound

def sentim_analyzer(text, tokenizer, model):
    # Pre-process input phrase
    input = tokenizer(text, padding = True, truncation = True, return_tensors='pt')

    # Estimate output
    output = model(**input)

    # Pass model output logits through a softmax layer.
    predictions = softmax(output.logits, dim=-1)
    return compound_score(predictions[0][0], predictions[0][1],  predictions[0][2]).tolist()

def lemmatize_text(list):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = [lemmatizer.lemmatize(txt) for txt in list]
    return lemmatized_sentence

def simplify(text):
    import re
    from nltk.tokenize import word_tokenize

     # remove unwanted characters
    text = re.sub(r'[^\w\s]', '', text)

    # convert to lowercase
    text = text.lower()

    # tokenize the text
    text_words = word_tokenize(text)

    # Get a list of stop words, words that are not relevant to sentiment analysis, and business-specific stop words
    stop_words = set(open('/Users/evankaiden/Documents/pythonprojects/Web Scraping/trading_bot/stopwords.txt', 'r').readlines())

    # Remove stop words
    text_words = [word for word in text_words if word not in stop_words]

    # Lemmatize the text
    text_words = lemmatize_text(text_words)

    # join the processed words back into a single string
    text = ' '.join(text_words)
    text = text.replace('bear', 'negative').replace('bull', 'positive')
    
    return text


def get_text(link):
    print(link)
    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    headers = {'user-agent' : user_agent}
    res = requests.get(link, headers=headers)
    yahoo = BeautifulSoup(res.text, 'html.parser')

    try:
        button_link = yahoo.find('a', class_='link caas-button').get('href')
        if 'video' in button_link:
            return None
        if 'fool.com' in button_link.lower():
            fool = requests.get(button_link, headers=headers)
            fool = BeautifulSoup(fool.text, 'html.parser')
            article_body = fool.find('div', class_='tailwind-article-body')
            article_text = article_body.find_all('p')
            article_text = ' '.join(article.text for article in article_text)
            return article_text
        elif 'investors.com' in button_link.lower():
            investor = requests.get(button_link, headers=headers)
            investor = BeautifulSoup(investor.text, 'html.parser')
            article_body = investor.find('div', class_='single-post-content post-content drop-cap')
            article_text = article_body.find_all('p')
            article_text = ' '.join(article.text for article in article_text)
            return article_text
        elif 'bizjournals.com' in button_link.lower():
            bizjournal = requests.get(button_link, headers=headers)
            bizjournal = BeautifulSoup(bizjournal.text, 'html.parser')
            article_text = bizjournal.find_all('p')
            article_text = ' '.join(article.text for article in article_text)
            return article_text
        elif 'wsj.com' in button_link.lower():
            wallstreet = requests.get(button_link, headers=headers)
            wallstreet = BeautifulSoup(wallstreet.text, 'html.parser')
            article_body = wallstreet.find('section', class_='css-1ducvg2-Container e1d75se20')
            article_text = article_body.find_all('p', class_='css-xbvutc-Paragraph e3t0jlg0')
            return article_text
        elif 'barrons.com' in button_link.lower():
            barrons = requests.get(button_link, headers=headers)
            barrons = BeautifulSoup(barrons.text, 'html.parser')
            article_body = barrons.find('div', class_="article__body article-wrap at16-col16 barrons-article-wrap crawler")
            article_text = article_body.find_all('p')
            article_text = ' '.join(article.text for article in article_text)
            return article_text
        elif 'ft.com' in button_link.lower():
            ft = requests.get(button_link, headers=headers)
            ft = BeautifulSoup(ft.text, 'html.parser')
            article_body = ft.find('div', class_='article__content-body n-content-body js-article__content-body')
            article_text = article_body.find_all('p')
            article_text = ' '.join(article.text for article in article_text)
            return article_text
        elif 'thestreet.com' in button_link.lower():
            street = requests.get(button_link, headers=headers)
            street = BeautifulSoup(street.text, 'html.parser')
            article_body = street.find('div', class_='article__body article-back-header__body')
            article_text = article_body.find_all('p')
            article_text = ' '.join(article.text for article in article_text)        
            return article_text
    except:
        article_body = yahoo.find('div', class_='caas-body')
        article_text = article_body.find_all('p')
        article_text = ' '.join(article.text for article in article_text)
        return article_text

    


    

def get_info(ticker):
    url = 'https://finviz.com/quote.ashx?t=' + ticker 
    html = requests.get(url, headers={'user-agent' : 'my-app'})
    soup = BeautifulSoup(html.text, 'html.parser')
    news_table = soup.find('table', class_='fullview-news-outer')
    tables = news_table.find_all('tr')
    parsed_data = []

    for table in (tables):
        date_time = table.find('td').text.split()
        link = table.find('a').get('href')

        if len(date_time) == 1:
            time = date_time[0]
        else:
            date = date_time[0]
            time = date_time[1]

        text = get_text(link)

        if len(text) > 512:
            text = text[:512]
# remove any article duplicates 
        parsed_data.append([ticker, date, time, text])
    return parsed_data



def main():
    data = []
    for ticker in tqdm(tickers):
        data.extend(get_info(ticker))
   
    df = pd.DataFrame(data, columns=['ticker', 'date', 'time', 'text'])

    df['compound'] = df['text'].apply(lambda x: sentim_analyzer(x, tokenizer, model))
    df['date'] = df['date'].replace({'Today': datetime.datetime.now().date()})

    df['date'] = pd.to_datetime(df.date).dt.date

    # Define the date time threshold
    date_time_threshold = pd.to_datetime('2023-01-01')

    # Create a boolean mask
    mask = df['date'] > date_time_threshold

    # Use the mask to select the rows that you want to keep
    df = df[mask]
    mean_df = df.groupby(['ticker', 'date']).mean()
    mean_df = mean_df.unstack()
    mean_df = mean_df.xs('compound', axis='columns').transpose()
    mean_df.plot(kind='bar')

    plt.show()

if __name__ == '__main__':
    main()