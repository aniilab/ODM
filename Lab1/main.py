from matplotlib import dates
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

def download_data_from_csv(file_path):
    data = pd.read_csv(file_path, encoding='ISO-8859-1', names=['target', 'ids', 'date', 'flag', 'user', 'text'])
    return data

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)  # Remove special characters
    text = text.lower()  # Convert to lower case
    return text

def tokenize_and_remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_words = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_words)

def assign_sentiment_score(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def plot_sentiment_distribution(sentiments):
    plt.figure(figsize=(8, 6))
    plt.hist(sentiments, bins=20, color='pink')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.show()
        
def plot_sentiment_category_proportions(df):
    df['sentiment_category'] = df['sentiment_score'].apply(categorize_sentiment)
    sentiment_counts = df['sentiment_category'].value_counts()
    colors = ['lightgreen', 'lightcoral', 'lightgray']
    explode = (0.1, 0, 0)
    
    plt.figure(figsize=(8, 8))
    plt.pie(sentiment_counts, explode=explode, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title('Proportion of Each Sentiment Category')
    plt.show()
        
def categorize_sentiment(score):
    if score > 0.1:
        return 'Positive'
    elif score < -0.1:
        return 'Negative'
    else:
        return 'Neutral'
    
def plot_hourly_sentiment_time_series(df):
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    
    hourly_sentiment = df.resample('H', on='date')['sentiment_score'].mean()
    
    moving_average = hourly_sentiment.rolling(window=3).mean() 

    plt.figure(figsize=(12, 6))
    plt.plot(hourly_sentiment.index, hourly_sentiment, label='Hourly Average', color='thistle', alpha=0.5, marker='o', linestyle='-', markersize=5)
    plt.plot(moving_average.index, moving_average, label='3-hour Moving Average', color='darkviolet', linewidth=2)
    plt.title('Hourly Average Sentiment Score with Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    plt.legend()
    plt.show()
    
def generate_sentiment_word_clouds(df):
   df['sentiment_category'] = df['sentiment_score'].apply(lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral'))
   
   for category in ['positive', 'negative', 'neutral']:
        plt.figure(figsize=(8, 8))
        text = ' '.join(df[df['sentiment_category'] == category]['clean_text'])
        wordcloud = WordCloud(background_color='white', colormap = 'RdPu', max_words=200, contour_width=3, contour_color='darkslateblue').generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'{category.capitalize()} Sentiment Word Cloud')
        plt.axis('off')
        plt.show()
    

def main():
    download_nltk_data()
    df = download_data_from_csv('dataset.csv')
    
    df['clean_text'] = df['text'].apply(lambda x: tokenize_and_remove_stop_words(clean_text(x)))
    
    df['sentiment_score'] = df['clean_text'].apply(assign_sentiment_score)

    plot_sentiment_distribution(df['sentiment_score'])
    plot_hourly_sentiment_time_series(df)
    plot_sentiment_category_proportions(df)
    
    generate_sentiment_word_clouds(df)

if __name__ == "__main__":
    main()
