"""
Script documentation :

This script retrieves a user's tweets from Twitter, cleans them up and processes them to extract useful information.

To use this script, you must first authenticate yourself using the authentication.authenticate() function.

For more information on using this script, use help(script_name).
"""

################################################################################
# File  : MyBOT.py
# Author : 18017952
################################################################################

################################################################################
# Import of external functions :
import re
import tweepy
import spacy
import stylecloud
import pandas as pd
import unicodedata
import os
import time
import argparse
import sqlite3
import configparser
from datetime import datetime, timedelta, date
import authentification
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

################################################################################
# Constants :
TWEET_MODE = 'extended'
################################################################################
# Local definition of functions :
def get_tweets(screen_name, api):
    """
    Retrieves tweets from the specified user using the specified Twitter API.

    Parameters
    ----------
    screen_name : str
        Screen name of the user whose tweets are to be retrieved.
    api: tweepy.API
        Twitter API object used to access the user's tweets.

    Returns
    -------
    list of tweepy.models.Status
        List of the user's tweets.
    """
    tweets = api.user_timeline(screen_name=screen_name,
                               # 200 is the maximum number allowed
                               count=200,
                               exclude_replies=True,
                               include_rts=False,
                               # Necessary to keep the full text
                               # otherwise only the first 140 words are extracted
                               tweet_mode=TWEET_MODE
                               )
    return tweets

def get_all_tweets(screen_name, api):
    """
    Retrieves all tweets from the specified user using the specified Twitter API.

    Parameters
    ----------
    screen_name : str
        Screen name of the user whose tweets are to be retrieved.
    api: tweepy.API
        Twitter API object used to access the user's tweets.

    Returns
    -------
    list of tweepy.models.Status
        List of all the user's tweets.
    """
    tweets = get_tweets(screen_name, api)
    all_tweets = []
    all_tweets.extend(tweets)
    oldest_id = tweets[-1].id
    while True:
        tweets = api.user_timeline(screen_name=screen_name,
                                   # 200 is the maximum number allowed
                                   count=200,
                                   exclude_replies=True,
                                   include_rts=False,
                                   max_id=oldest_id - 1,
                                   # Necessary to keep the full text
                                   # otherwise only the first 140 words are extracted
                                   tweet_mode=TWEET_MODE
                                   )
        if not len(tweets):
            break
        oldest_id = tweets[-1].id
        all_tweets.extend(tweets)

    return all_tweets

def get_tweets_by_date(screen_name, api):
    # Checks that screen_name is a valid string
    if not isinstance(screen_name, str):
        raise ValueError('screen_name doit être une chaîne de caractères')
    # Retrieves all the user's tweets
    all_tweets = get_all_tweets(screen_name, api)
    # Create an empty list to store filtered tweets
    filtered_tweets = []
    # Gets the current date and time in UTC
    now = datetime.utcnow()
    # Sets the date range (1 year from now)
    start_date = now - timedelta(days=365)
    end_date = now
    # Browse all the user's tweets
    for tweet in all_tweets:
        # Remove the time zone offset from the tweet.created_at object
        created_at_datetime = tweet.created_at.replace(tzinfo=None)
        # If the date of publication of the tweet is between start_date and end_date
        if start_date <= created_at_datetime <= end_date:
            # Ajoute le tweet à la liste des tweets filtrés
            filtered_tweets.append(tweet)
    # Returns the list of filtered tweets
    return filtered_tweets

def cleaning_tweets(tweets):
    """
    Clean up a tweet by removing links, hashtags, mentions and emoticons.

    Parameters
    ----------
    tweet : str
        Tweet to clean up.

    Returns
    -------
    str
        Tweet cleaned up.
    """
    regex_pattern = re.compile(pattern="["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictograms
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    pattern = re.compile(r'(https?://)?(www\.)?(\w+\.)?(\w+)(\.\w+)(/.+)?')
    tweets = re.sub(regex_pattern, '', tweets)  # replaces the pattern with ''
    tweets = re.sub(pattern, '', tweets)
    tweets = re.sub(r'@[^\s]+', '', tweets)
    tweets = re.sub(r'#[^\s]+', '', tweets)
    # Removes special characters and links
    #tweets = re.sub(r'[^\w\s]', '', tweets)
    tweets = re.sub(r'https?://[A-Za-z0-9./]+', '', tweets)
    # Removes user mentions
    tweets = re.sub(r'@[A-Za-z0-9]+', '', tweets)

    token = WordPunctTokenizer()
    words = token.tokenize(tweets)
    result_words = [x for x in words]
    return " ".join(result_words)

def remove_emojis(tweets):
    text = cleaning_tweets(tweets)
    # Create an empty list to store the cleaned text
    cleaned_text = []

    # Scroll through each character of the text
    for character in text:
        # Use the `category` function of the `unicodedata` library to get the Unicode category of the character
        character_category = unicodedata.category(character)
        # If the character's Unicode category is not "So" (Symbol, Other), add the character to the clean text list
        if character_category != "So":
            cleaned_text.append(character)

    # Join the characters in the cleaned text list into a string and return it
    return "".join(cleaned_text)

def process_tweets(tweets):
    """
    Processes tweets by removing empty words and lemmatizing the remaining words.

    Parameters
    ----------
    tweets : list of str
        List of tweets to process.

    Returns
    -------
    list of str
        List of processed tweets.
    """
    with open("french_stopwords.txt", "r", encoding="utf-8") as stopwords_file:
        stopwords = []
        for line in stopwords_file:
            word = line.split("|")[0].strip()
            stopwords.append(word)

    cleaned_tweets, lemmatized = [], []
    for tweet in tweets.lower().split():
        if (tweet not in stopwords) and (len(tweet) > 1):
            cleaned_tweets.append(remove_emojis(tweet))
    
    nlp = spacy.load('fr_core_news_md')
    processed_tweets = []
    for tweet in cleaned_tweets:
        doc = nlp(tweet)
        for x in doc:
            if (x.lemma_ not in stopwords) and (len(x.lemma_) > 1):
                lemmatized.append(x.lemma_)
            processed_tweets.append(' '.join(lemmatized))
            lemmatized = []
                
    return processed_tweets

def DTM(tweets):
    """
    Creates a word cloud from the specified list of tweets.

    Parameters
    ----------
    tweets : list of str
        List of tweets to use to create the word cloud.

    Returns
    -------
    None
    """
    docs = process_tweets(tweets)
    vectorizer = TfidfVectorizer(analyzer='word', use_idf=True, smooth_idf=True)
    X = vectorizer.fit_transform(docs)
    feature_names = sorted(vectorizer.get_feature_names_out())
    docList=['Doc'+' '+str(i) for i in range(len(docs))]
    skDocsTfIdfdf = pd.DataFrame(X.todense(), index=sorted(docList), columns=feature_names)
    return skDocsTfIdfdf

def create_wordcloud(DTM):
    tf_idf_counter = DTM.T.sum(axis=1)
    stylecloud.gen_stylecloud(text=tf_idf_counter.to_dict(),
                              icon_name='fab fa-twitter',
                              #palette='scientific.sequential.Oslo_5',
                              colors=['#010101',
                               #'#080F17',
                               #'#0D1A28',
                               #'#D5D8DC',             
                               '#FFFFFF'],
                              background_color='#1d9bf0',
                              #random_state=0,
                              #max_font_size = 200,
                              stopwords = True)
     

def upload_image(image_path, api):
    # Open the image and upload it to Twitter
    if os.path.exists(image_path):
        media_id = api.media_upload(image_path).media_id
        return media_id
    else:
        print("Error uploading image")
        return None

def read_file(nom_fichier):
  # Opens the file in read mode
  with open(nom_fichier, 'r', encoding="utf-8") as f:
    # Read all the lines in the file
    lignes = f.readlines()
    # Removes line breaks from each line
    lignes = [ligne.lower().strip() for ligne in lignes]
    # Returns the list of lines
    return lignes

def main():
    """
    Main function of the script.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-interval', type=int, default=15, help='Interval in seconds between checking for mentions')
    parser.add_argument('--words', type=str, nargs='+', default=['salut', 'word2'], help='List of words to look for in mentions')
    parser.add_argument('--response-message', type=str, default='Hello!', help='Custom response message')
    parser.add_argument('--response-prefix', type=str, default='', help='Prefix to add to the response message')
    parser.add_argument('--mention-min-length', type=int, default=10, help='Minimum length of the mention in characters')
    parser.add_argument('--mention-max-length', type=int, default=280, help='Maximum length of the mention in characters')
    parser.add_argument('--forbidden-words', type=str, nargs='+', default=read_file('forbidden-words.txt'), help='List of forbidden words')
    parser.add_argument('--config-file', type=str, default='bot.cfg', help='Configuration file path')
    parser.add_argument('--max-replies-per-day', type=int, default=50, help='Maximum number of replies to send per day')
    parser.add_argument('--only-followers', action='store_true', help='Only reply to mentions from followers')
    args = parser.parse_args()

    # Read configuration from file
    config = configparser.ConfigParser()
    config.read(args.config_file)

    api = authentification.credentials()
    
    # Initialize reply counter
    replies_sent = 0

    # Get current date
    today = date.today()
    
    # Some important variables which will be used later
    bot_id = int(api.get_user(screen_name='TestTweepy5').id_str)
    mention_id = 1

    conn = sqlite3.connect('mentions.db')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS mentions (id TEXT PRIMARY KEY, screen_name TEXT, mention TEXT, time TEXT, location TEXT)")

    # The actual bot
    while True:
        try:
            mentions = api.mentions_timeline(since_id=mention_id) # Finding mention tweets
        except tweepy.error.TweepError as e:
            print(f"Erreur lors de la récupération des mentions : {e}")
            time.sleep(60) # Wait one minute before trying again
            continue
        # Iterating through each mention tweet
        for mention in mentions:
            print("Mention tweet found")
            print(f"{mention.author.screen_name} - {mention.text}")
            mention_id = mention.id
            # Check if the number of replies sent today has reached the maximum
            if replies_sent >= args.max_replies_per_day:
                break
            # Check if the mention is from today
            if mention.created_at.date() == today:
                replies_sent += 1
            # Check if the mention is from a suspended
            try:
                user = api.get_user(user_id=mention.author.id)
            except tweepy.error.TweepError as e:
                # The user is suspended
                print(f"The user is suspended or there is an error of type {e}.")
                break
            # Check if the mention is from a protected account
            if user.protected:
                continue
            # Check if the mention is from a follower, if the --only-followers option is set
            if args.only_followers:
                friendship = api.get_friendship(source_id=mention.author.id, target_id=bot_id)
                if not friendship[0].following:
                    continue
            # Check if the mention has already been processed
            mention_time = mention.created_at
            mention_location = mention.user.location
            mention_id = mention.id_str
            cursor.execute("SELECT * FROM mentions WHERE id=? AND screen_name=? AND mention=? AND time=? AND location=?", (mention.id_str, mention.author.screen_name, mention.text, mention.created_at, mention.user.location))
            if cursor.fetchone():
                continue
            # Check if the mention contains any of the words in the list
            if not any(word.lower() in mention.text for word in args.words):
                continue
            # Check if the mention is within the specified length range
            if len(mention.text) < args.mention_min_length or len(mention.text) > args.mention_max_length:
                continue
            # Check if the mention contains any forbidden words
            if any(word.lower() in mention.text for word in args.forbidden_words):
                continue
            # Checking if the mention tweet is not a reply, we are not the author, and
            # that the mention tweet contains one of the words in our 'words' list
            # so that we can determine if the tweet might be a question.
            if mention.in_reply_to_status_id is None and mention.author.id != bot_id and any(word in mention.text for word in args.words):
                try:
                    print("Attempting to reply...")
                    screen_name = mention.author.screen_name
                    try:
                        tweets_by_date = get_tweets_by_date(screen_name, api)
                    except Exception as e:
                        print(f"Error when retrieving tweets : {e}")
                    
                    tweets = ' '.join([tweet.full_text for tweet in tweets_by_date])
                    skDocsTfIdfdf = DTM(tweets)
                    create_wordcloud(skDocsTfIdfdf)
                        
                    if not os.path.exists('stylecloud.png'):
                        print("Error: generated image not found")
                        continue
                    media_id = upload_image('stylecloud.png', api)
                    message = "Hello!"
                    api.update_status(status=message, media_ids=[media_id], in_reply_to_status_id=mention.id_str)
                    print("Successfully replied :)")
                except Exception as exc:
                    print(f"Error: {exc}")
                    
                # Mark the mention as processed
                cursor.execute("INSERT INTO mentions (id, screen_name, mention, time, location) VALUES (?, ?, ?, ?, ?)", (mention.id, screen_name, mention.text, mention_time, mention_location))
                conn.commit()
                    
        # Reset reply counter if it's a new day
        if datetime.now().date() > today:
            replies_sent = 0
            today = datetime.now().date()
            
        if os.path.exists('stylecloud.png'):
            os.remove('stylecloud.png')
               
        time.sleep(args.check_interval) # The bot will check for mentions at the specified interval

if __name__ == "__main__":
    main()
