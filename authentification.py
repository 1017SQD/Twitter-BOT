# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 00:09:13 2022

@author: 18017952
"""
import tweepy

def credentials():
    consumer_key='API / Consumer Key here'
    consumer_secret='API / Consumer Secret here'

    access_token='Access Token here'
    access_token_secret='Access Token Secret here'


    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    return tweepy.API(auth, wait_on_rate_limit=True)