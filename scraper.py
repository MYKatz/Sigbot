import config

import praw
import pandas as pd

reddit = praw.Reddit(client_id=config.PRAW_CONFIG["appID"],
                     client_secret=config.PRAW_CONFIG["secret"],
                     user_agent=config.PRAW_CONFIG["useragent"],
                     username=config.PRAW_CONFIG["username"],
                     password=config.PRAW_CONFIG["password"])

ja = reddit.subreddit("jakeandamirscripts")

posts = ja.top(limit=1000)
f = open("data/scripts.txt", "w+", encoding="utf-8")
          
for post in posts:
    if post.is_self and "Jake and Amir:" in post.title:
        f.write(post.selftext)

