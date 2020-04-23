import os
import sys
import numpy as np
import praw

script_id = os.environ.get('PERSONAL_USE_SCRIPT_14_CHARS')
secret_key = os.environ.get('SECRET_KEY_27_CHARS')
app_name = os.environ.get('APP_NAME')
reddit_username = os.environ.get('REDDIT_USER_NAME')
reddit_pass = os.environ.get('REDDIT_LOGIN_PASSWORD')

reddit = praw.Reddit(client_id=script_id,
                     client_secret=secret_key,
                     user_agent=app_name,
                     username=reddit_username,
                     password=reddit_pass)
subreddit = reddit.subreddit("showerthoughts")

top_submissions = subreddit.top(limit=100)

n_sub = int(sys.argv[1]) if sys.argv[1] else 1

i = 0
while i < n_sub:
    top_submission = next(top_submissions)
    i += 1

top_post = top_submission.title

upvotes = []
downvotes = []
contents = []

for sub in top_submissions:
    try:
        ratio = sub.upvote_ratio
        ups = int(round((ratio * sub.score) / (2 * ratio - 1)) if ratio != 0.5 else round(sub.score / 2))
        upvotes.append(ups)
        downvotes.append(ups - sub.score)
        contents.append(sub.title)
    except Exception as e:
        continue
votes = np.array([upvotes, downvotes]).T
