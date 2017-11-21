import os

from flask import Flask, render_template, jsonify, request, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, Date, Float, ForeignKey, func
from sqlalchemy.orm import relationship, backref
from sqlalchemy.types import PickleType

import json
import time
import io
import sqlite3
import numpy as np
import shutil
import sys
import glob
import pandas as pd
import requests
from scipy import spatial

auth = ('mikephul', 'password')

def get_follower(followers_url):
    page_num = 0
    follower = []
    for page_num in range(1,10):
        followers = requests.get(followers_url + '?page=' + str(page_num)+ '&per_page=100', auth=auth, verify=False).json()
        if len(followers) == 0:
            break
        for f in followers:
            follower.append(f['login'])
    return follower

def get_following(following_url):
    page_num = 0
    following = []
    for page_num in range(1,10):
        followings = requests.get(following_url + '?page=' + str(page_num)+ '&per_page=100', auth=auth, verify=False).json()
        if len(followings) == 0:
            break
        for f in followings:
            following.append(f['login'])
    return following

def get_following(following_url):
    page_num = 0
    following = []
    for page_num in range(1,10):
        followings = requests.get(following_url + '?page=' + str(page_num)+ '&per_page=100', auth=auth, verify=False).json()
        if len(followings) == 0:
            break
        for f in followings:
            following.append(f['login'])
    return following

def get_repo_info(repos_url, username):
    languages = {}
    owned_repos = []
    forked_repos = []
    num_stars = 0
    num_watches = 0
    num_commits = 0
    num_pulls = 0
    page_num = 0
    for page_num in range(1,10):
        repos = requests.get(repos_url + '?page=' + str(page_num)+ '&per_page=100', auth=auth, verify=False).json()
        if len(repos) == 0:
            break
        for repo in repos:
            name = repo['name']
            if not repo['fork']:
                owned_repos.append(name)
                num_stars += repo['stargazers_count']
                num_watches += repo['watchers_count']

                commit = requests.get('https://api.github.com/repos/' + username + '/' + name + '/stats/participation', auth=auth, verify=False).json()
                try:
                    num_commits += np.sum(commit['owner'])
                except:
                    pass
                pulls = requests.get('https://api.github.com/repos/' + username + '/' + name + '/pulls', auth=auth, verify=False).json()
                num_pulls += len(pulls)

            else:
                forked_repos.append(name)
            l = repo['language']
            if l in languages:
                languages[l] += 1
            else:
                languages[l] = 1
    return owned_repos, forked_repos, languages, num_stars, num_watches, num_commits, num_pulls



def pop_dict(d):
    d.pop('avatar_url', None)
    d.pop('bio', None)
    d.pop('email', None)
    d.pop('forked_repos', None)
    d.pop('hireable', None)
    d.pop('html_url', None)
    d.pop('owned_repos', None)
    d.pop('company', None)
    d.pop('location', None)
    d.pop('name', None)
    d.pop('update_at', None)

def KNN(login_hello, num_near):
	df = pd.DataFrame()
	for file in glob.glob("./data/*.json"):
	    with open(file, 'r') as f:
	        d = json.load(f)
	        pop_dict(d)
	        df = df.append(pd.DataFrame.from_dict(d).reset_index(), ignore_index=True)
	if len(df[df['login'] == login_hello]) == 0:
		return []
	language_map = df.groupby('login')['index'].value_counts().unstack()

	for row in df.iterrows():
	    lang = row[1]['index']
	    login = row[1].login.lower()
	    language_map.loc[language_map.index == login, lang] = row[1].languages 
	    
	language_map = language_map.fillna(0)
	language_map = language_map.reset_index()
	del language_map['null']
	del df['index']
	del df['languages']
	df = df.drop_duplicates()
	X = language_map.set_index('login').as_matrix()
	language_map = language_map.set_index('login')

	kdt = spatial.KDTree(X)

	vec = language_map[language_map.index == login_hello].as_matrix()
	distance, nearest = kdt.query(vec, k=num_near)
	language_map = language_map.reset_index()
	arr = []
	for i in nearest[0][1:]:
	    arr.append(language_map.loc[int(i)].login)
	return arr

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join('/', 'data')

app = Flask(__name__)

# ==================== CONFIG ====================
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'

# ==================== Database Class ====================
db = SQLAlchemy(app)

@app.route('/create/<login_hello>')
def create_user_info(login_hello):
	user_url = 'https://api.github.com/users/' + login_hello.lower()
	response = requests.get(user_url, verify=False).json()

	# Get User Basic data
	name = response['name']
	avatar_url = response['avatar_url']
	login = response['login']
	bio = response['bio']
	company = response['company']
	email = response['email']
	hireable = response['hireable']
	html_url = response['html_url']
	location = response['location']

	num_followers = response['followers']
	num_following = response['following']
	num_public_gists = response['public_gists']
	num_public_repos = response['public_repos']

	update_at = response['updated_at']

	follower = get_follower(response['followers_url'])
	following = get_following(response['following_url'][:-13])
	owned_repos, forked_repos, languages, num_stars, num_watches, num_commits, num_pulls = get_repo_info(response['repos_url'], login)

	output = {
	    'name': name,
	    'avatar_url': avatar_url,
	    'login': login,
	    'bio': bio,
	    'company': company,
	    'email': email,
	    'hireable': hireable,
	    'html_url': html_url,
	    'location': location,
	    'num_followers': num_followers,
	    'num_following': num_following,
	    'num_public_gists': num_public_gists,
	    'num_public_repos': num_public_repos,
	    'update_at': update_at,
	    'owned_repos': owned_repos,
	    'num_owned_repos': len(owned_repos),
	    'forked_repos': forked_repos,
	    'num_forked_repos': len(forked_repos),
	    'languages': languages,
	    'num_stars': num_stars,
	    'num_watches': num_watches,
	    'num_commits': int(num_commits),
	    'num_pulls': num_pulls,
	}

	with open('./data/' + login + '.json', 'w') as fp:
	    json.dump(output, fp)

	with open('./data/' + login + '.json') as json_data:
		d = json.load(json_data)

	return jsonify(d)

@app.route('/user/<login_hello>')
def get_user_info(login_hello):
	with open('./data/' + login_hello + '.json') as json_data:
		d = json.load(json_data)		
	return jsonify(d)

@app.route('/level/<login_hello>')
def get_user_level(login_hello):
	df = pd.DataFrame()
	for file in glob.glob("./data/*.json"):
	    with open(file, 'r') as f:
	        d = json.load(f)
	        pop_dict(d)
	        df = df.append(pd.DataFrame.from_dict(d).reset_index(), ignore_index=True)
	del df['index']
	del df['languages']
	df = df.drop_duplicates()
	df['score'] = df['num_followers'] + df['num_pulls'] + df['num_stars'] + df['num_watches'] + 10*df['num_owned_repos'] + df['num_public_gists']
	
	df['level'] = np.round(df['score']/100)
	
	return jsonify(int(df[df['login'] == login_hello]['level']))

@app.route('/near/<login_hello>')
def get_near_user(login_hello):	
	l = []
	for dev in KNN(login_hello, 5):
		with open('./data/' + dev + '.json') as json_data:
			l.append(json.load(json_data))
	return jsonify(l)

if __name__=='__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
