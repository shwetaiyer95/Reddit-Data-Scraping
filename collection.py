# NAME(s): Adhiraj Sood, Shweta Nateshan Iyer
# Homework 2 for csci 720 - Big data analytics
# Lines 3--30 are from 
# https://towardsdatascience.com/how-to-use-the-reddit-api-in-python-5e05ddfd1e5c
import nltk
import requests
import json
from matplotlib import pyplot as plt
import cv2
from nltk import *
from nltk.corpus import stopwords

nltk.download('stopwords')
# appended some custom stopwords
stopwords = stopwords.words("english")
stopwords.append('_')
nltk.download('wordnet')
tknzr = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

# note that CLIENT_ID refers to 'personal use script' and SECRET_TOKEN to 'token'
# IMPORTANT: please remove your actual ID and TOKEN before pushing up to Github!!!
auth = requests.auth.HTTPBasicAuth('<CLIENT_ID>', '<SECRET_TOKEN>')

# here we pass our login method (password), username, and password
# IMPORTANT: please remove your username and password before pushing up to Github!!!
data = {'grant_type': 'password',
        'username': '<USER>',
        'password': '<PASSWORD>'}

# setup our header info, which gives reddit a brief description of our app
headers = {'User-Agent': 'MyBot/0.0.1'}

# send our request for an OAuth token
res = requests.post('https://www.reddit.com/api/v1/access_token',
                    auth=auth, data=data, headers=headers)

# convert response to JSON and pull access_token value
TOKEN = res.json()['access_token']

# add authorization to our headers dictionary
headers = {**headers, **{'Authorization': f"bearer {TOKEN}"}, **{'Cache-Control': 'no-cache'}}

# while the token is valid (~2 hours) we just add headers=headers to our requests
requests.get('https://oauth.reddit.com/api/v1/me', headers=headers)
# Created a list of all subreddits
list_of_subreddit = ["https://oauth.reddit.com/r/Rochester/", "https://oauth.reddit.com/r/DMV/",
                     "https://oauth.reddit.com/r/bookclub/", "https://oauth.reddit.com/r/plotholes/",
                     "https://oauth.reddit.com/r/NarutoFanfiction/", "https://oauth.reddit.com/r/Ghoststories/",
                     "https://oauth.reddit.com/r/InformationTechnology/"]

finalResponse = []
for subreddit in list_of_subreddit:
    page = requests.get(subreddit,
                        headers=headers, params={'limit': '500'})
    finalResponse = finalResponse + page.json()['data']['children']

"""
1) Created a reddit developers account and a script application
2) Used the api to download at least 100 text posts from the subreddit
        place these in a file called "posts.json" in the root level of your repo.
"""
with open("posts.json", 'w') as file:
    """
    Dumped all the posts in the subreddit.
    These are the posts which are found inside data.children
    """
    json.dump(finalResponse, file)

"""
3) Created a new json file called "texts.json" and placed the 
main text of the subreddit in this file. 
Some preprocessing, i.e., remove stopwords, and possible stem and lemmatize has been performed before writing to file.
"""
# We loop around the list of posts we found and then process the selftexts by first tokenizing,
# then lemmatize, then stemming and in the end removing the stopwords. These texts are then stored in texts.json

final = [] # the final list of words which are used for counting the frequency
finalTexts = [] # The selftexts we get in every subreddit
for i in finalResponse:
    text = i['data']['selftext']
    if len(text.strip()) > 0:
        finaltext = []
        words = tknzr.tokenize(text)
        for word in words:
            lem = lemmatizer.lemmatize(word)
            stem = ps.stem(lem)
            if stem not in stopwords:
                finaltext.append(stem)
        finalTexts.append(' '.join([word for word in finaltext]))
        final.append(finaltext)
with open("texts.json", 'w') as file:
    json.dump(finalTexts, file)

"""
4) Print out the 20 words with the highest document frequency (df) values, ranked in
        descending order. A copy of this list is in the README.md file
"""

def calculate_frequency(list):
    """
    Method to calculate the frequency of the words using the "final" list created in 5th step
    :param list: The list of list which contains the words
    :return: The dictionary with the word and it's count
    """
    worddict = dict()
    for firstlist in list:
        for word in firstlist:
            if word in worddict:
                worddict[word] += 1
            else:
                worddict[word] = 1
    return worddict

# Find the frequency of the word
frequencyList = calculate_frequency(final)
# The descending sorted list sorted on the value in dictionary
descendingSorted = dict(sorted(frequencyList.items(), key=lambda item: item[1], reverse=True))
# Printing the 20 words
print(dict(list(descendingSorted.items())[:20]))

"""
5) Created a directory called images and placed 5 images in it.
"""
# Image 4 and 5 are from own camera
# Image 1, 2 and 3 are from the internet

"""
6) Converted the images to grayscale with 256 shades. Stored these images
        in the same images directory. Append the prefix of each image with _gray. 
        Thus if the original image was foo.jpg, the grayscale image should be foo_gray.jpg
"""
# Images we stored in the image folder
list_of_images = ["Image1.jpeg", "Image2.jpeg", "Image3.jpeg", "Image4.jpeg", "Image5.jpeg"]
list_of_gray_images = []
for image in list_of_images:
    image_in = cv2.imread("images/" + image)
    image_gray = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
    list_of_gray_images.append((image, image_gray))
    cv2.imwrite("images/" + image[:image.find(".")]+"_gray"+image[image.find("."):], image_gray)


"""
7) Used matlplotlib to construct histograms for each image. Saved them as .svg files, in the
same image directory and same names, but with the suffix changed to .svg. Also, the .svg's are put in the README.md. file, labeled so that it is clear which 
image created each histogram.
"""

"""
Using matplotlib plotted the histogram for colored image and the gray image
"""
for image in list_of_images:
    image_in = cv2.imread("images/" + image)
    image_gray = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
    plt.hist(image_gray.ravel(), 256, (0, 256), label=image)
    plt.legend()
    plt.title("Grayscale " + image + " Histogram")
    plt.xlabel("Bins")
    plt.ylabel("Number of Pixels")
    plt.xlim([0, 256])
    plt.savefig("images/" + image[:image.find(".")] + "_gray.svg")
    plt.close()
    plt.title("Color " + image + " Histogram")
    plt.xlabel("Bins")
    plt.ylabel("Number of Pixels")
    for channel_index, color in enumerate(['blue', 'green', 'red']):
        plt.hist(image_in[:, :, channel_index].ravel(), 256, (0, 256), label=image, color=color)
        plt.xlim([0, 256])
    plt.savefig("images/" + image[:image.find(".")] + ".svg")
    plt.close()
