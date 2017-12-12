# coding: utf-8
import re
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer

def normalize(content):
    """
    Lower-casing: The entire email is converted into lower case, so that captialization is ignored (e.g., IndIcaTE is treated the same as Indicate).
    Stripping HTML: All HTML tags are removed from the emails.Many emails often come with HTML formatting; we remove all the HTML tags, so that only the content remains.
    Normalizing URLs: All URLs are replaced with the text "httpaddr".
    Normalizing Email Addresses: All email addresses are replaced with the text "emailaddr".
    Normalizing Numbers: All numbers are replaced with the text "number".
    Normalizing Dollars: All dollar signs ($) are replaced with the text "dollar".
    Word Stemming: Words are reduced to their stemmed form. For example, "discount", "discounts", "discounted" and "discounting" are all replaced with "discount". Sometimes, the Stemmer actually strips off additional characters from the end, so "include", "includes", "included",and "including" are all replaced with "includ".
    Removal of non-words: Non-words and punctuation have been removed. All white spaces (tabs, newlines, spaces) have all been trimmed to a single space character.
    """
    content = content.lower()

    #Strip all HTML Looks for any expression that starts with < and ends with > and replace and does not have any < or > in the tag it with a space
    content,_ = re.subn( r'<[^<>]+>', ' ',content)

    #Handle Numbers Look for one or more characters between 0-9
    content,_ = re.subn( r'[0-9]+', 'number',content)

    #Handle URLS Look for strings starting with http:// or https://
    content,_ = re.subn( r'(http|https)://[^\s]*', 'httpaddr',content)

    #Handle Email Addresses Look for strings with @ in the middle
    content,_ = re.subn( r'[^\s]+@[^\s]+', 'emailaddr',content)

    #Handle $ sign
    content,_ = re.subn( r'[$]+', 'dollar',content)

    return content

def indice(content):
    """
    Look up the word in the dictionary and add to indices if found
    """
    words = re.split(r'[\[\]\'\n\s!(){},>_<;%@/#-:&=$.*+?"]',content)
    # remove blank
    words = [item for item in filter(lambda x:x != '', words)]
    stemmer = PorterStemmer()
    vocalist = pd.read_table('data/vocab.txt',header=None,index_col=0)[1]
    indices = []
    for w in words:
        # Remove any non alphanumeric characters
        w,_ = re.subn( r'[^a-zA-Z0-9]', '',w)
        # Stem the word
        w = stemmer.stem(w)
        if len(w) < 1:
            continue
        for i in range(1,len(vocalist)+1):
            if vocalist[i] == w:
                indices.append(i)
                break
    return indices

def featureMap(cotent):
    """
    Your task is take one such indices vector and construct 
    a binary feature vector that indicates whether a particular
    word occurs in the email. That is, x(i) = 1 when word i
    is present in the email. Concretely, if the word 'the' (say,
    index 60) appears in the email, then x(60) = 1. The feature
    vector should look like:
     x = [ 0 0 0 0 1 0 0 0 ... 0 0 0 0 1 ... 0 0 0 1 0 ..];
    """
    vocalist = pd.read_table('data/vocab.txt',header=None,index_col=0)[1]
    features = np.zeros(vocalist.size)
    indices = indice(normalize(email))
    # index of features starts from 0
    indices = np.array(indices) - 1
    features[ indices ] = 1
    return features

if __name__ == '__main__':
    path = './data/'
    email = open(path+'emailSample1.txt').read()
    print featureMap(email)[85]
