#!/usr/bin/python
import wikipedia
def get_wikipedia_content(url):
    title = url[url.rindex('/') + 1:len(url)]
    return wikipedia.summary(title=title, auto_suggest=False) 