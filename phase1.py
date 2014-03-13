# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:45:31 2014

@Assignment 1 - IR course
@author: Joshua
"""
import nltk,os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sys import maxint

inverted_table = {}
max_posting = -maxint-1
min_posting = maxint
document_size = 1400
sum_posting = 0
total_words = []

def seperateFile():
    count = 1
    f = open('cran.txt','r')
    for line in f.readlines():      
        if line.startswith('.I'):
            wf = open('data/'+str(count),'w')
            count += 1
        wf.write(line)
    
def generateInvert():
    #We use two regular expression tokenizer, since text and author/affilication are different
    #tokenizer1 is for text, it allows= just word.
    #tokenizer2 is for author/affilication, it allows something like, clarkson,b.l.
    #however, in cran722 clarkson,b.l., but in 640, b. l. clarkson, I didn't normalize them
    #the author and affiliation information are pretty messy
    tokenizer1 = RegexpTokenizer(r'\w+') 
    tokenizer2 = RegexpTokenizer(r'(\w+[-*:_,. ]{0,1})+')
    porter = nltk.PorterStemmer() #poter stemmer
    stop = stopwords.words('english')
    global inverted_table, max_posting, min_posting, sum_posting, total_words
    g = open('doc_length.txt','w')    
    for fs in os.listdir('data'):
        print fs
        s1,s2 = '',''
        f = open('data/'+fs,'r')
        line = ''
        while 1:
            line = f.readline().strip()
            if line == '':
                break                     
            elif line.startswith(".I"):
                continue
            elif line.startswith(".T"):
                s1 += f.readline()
            elif line.startswith(".A") or line.startswith(".B"):
                s2 += f.readline()
            else:
                s1 += line
            
        res1 = [t for t in tokenizer1.tokenize(s1)] #tokenization and stemming using nltk 
        res2 = [t for t in tokenizer2.tokenize(s2)] #using two different tokenizer, one for text, one for author and affiliation
        total_words.extend(res1+res2)
        stem1 = [porter.stem(t) for t in res1]  #res2 doesn't need stemming, it contains only author and affiliation        
        res = [i for i in res2+stem1 if i not in stop] #stopword     
        
        #pre_computing length of docs           
        g.write(fs+":"+str(len(res)) + "\n")
        
        #compute the term frequency in one doc
        termFrequency = {}
        for term in res:
            if term not in termFrequency:
                termFrequency[term] = 1
            else:  # term already there, just update frequency
                termFrequency[term] = termFrequency[term] + 1
        
        #put all information into inverted file, like,
        #term1 #(term1 in corp) [doc1:freq(doc1,term1), etc..] 
        for term in termFrequency:
            if term not in inverted_table:
                #posting and tf value will be stored
                inverted_table[term] = [str(fs+":"+str(termFrequency[term]))]
            else:
                cur = inverted_table[term]
                cur.append(str(fs+":"+str(termFrequency[term])))
                inverted_table[term] = cur
            
        f.close()
        
    f = open("inverted_table.txt",'w')
    sort_items = inverted_table.items();
    sort_items.sort();

    for (key,item) in sort_items:
        max_posting = max(len(item),max_posting)
        min_posting = min(len(item),min_posting)
        sum_posting +=len(item)
        f.write(key+"\t"+str(len(item))+"\t"+str(item)+"\n")
    f.close()
    g.close()
    statistics()
    
def statistics():
    f = open("statistics.txt",'w')
    f.write("There are total " + str(document_size) + " documents\n")
    f.write("There are total " + str(len(total_words)) + " words without stemming or tokenization\n")
    f.write("There are total " + str(len(inverted_table))+" words after stemming and tokenization, i.e. unique words\n")
    f.write("The longest & shortest posting length are " + str(max_posting) + " / " + str(min_posting) +"\n")
    f.write("The total posting length is " + str(sum_posting) + "\n")
    f.write("The average posting length is " + str(sum_posting/len(inverted_table)) + "\n")
    f.close()
    
if __name__ == "__main__":
    seperateFile()
    generateInvert()
    