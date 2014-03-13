# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:09:21 2014

@author: Joshua
@project: IR course asg1 vector space search engine
"""
import nltk,os,math,re,numpy,time
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sys import maxint
from operator import itemgetter

class IRsystem:
    inverted_table = {}
    max_posting = -maxint-1
    min_posting = maxint
    document_size = 1400
    sum_posting = 0
    total_words = []   
    qlist = []
    eval_list = set()
    doc_length = range(1401)
    #if it's firstTime setup, we need to seperate the corp and generate Inverted file
    def __init__(self,firstTime=False):
        if firstTime == True:
            self.seperateFile()
            self.generateInvert()
        self.qlist = self.loadQuerys()
        self.eval_list = self.loadQueryEval()
        self.doc_length = self.loadDocLength()
    
    #accept two argument, query and how many rakned docs you want to see, default its top 20
    #return as a list
    def query(self,query, top=20, interact = True):
        
        query = str(query)        
        score = {}
        tokenizer = RegexpTokenizer(r'\w+')
        porter = nltk.PorterStemmer()        
        stop = stopwords.words('english')
        #assume the user won't query stop words
        keywords = [porter.stem(t) for t in tokenizer.tokenize(query) if porter.stem(t) not in stop]
        docs = []    
        for term in keywords:        
            inf = self.fetchTerm(term)
            if inf == "":
                continue
            else:
                t,length,posting = inf.split("\t")
                if term == t:
                    idf = math.log10(1+self.document_size/int(length))
                    for docinf in re.findall(r'\d+[:]\d+',posting):                   
                        doc,tf = docinf.split(":")
                        tf = 1+math.log10(int(tf))
                        score[doc] = score[doc]+tf*idf if doc in score else tf*idf
        #normalize                
        for key in score.keys():
            score[key] = score[key]/self.doc_length[int(key)] 
    
        #sort the score dictionary  and return the top 20 docs
        count = 1
        for doc in sorted(score.items(), key=itemgetter(1), reverse=True):
            docs.append(doc[0])
            count += 1
            if count > top :
                break

        if interact == True:
            if score == {}:
                print "There's no related document!"
            else:
                print "The total number of retrieved docs is %d" %len(score)
                print "The top up-to-"+str(top)+" lists are:"
                for d in docs[:20]:
                    print "Doc:",d, "score:", score[d]
        else:         
            return docs                  

    #fetch the term in inverted file    
    def fetchTerm(self,term):
        
        f = open("inverted_table.txt",'r')  
        for line in f.readlines():
            if line.startswith(term+"\t"):
                return line.strip()
        return ""
        
    #only print the file content
    def viewFile(self,fileNum):
        try:
            f = open('data/'+fileNum,'r')
            for line in f.readlines():
                print line.strip()
            f.close()
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
          
    #Using numpy to do vector add, and using matplotlib to plot graph
    def eval_show(self):
        i = 1
        precisions = numpy.zeros(shape=(20))
        recalls = numpy.zeros(shape=(20))
        for q in self.qlist:
            p_vec, r_vec = self.evaluation(i,q)          
            precisions = precisions + numpy.array(p_vec)
            recalls = recalls + numpy.array(r_vec)
            i=i+1
        precisions = precisions/len(self.qlist)
        recalls = recalls/len(self.qlist)
        plt.plot(recalls,precisions)
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.grid()
        plt.show()

    #takes 2 arguments, query index, and query
    #It's just because I made two hashtable for query and relevant file
    #of course, we can simplify that by merging the hashtable 
    def evaluation(self,index,q): 
        res = self.query(q,20,False)
        precisions = []
        recalls = []
        i = 1
        while i <= len(res):
            tp = len(set(res[:i]).intersection(self.eval_list[index]))
            precisions.append(tp*1.0/len(res[:i]))
            recalls.append(tp*1.0/len(self.eval_list[index]))
            i=i+1
        return precisions,recalls
        
    #load 225 query into memory
    def loadQuerys(self):
        qf = open("query.text/query.text",'r')
        qlist = []
        s = ''
        for line in qf.readlines():
            if line.startswith('.I'):           
                if s != '':
                    qlist.append(s)
                    s = ''
            elif line.startswith('.W'):
                continue
            else:
                s += line.strip()
        qlist.append(s)
        return qlist

     #load the relevant file into memory for evalution
    def loadQueryEval(self):
        evalf = open("qrels.text/qrels.text",'r')
        s = set()
        eval_list = {}
        pre = -1
        for line in evalf.readlines():
            ary = line.split(" ")
            index = int(ary[0])
            if pre != index:               
                eval_list[pre] = s
                s = set()            
                pre = index
            s.add(ary[1])
            eval_list[pre] = s
        eval_list.pop(-1)
        return eval_list
    
    def loadDocLength(self):
        doc_length = range(1401)
        g = open('doc_length.txt','r')    
        for line in g.readlines():
            index,length = line.split(":")
            doc_length[int(index)] = int(length)
        return doc_length
        
    #seperate corp into 1400 file
    def seperateFile(self):
        count = 1
        f = open('cran.txt','r')
        for line in f.readlines():      
            if line.startswith('.I'):
                wf = open('data/'+str(count),'w')
                count += 1
            wf.write(line)
        f.close()
    
    
    #generate Inverted file
    #We use two regular expression tokenizer, since text and author/affilication are different
    #tokenizer1 is for text, it allows= just word.
    #tokenizer2 is for author/affilication, it allows something like, clarkson,b.l.
    #however, in cran722 clarkson,b.l., but in 640, b. l. clarkson, I didn't normalize them
    #the author and affiliation information are pretty messy
    #the stopword list and stemmer are from nltk toolkit
    def generateInvert(self):       
        start = time.time()
        tokenizer1 = RegexpTokenizer(r'\w+') 
        tokenizer2 = RegexpTokenizer(r'(\w+[-*:_,. ]{0,1})+')
        porter = nltk.PorterStemmer() #poter stemmer
        stop = stopwords.words('english')
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
            self.total_words.extend(res1+res2)
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
                if term not in self.inverted_table:
                    #posting and tf value will be stored
                    self.inverted_table[term] = [str(fs+":"+str(termFrequency[term]))]
                else:
                    cur = self.inverted_table[term]
                    cur.append(str(fs+":"+str(termFrequency[term])))
                    self.inverted_table[term] = cur
            
            f.close()
        
        f = open("inverted_table.txt",'w')
        sort_items = self.inverted_table.items();
        sort_items.sort();

        for (key,item) in sort_items:
            self.max_posting = max(len(item),self.max_posting)
            self.min_posting = min(len(item),self.min_posting)
            self.sum_posting +=len(item)
            f.write(key+"\t"+str(len(item))+"\t"+str(item)+"\n")
        f.close()
        g.close()
        print time.time() - start
        self.statistics()
        
    
    #write some statistics into file
    def statistics(self):
        f = open("statistics.txt",'w')
        f.write("There are total " + str(self.document_size) + " documents\n")
        f.write("There are total " + str(len(self.total_words)) + " words without stemming or tokenization\n")
        f.write("There are total " + str(len(self.inverted_table))+" words after stemming and tokenization, i.e. unique words\n")
        f.write("The longest & shortest posting length are " + str(self.max_posting) + " / " + str(self.min_posting) +"\n")
        f.write("The total posting length is " + str(self.sum_posting) + "\n")
        f.write("The average posting length is " + str(self.sum_posting/len(self.inverted_table)) + "\n")
        f.close()
    
if __name__ == "__main__":
    ir = IRsystem()
    print "The IR system is already built, you can just query something"
  
    while raw_input("Want to query something? Y/N\n").lower()=='Y'.lower():
        start = time.time()
        q = raw_input("Please input your keywords:\n")
        ir.query(q)
        while raw_input("Want to see one of these docs? Y/N\n").lower()=='Y'.lower():     
            user_input = raw_input("Please input which file you want to look at:\n")
            ir.viewFile(user_input)
        print "it takes %f time" %(time.time() - start)
        
    print "Below is the precision-recall graph, using top-20 docs across the 225 query"
    print "It will take several seconds. Just hold on!"
    ir.eval_show()