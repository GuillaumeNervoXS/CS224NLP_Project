#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 09:54:51 2020

@author: pabloveyrat
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
data_path_train='./data/train.npz'
data_path_dev='./data/dev.npz'
word2idx_file='./data/word2idx.json'
dataset_dev = np.load(data_path_dev)
dataset_train = np.load(data_path_train)
#questions=dataset['ques_idxs']

what=[79,191,57953]
who=[66874,8372,999,75]
how=[8801,105,255,63333]
when=[64639,256,83]
where=[74129,1239,162]
which=[68,2461,75937]
why=[297,635,63867]
was=[2436,32]
index_questions=[what,who,how,when,where,which,why,was]

#file=open(word2idx_file,'r')

def histogram(index_questions,file):
    dataset=np.load(file)
    questions=dataset['ques_idxs']
    counts=np.zeros(len(index_questions)+1)
    others=[]
    for j in questions:
       i=0
       achieved=False
       while i<len(index_questions) and achieved==False:
           k=0
           not_zero=False
           while k<len(j) and achieved==False and not_zero==False:
               if j[k] in index_questions[i]:
                   counts[i]+=1
                   achieved=True
               if j[k]==0:
                   not_zero=True
               k+=1
           i+=1
       if i==len(index_questions):
           counts[i]+=1
           others+=[j[0]]
    return counts/len(questions),Counter(others)

def histogram_2(index_questions,file):
    dataset=np.load(file)
    questions=dataset['ques_idxs']
    counts=np.zeros(len(index_questions)+1)
    others=[]
    for j in questions:
       achieved=False
       k=0
       not_zero=False
       while k<len(j) and achieved==False and not_zero==False:
           if j[k]==0:
                   not_zero=True
           i=0
           while i<len(index_questions) and achieved==False and not_zero==False:
               if j[k] in index_questions[i]:
                   counts[i]+=1
                   achieved=True
               i+=1
           k+=1
       if not_zero:
           counts[len(index_questions)]+=1
           others+=[j[0]]
    return counts/len(questions),Counter(others)
           
#counts_dev,others_dev=histogram(index_questions,data_path_dev)
#counts_train,others_train=histogram(index_questions,data_path_train)
    
counts_dev,others_dev=histogram_2(index_questions,data_path_dev)
counts_train,others_train=histogram_2(index_questions,data_path_train)

def transform_error(count):
    count_t=np.zeros(8)
    for i in range(7):
        count_t[i]=count[i]
    count_t[7]=count[8]
    return count_t

def plot(counts_train,counts_dev):
    X=["What","Who","How","When","Where","Which","Why",'Was','Other']
    """
    plt.bar(X,counts_train,label="Train Set")
    plt.bar(X,counts_dev,label="Dev Set")
    plt.title("Frequency of the different types of questions")
    plt.xlabel("Question Categories")
    plt.ylabel("% instances")
    plt.legend()
    plt.grid(axis='y')
    plt.show()
    """
    #counts_train=transform_error(counts_train)
    #counts_dev=transform_error(counts_dev)
    fig = plt.figure()  
    ax = fig.add_subplot(111)
    
    ## the data
    N = len(counts_train)

    ## necessary variables
    ind = np.arange(N)                # the x locations for the groups
    width = 0.35                      # the width of the bars
    ## the bars
    rects1 = ax.bar(ind, counts_train, width,
                    color='grey')
    
    rects2 = ax.bar(ind+width, counts_dev, width,
                        color='orange')
    
    # axes and labels
    ax.set_xlim(-width,len(ind)+width)
    ax.set_ylim(0,0.65)
    ax.set_ylabel('Frequency')
    ax.set_title("Frequency of the different types of questions")
    xTickMarks = X
    ax.set_xticks(ind+width)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=10)
    
    ## add a legend
    ax.legend( (rects1[0], rects2[0]), ('Train Set', 'Dev Set') )
    
    plt.show()
    return

def length(dataset,typ):
    interest=dataset[typ]
    first_index=np.zeros(len(interest))
    for i in range(len(interest)):
        j=0
        already_zero=False
        while j<len(interest[i]) and already_zero==False:
            if interest[i][j]==0:
                first_index[i]=j
                already_zero=True
            j=j+1
    return first_index

def countNA(dataset):
    interest=dataset['y1s']
    count=0
    for i in interest:
        if i==-1:
            count+=1
    return count/len(interest)

def bin_answerable_start(dataset):
    interest=dataset['y1s']
    binned=np.zeros(14)
    for i in interest:
        if i<20 and i>=0:
            binned[0]+=1
        elif i>=20 and i<40:
            binned[1]+=1
        elif i>=40 and i<60:
            binned[2]+=1
        elif i>=60 and i<80:
            binned[3]+=1            
        elif i>=80 and i<100:
            binned[4]+=1            
        elif i>=100 and i<120:
            binned[5]+=1            
        elif i>=120 and i<140:
            binned[6]+=1
        elif i>=140 and i<160:
            binned[7]+=1
        elif i>=160 and i<180:
            binned[8]+=1
        elif i>=180 and i<200:
            binned[9]+=1
        elif i>=200 and i<220:
            binned[10]+=1
        elif i>=220 and i<240:
            binned[11]+=1  
        elif i>=240 and i<260:
            binned[12]+=1
        elif i>=260:
            binned[13]+=1
    binned=binned/len(dataset['y1s']) 
    return binned/(binned.sum())

def bin_answerable_length(y):
    binned=np.zeros(12)
    for i in y:
        if i<2 and i>=0:
            binned[0]+=1
        elif i>=2 and i<4:
            binned[1]+=1
        elif i>=4 and i<6:
            binned[2]+=1
        elif i>=6 and i<8:
            binned[3]+=1            
        elif i>=8 and i<10:
            binned[4]+=1            
        elif i>=10 and i<12:
            binned[5]+=1            
        elif i>=12 and i<14:
            binned[6]+=1
        elif i>=14 and i<16:
            binned[7]+=1
        elif i>=16 and i<18:
            binned[8]+=1
        elif i>=18 and i<20:
            binned[9]+=1
        elif i>=20 and i<22:
            binned[10]+=1
        elif i>=22:
            binned[11]+=1  
    return binned/(binned.sum())

def get_answerable_length(dataset):
    interest=dataset['y1s']
    interest2=dataset['y2s']
    liste=np.array([])

    for i in range(len(interest)):
        if interest[i]!=-1 and interest2[i]!=-1:
            liste=np.append(liste,interest2[i]-interest[i])
    return liste


def binn(liste):
    binned=np.zeros(8)
    for i in liste:
        if i<50:
            binned[0]+=1
        elif i>=50 and i<100:
            binned[1]+=1
        elif i>=100 and i<150:
            binned[2]+=1
        elif i>=150 and i<200:
            binned[3]+=1            
        elif i>=200 and i<250:
            binned[4]+=1            
        elif i>=250 and i<300:
            binned[5]+=1            
        elif i>=300 and i<350:
            binned[6]+=1            
        elif i>=350:
            binned[7]+=1            
    return binned/len(liste)

def bin_question(liste):
    binned=np.zeros(6)
    for i in liste:
        if i<5:
            binned[0]+=1
        elif i>=5 and i<10:
            binned[1]+=1
        elif i>=10 and i<15:
            binned[2]+=1
        elif i>=15 and i<20:
            binned[3]+=1            
        elif i>=20 and i<25:
            binned[4]+=1            
        elif i>=25 and i<30:
            binned[5]+=1                      
    return binned/len(liste)
          
   
#context_dev=length(dataset_dev,'context_idxs')
#context_dev=binn(context_dev)
#answer_dev=length(dataset_dev,'ques_idxs')
    
#context_train=length(dataset_train,'context_idxs')
#context_train=binn(context_train)
#answer_train=length(dataset_train,'ques_idxs')

#y_dev=bin_answerable_start(dataset_dev)
#y_train=bin_answerable_start(dataset_train)
            
#length_dev=get_answerable_length(dataset_dev)
#length_dev=bin_answerable_length(length_dev)
#length_train=get_answerable_length(dataset_train) 
#length_train=bin_answerable_length(length_train)           
            
def plot_tokens_context(counts_train,counts_dev):
    X=["0-50","50-100","100-150","150-200","200-250","250-300","300-350","350-400"]
    fig = plt.figure()  
    ax = fig.add_subplot(111)
    
    ## the data
    N = len(counts_train)

    ## necessary variables
    ind = np.arange(N)                # the x locations for the groups
    width = 0.35                      # the width of the bars
    ## the bars
    rects1 = ax.bar(ind, counts_train, width,
                    color='grey')
    
    rects2 = ax.bar(ind+width, counts_dev, width,
                        color='orange')
    
    # axes and labels
    ax.set_xlim(-width,len(ind)+width)
    ax.set_ylim(0,0.5)
    ax.set_ylabel('Frequency')
    ax.set_title("Context Lengths")
    xTickMarks = X
    ax.set_xticks(ind+width)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=10)
    
    ## add a legend
    ax.legend( (rects1[0], rects2[0]), ('Train Set', 'Dev Set') )
    
    plt.show()
    return

def plot_tokens_questions(counts_train,counts_dev):
    X=["0-5","5-10","10-15","15-20","20-25","25-30"]
    fig = plt.figure()  
    ax = fig.add_subplot(111)
    
    ## the data
    N = len(counts_train)

    ## necessary variables
    ind = np.arange(N)                # the x locations for the groups
    width = 0.35                      # the width of the bars
    ## the bars
    rects1 = ax.bar(ind, counts_train, width,
                    color='grey')
    
    rects2 = ax.bar(ind+width, counts_dev, width,
                        color='orange')
    
    # axes and labels
    ax.set_xlim(-width,len(ind)+width)
    ax.set_ylim(0,0.5)
    ax.set_ylabel('Frequency')
    ax.set_title("Question Lengths")
    xTickMarks = X
    ax.set_xticks(ind+width)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=10)
    
    ## add a legend
    ax.legend( (rects1[0], rects2[0]), ('Train Set', 'Dev Set') )
    
    plt.show()
    return

def plot_tokens_start_index(counts_train,counts_dev):
    X=["0-20","20-40","40-60","60-80","80-100","100-120","120-140","140-160","160-180","180-200","200-220","220-240","240-260",">260"]
    fig = plt.figure()  
    ax = fig.add_subplot(111)
    
    ## the data
    N = len(counts_train)

    ## necessary variables
    ind = np.arange(N)                # the x locations for the groups
    width = 0.35                      # the width of the bars
    ## the bars
    rects1 = ax.bar(ind, counts_train, width,
                    color='grey')
    
    rects2 = ax.bar(ind+width, counts_dev, width,
                        color='orange')
    
    # axes and labels
    ax.set_xlim(-width,len(ind)+width)
    ax.set_ylim(0,0.27)
    ax.set_ylabel('Frequency')
    ax.set_title("Answer Start Index for Answerable Questions")
    xTickMarks = X
    ax.set_xticks(ind+width)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=10)
    
    ## add a legend
    ax.legend( (rects1[0], rects2[0]), ('Train Set', 'Dev Set') )
    
    plt.show()
    return

def plot_tokens_length_index(counts_train,counts_dev):
    X=["0-2","2-4","4-6","6-8","8-10","10-12","12-14","14-16","16-18","18-20","20-22",">22"]
    fig = plt.figure()  
    ax = fig.add_subplot(111)
    
    ## the data
    N = len(counts_train)

    ## necessary variables
    ind = np.arange(N)                # the x locations for the groups
    width = 0.35                      # the width of the bars
    ## the bars
    rects1 = ax.bar(ind, counts_train, width,
                    color='grey')
    
    rects2 = ax.bar(ind+width, counts_dev, width,
                        color='orange')
    
    # axes and labels
    ax.set_xlim(-width,len(ind)+width)
    ax.set_ylim(0,0.6)
    ax.set_ylabel('Frequency')
    ax.set_title("Answer Length for Answerable Questions")
    xTickMarks = X
    ax.set_xticks(ind+width)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=10)
    
    ## add a legend
    ax.legend( (rects1[0], rects2[0]), ('Train Set', 'Dev Set') )
    
    plt.show()
    return






