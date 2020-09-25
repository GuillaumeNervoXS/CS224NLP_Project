#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 21:19:01 2020

@author: pabloveyrat
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import json
from util import compute_f1, compute_em

data_path_train='./data/train.npz'
data_path_dev='./data/dev.npz'
word2idx_file='./data/word2idx.json'

id_file='./data/dev-v2.0.json'


result_file='./models_for_analysis/bidaf_fusion.csv'
#dataset_dev = np.load(data_path_dev)
#dataset_train = np.load(data_path_train)

def preprocess_results(result_file):
    """
    To process the results file in a dictionnary

    """
    dic={}
    ids=[]
    answers=[]
    count=0
    with open(result_file,'r') as prediction:
      for row in prediction:
         if count!=0:
            result=row.strip()
            ID=result[:25]
            answer=result[26:]
            ids+=[ID]
            answers+=[answer]
            dic[ID]=answer
         count+=1
    return dic #,ids,answers

    
def preprocess_questions(id_file):
    """
    
    dic[ID]=question corresponding to that ID

    """
    dic={}
    with open(id_file,'r') as prediction:
        file=json.load(prediction)
        for p in file['data']:
            for i in p['paragraphs']:
                for j in i['qas']:
                    question=j['question']
                    ID=j['id']
                    dic[ID]=question.lower()
    return dic
    
def preprocess_gold(id_file):
    """
    dic[ID]= either [] if no answer or list of dictionnaries with keys [text] [answer_start]
    """
    ids=[]
    answers_tab=[]
    dic={}
    with open(id_file,'r') as prediction:
        file=json.load(prediction)
        for p in file['data']:
            for i in p['paragraphs']:
                for j in i['qas']:
                    answer=j['answers']
                    ID=j['id']
                    answers_tab+=[answer]
                    ids+=[ID]
                    dic[ID]=answer
    return dic #,ids,answers_tab

def preprocess_gold_context(id_file):
    dic={}
    with open(id_file,'r') as prediction:
        file=json.load(prediction)
        for p in file['data']:
            for i in p['paragraphs']:
                context=i['context']
                for j in i['qas']:
                    ID=j['id']
                    dic[ID]={}
                    dic[ID]['context']=context
                    if j['answers']==[]:
                        dic[ID]['answer']=-1
                    else:   
                        minimum=10000000
                        for k in j['answers']:
                            if k['answer_start']<minimum:
                                minimum=k['answer_start']
                        dic[ID]['answer']=minimum
    return dic #,ids,answers_tab


def preprocess_word2idx(word2idx_file):
    """
    dic[word]=token_id
    """
    with open(word2idx_file,'r') as prediction:
        file=json.load(prediction)
    return file

def process(dic_predicted):
    """
    dic[ID]=Answer length of the result
    """
    dic={}
    for ID in dic_predicted.keys():
        if dic_predicted[ID]=='':
            dic[ID]=-1
        else:
            j=dic_predicted[ID].split(' ')
            dic[ID]=len(j)
    return dic

def process_gold_dic_small_ans(dic_expected):
    """
    dic[ID]=Minimum Answer length among the results
    """
    dic={}
    for ID in dic_expected.keys():
        if dic_expected[ID]==[]:
            dic[ID]=-1
        else:
            minimum=10000
            for j in dic_expected[ID]:
                length=len(j['text'].split(' '))
                if length<minimum:
                    minimum=length
            dic[ID]=minimum
    return dic


def compare_length(dic_length_predicted,dic_length_expected):
    """
    Clusters by expected answer length and gets the average difference per cluster
    """
    binned=np.zeros([9])
    avg_dif=np.zeros([9])
    for ID in dic_length_predicted.keys():
        dif=dic_length_predicted[ID]-dic_length_expected[ID]
        i=dic_length_expected[ID]
        if i<2 and i>=0:
            avg_dif[0]+=dif
            binned[0]+=1
        elif i>=2 and i<4:
            binned[1]+=1
            avg_dif[1]+=dif
        elif i>=4 and i<6:
            binned[2]+=1
            avg_dif[2]+=dif
        elif i>=6 and i<8:
            binned[3]+=1
            avg_dif[3]+=dif            
        elif i>=8 and i<10:
            binned[4]+=1
            avg_dif[4]+=dif        
        elif i>=10 and i<12:
            binned[5]+=1
            avg_dif[5]+=dif
        elif i>=12 and i<14:
            binned[6]+=1
            avg_dif[6]+=dif
        elif i>=14:
            binned[7]+=1
            avg_dif[7]+=dif
        elif i==-1:
            binned[8]+=1
            avg_dif[8]+=dif
    return avg_dif/binned,binned

def compare_start_index(dic_predicted,dic_expected):
    """
    Gives start index for all types of questions
    """
    start_index=np.array([])
    start_index_expected=np.array([])
    for ID in dic_predicted.keys():
        ans=dic_predicted[ID]
        if ans=='':
            start_index=np.append(start_index,-1)
        else:
            length=len(ans)
            context=dic_expected[ID]['context']
            i=0
            check=False
            while i<len(context) and check==False:
                 check=(context[i:i+length]==ans)
                 i+=1
            start_index=np.append(start_index,i-1)
        start_index_expected=np.append(start_index_expected,dic_expected[ID]['answer'])
    return start_index,start_index_expected  

def compare_start_index_answerable(dic_predicted,dic_expected):
    """
    Gives start index for answerable questions
    """   
    start_index=np.array([])
    start_index_expected=np.array([])
    for ID in dic_predicted.keys():
        index_ans_e=dic_expected[ID]['answer']
        if index_ans_e!=-1:
            ans=dic_predicted[ID]
            if ans=='':
                start_index=np.append(start_index,-1)
            else:
                length=len(ans)
                context=dic_expected[ID]['context']
                i=0
                check=False
                while i<len(context) and check==False:
                     check=(context[i:i+length]==ans)
                     i+=1
                start_index=np.append(start_index,i-1)
            start_index_expected=np.append(start_index_expected,dic_expected[ID]['answer'])
    return start_index,start_index_expected 

def get_question_type(dic_questions,ID):
    """
    Returns the type of a question with ID in dic_questions
    """
    question=dic_questions[ID]
    question=question.split(' ')
    i=0
    Found=False
    liste=["what","who","how","when","where","which","why"]
    while i<len(question) and Found==False:
        First_one=True
        j=0
        while j<len(liste) and First_one==True:
            if(liste[j]==question[i]):
                Found=True
                First_one=False
                Type=j
            j+=1
        i+=1
    if (i==len(question)):
        Type=7
    return Type
    
def plot_start_index_difference(dif_ans):
    """
    Plots the distributio of difference of the start indices
    """
    plt.hist(dif_ans,17,color='grey')
    plt.xlabel("Difference between respective answer start indices")
    plt.xlim(-1000,1000)
    plt.ylabel("Count")
    plt.title("Difference between the start index predicted by the model \n and the true start index for answerable questions")
    return
       
def plot_difference(counts_train):
    X=["0-2","2-4","4-6","6-8","8-10","10-12","12-14",">14","-1"]
    fig = plt.figure()  
    ax = fig.add_subplot(111)
    
    ## the data
    N = len(counts_train)

    ## necessary variables
    ind = np.arange(N)                # the x locations for the groups
    width = 0.35                      # the width of the bars
    ## the bars
    ax.bar(ind, counts_train, width,color='grey')
    
    # axes and labels
    ax.set_xlim(-width,len(ind)+width)
    ax.set_ylim(-10.5,2)
    ax.set_ylabel('Difference')
    ax.set_xlabel("Expected Answer Length")
    ax.set_title("Average Difference Between the Predicted \n Answer Length and the Expected Answer Length")
    ax.grid(axis='y')
    xTickMarks = X
    ax.set_xticks(ind)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=10)  
    plt.show()
    #SAMPLE SIZES = array([1328., 1066.,  281.,   95.,   43.,   12.,   11.,   12., 3103.])
    return


def decompose_error_types(dic_predicted,dic_expected):
    """
    Error[0]: Truth: Answer, Predict: Answer
    Error[1]: Truth: Answer, Predict: No Answer
    Error[2]: Truth: No Answer, Predict: Answer
    Error[3]: Truth: No Answer, Predict: No Answer
    dic_expected: obtained after using preprocess_gold_context(id_file)
    
    Gives each type of error
    """
    Errors=[0]*4
    Percent=[0]*4
    for ID in dic_predicted.keys():
        index_ans_e=dic_expected[ID]['answer']
        if index_ans_e!=-1:
            ans=dic_predicted[ID]
            if ans=='':
                Errors[1]+=1
            else:
                Errors[0]+=1
        if index_ans_e==-1:
            ans=dic_predicted[ID]
            if ans=='':
                Errors[3]+=1
            else:
                Errors[2]+=1
    for i in range(4):
        Percent[i]=Errors[i]/len(dic_predicted)
    return Errors,Percent

def approximate_match(A,B):
    """
    Computes if two string match approximately
    """
    lA=A.split(' ')
    lB=B.split(' ')
    result=0
    for i in lA:
        if i in lB:
            result+=1
    return result>=1
    
    
def exact_match(dic_predicted,dic_expected):
    """
    Attention: dic_expected: obtained after using preprocess_gold(id_file)
    
    Computes the number of errors in the case in which we're both
    predicting answer
    """
    N_exact_match=0
    N_no_match=0
    N_approximate_match=0
    count_match=0
    for ID in dic_predicted.keys():
        ans=dic_predicted[ID]
        if (dic_expected[ID]!=[] and ans!=''):
            count_match+=1
            exact_match=False
            approximate=False
            for i in dic_expected[ID]:
                string=i['text']
                if (string==ans):
                    exact_match=True
                elif approximate_match(string,ans):
                    approximate=True
            if exact_match:
                N_exact_match+=1
            elif approximate:
                N_approximate_match+=1
            else:
                N_no_match+=1
    return N_exact_match,  N_approximate_match, N_no_match, count_match

def match_f1(dic_predicted,dic_expected):
    """
    To compare with previous scores
    
    Computes the distribution of the F1 scores in the case in which we're both
    predicting answer
    """
    f1=np.array([])
    count_match=0
    for ID in dic_predicted.keys():
        ans=dic_predicted[ID]
        if (dic_expected[ID]!=[] and ans!=''):
            count_match+=1
            maximum=0
            for i in dic_expected[ID]:
                string=i['text']
                f1_score=compute_f1(string,ans)
                if f1_score>maximum:
                    maximum=f1_score
            f1=np.append(f1,maximum)
    return f1, count_match

def plot_f1_distribution(f1):
    """
    Plots the f1 distribution obtained above

    """
    plt.hist(f1,10,color='grey')
    plt.xlabel("F1 Score")
    plt.xlim(0,1)
    plt.ylabel("Count")
    plt.title("F1 Score Distribution between Predictions and Truth \n when Truth=Answer and Model=Answer")
    return
def avna_per_type(dic_questions,dic_predicted,dic_expected,quest):
    """
    Error[0]: Truth: Answer, Predict: Answer
    Error[1]: Truth: Answer, Predict: No Answer
    Error[2]: Truth: No Answer, Predict: Answer
    Error[3]: Truth: No Answer, Predict: No Answer
    dic_expected: obtained after using preprocess_gold_context(id_file)
    
    Not really useful
    """
    Errors=[0]*4
    Percent=[0]*4
    for ID in dic_predicted.keys():
        Type=get_question_type(dic_questions, ID)
        if(Type==quest):
            index_ans_e=dic_expected[ID]['answer']
            if index_ans_e!=-1:
                ans=dic_predicted[ID]
                if ans=='':
                    Errors[1]+=1
                else:
                    Errors[0]+=1
            if index_ans_e==-1:
                ans=dic_predicted[ID]
                if ans=='':
                    Errors[3]+=1
                else:
                    Errors[2]+=1
    for i in range(4):
        Percent[i]=Errors[i]/len(dic_predicted)
    return (Errors[0]+Errors[3])/(Errors[0]+Errors[1]+Errors[2]+Errors[3])

def compare_questions_types(dic_questions,dic_predicted,dic_expected,quest):
    """
    dic_expected: after preprocess_gold
    For a given type of question, computes f1,em,avna
    """
    f1=np.array([])
    em=np.array([])
    Errors=[0]*4
    for ID in dic_predicted.keys():
        Type=get_question_type(dic_questions, ID)
        if(Type==quest):
            ans=dic_predicted[ID]
            max_f1=0
            max_em=0
            if dic_expected[ID]==[]:
                max_f1=compute_f1('',ans)
                max_em=compute_em('',ans)
                if ans=='':
                    Errors[3]+=1
                else:
                    Errors[2]+=1
            else:
                if ans=='':
                    Errors[1]+=1
                else:
                    Errors[0]+=1
                for i in dic_expected[ID]:
                        string=i['text']
                        f1_score=compute_f1(string,ans)
                        em_score=compute_em(string,ans)
                        if f1_score>max_f1:
                            max_f1=f1_score
                        if em_score>max_em:
                            max_em=em_score
            f1=np.append(f1,max_f1)
            em=np.append(em,max_em)
    avna=(Errors[0]+Errors[3])/(Errors[0]+Errors[1]+Errors[2]+Errors[3])
    return np.mean(f1),np.mean(em),avna,len(f1)



dic_predicted=preprocess_results(result_file) 
dic_expected= preprocess_gold(id_file)
dic_expected_context=preprocess_gold_context(id_file)
dic_questions=preprocess_questions(id_file)

for i in range(0,8):
    print(compare_questions_types(dic_questions,dic_predicted,dic_expected,i))

#dic_length_predicted=process(dic_predicted)
#dic_length_small_expected=process_gold_dic_small_ans(dic_expected)
    
#result,binned=compare_length(dic_length_predicted,dic_length_small_expected)
#start_index_p_a,start_index_e_a=compare_start_index_answerable(dic_predicted,dic_expected_context)
#dif_ans=start_index_p_a-start_index_e_a      
#start_index_p,start_index_e= compare_start_index(dic_predicted,dic_expected_context) 
#dif=start_index_p-start_index_e 

        
      
#liste=["what","who","how","when","where","which","why"]
#WHAT: (0.6697747644639596, 0.6274567321795248, 0.7430331475506013, 3409)
#WHO: (0.7149522268454307, 0.6941747572815534, 0.7491909385113269, 618)
#HOW: (0.6697119355255645, 0.6175438596491228, 0.7263157894736842, 570)
#WHEN: (0.7662750405544198, 0.753880266075388, 0.8137472283813747, 451)
#WHERE: (0.66648658008658, 0.62, 0.756, 250)
#WHICH: (0.7512690505841191, 0.7123287671232876, 0.817351598173516, 219)
#WHY: (0.6491465163140354, 0.5465116279069767, 0.7209302325581395, 86)
#OTHER: (0.6687499024339795, 0.6408045977011494, 0.7471264367816092, 348)
# Counts with other method:
#3.567e+03, 6.270e+02, 5.710e+02,4.550e+02, 2.560e+02, 2.130e+02, 8.70e+01,8.40e+01, 9.10e+01          
            
  