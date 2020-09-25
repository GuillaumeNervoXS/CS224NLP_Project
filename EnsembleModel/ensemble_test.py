#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:47:00 2020

@author: pabloveyrat
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import json
from util import compute_f1, compute_em, metric_max_over_ground_truths,compute_avna
from ujson import load as json_load
import csv
import pandas as pd

data_path_train='./data/train.npz'
data_path_dev='./data/dev.npz'
word2idx_file='./data/word2idx.json'

id_file='./data/dev-v2.0.json'
eval_file='./data/dev_eval.json'

qanet2='./models_for_test/qanet(2).csv'
#bidaf_baseline='./models_for_test/bidaf_baseline.csv'
bidaf_with_char='./models_for_test/bidaf_with_char.csv'
bidaf_fusion='./models_for_test/bidaf_fusion.csv'
qanet_dif_output='./models_for_test/qanet_diff_output.csv'
qanet_dif_output_2='./models_for_test/qanet_diff_output_2.csv'
qanet_unshared_weights='./models_for_test/qanet_unshared_weights.csv'
qanet_unshared_weights_2='./models_for_test/qanet_unshared_weights_2.csv'
qanet1='./models_for_test/qanet(1).csv'
qanet3='./models_for_test/qanet(3).csv'
test_guill='./models_for_test/test_guillaume.csv'


#dataset_dev = np.load(data_path_dev)
#dataset_train = np.load(data_path_train)

qanet_csv=pd.read_csv(qanet2)

def preprocess_results(result_file):
    df=pd.read_csv(result_file)
    df=df.fillna('')
    dic={}
    for index,row in df.iterrows():
        dic[row['Id']]=row['Predicted']
    return dic

def preprocess_results_2(result_file):
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

def preprocess_eval(eval_file):
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    return gold_dict

def preprocess_gold_dict(id_file):
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
                    if answer==[]:
                        dic[ID]=[]
                    else:
                        tab=[]
                        for k in answer:
                            tab+=[k['text']]
                        dic[ID]=tab
    return dic


def merge(best_dic,dic2,dic3):
    """ 
    Achieves {'EM': 66.79549655520081, 'F1': 69.76054094495765, 'AvNA': 75.3486808939674}
    """
    dic={}
    count1=0
    count2=0
    count3=0
    count4=0
    ids=[]
    for ID in best_dic.keys():
        best_value=best_dic[ID]
        value2=dic2[ID]
        value3=dic3[ID]
        if(best_value==value2):
            dic[ID]=best_value
            count1+=1
        elif (best_value==value3):
            dic[ID]=best_value
            count2+=1
        elif (value2==value3 and best_value!=''):
            ids+=[ID]
            dic[ID]=value2 
            count3+=1
        else:
            dic[ID]=best_value 
            count4+=1
    count=[count1,count2,count3,count4]
    return dic #,count,ids

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
    return result>=2

def merge_2(best_dic,dic2,dic3):
    """
    Achieves {'EM': 66.94673164174088, 'F1': 69.82204613378883, 'AvNA': 75.3486808939674}
    """
    dic={}
    count1=0
    count2=0
    count3=0
    count4=0
    ids=[]
    for ID in best_dic.keys():
        best_value=best_dic[ID]
        value2=dic2[ID]
        value3=dic3[ID]
        if(best_value==value2):
            dic[ID]=best_value
            count1+=1
        elif (best_value==value3):
            dic[ID]=best_value
            count2+=1
        elif (value2==value3 and best_value!=''):
            dic[ID]=value2 
            count3+=1
        elif (value2 in best_value and value2!=''):
            dic[ID]=value2
        elif (value3 in best_value and value3!=''):
            dic[ID]=value3
        else:
            dic[ID]=best_value 
            count4+=1
            #if approximate_match(best_value,value2) or approximate_match(best_value,value2):
                #ids+=[ID]
    count=[count1,count2,count3,count4]
    return dic #,count,ids

def merge_3(best_dic,dic2,dic3):
    """
    Achieves {'EM': 67.61888758191901, 'F1': 70.163888643803, 'AvNA': 74.84456393883382}
    Higher F1 but lower AvNA (check on the test submission what it does)
    CSV FILE NOT WRITTEN
    
    {'EM': 68.1566123340615, 'F1': 71.18285555436506, 'AvNA': 76.49134599227021}
    with QANet 2, dic_output, dic_weights
    
    {'EM': 68.49269030415056, 'F1': 71.33086290644673, 'AvNA': 76.00403293564106}
    <ith QANet 2, dic_fusion, dic_output
    """
    dic={}
    count1=0
    count2=0
    count3=0
    count4=0
    ids=[]
    for ID in best_dic.keys():
        best_value=best_dic[ID]
        value2=dic2[ID]
        value3=dic3[ID]
        if(best_value==value2):
            dic[ID]=best_value
            count1+=1
        elif (best_value==value3):
            dic[ID]=best_value
            count2+=1
        elif (value2==value3 and best_value!=''):
            dic[ID]=value2 
            count3+=1
        elif (value2 in best_value):
            dic[ID]=value2
        elif (value3 in best_value):
            dic[ID]=value3
        else:
            dic[ID]=best_value 
            count4+=1
            #if approximate_match(best_value,value2) or approximate_match(best_value,value2):
                #ids+=[ID]
    count=[count1,count2,count3,count4]
    return dic #,count,ids

#pred_dict,count,ids=merge(dic_qanet,dic_with_char,dic_baseline)

#pred_dict,count,ids=merge_2(dic_qanet,dic_with_char,dic_baseline)
#pred_dict2=merge(dic_qanet,dic_qanet,dic_qanet)

"""
for i in ids:
    print("QANET:", dic_qanet[i])
    print("Baseline: ", dic_baseline[i])
    print("with chars: ", dic_with_char[i])
    print("Expected:", gold_dict[i])
"""

def merge_4(dics):
    """
    Dics: list of all the dictionnaries that we have
    Achieves {'EM': 68.87917996975298, 'F1': 71.54610641857572, 'AvNA': 76.32330700722568}
    """
    dic={}
    best_dic=dics[0]
    second_best=dics[1]
    L=len(dics)
    dic_length={}
    index_best_v_tab={}
    for i in range(1,len(dics)+1):
        dic_length[i]=0
        index_best_v_tab[i-1]=0
    for ID in best_dic.keys():
        best_value=best_dic[ID]
        liste_ans=[dics[i][ID] for i in range(len(dics))]
        counts_dic=Counter(liste_ans)
        high=counts_dic.most_common(len(dics))
        index_best_v=0
        dic_length[len(high)]+=1
        for i in range(len(high)):
            if high[i][0] == best_value:
                index_best_v=i
        index_best_v_tab[index_best_v]+=1 
        
        if len(high)==1 or len(high)==len(dics) or index_best_v==0:
            dic[ID]=best_value
            
            
        elif high[0][1]>high[1][1] and best_value!='':
            dic[ID]=high[0][0]

        elif len(high)==2 and high[0][1]==high[1][1]:
            dic[ID]=best_value
 
        elif high[0][0] in best_value:
            dic[ID]=high[0][0]
            
        elif high[1][0] in best_value and high[0][1]==high[1][1]:
            dic[ID]=high[1][0]
         
        elif high[index_best_v][1]>1:
            dic[ID]=best_value 
              
        elif len(high)==3:
            if high[2][1]==high[1][1]:
                dic[ID]=best_value
            elif index_best_v==3:
                dic[ID]=best_value
            else:
                dic[ID]=high[0][0]
        #elif len(high)>=4:
           # dic[ID]=high[0][0]
        else:
            dic[ID]=best_value
        
    return dic,dic_length,index_best_v_tab #,count,ids

def merge_5(dics):
    """
    Dics: list of all the dictionnaries that we have
    Achieves {'EM': 68.84557217274407, 'F1': 71.54128208874806, 'AvNA': 76.35691480423458}
    """
    dic={}
    best_dic=dics[0]
    second_best=dics[1]
    L=len(dics)
    dic_length={}
    index_best_v_tab={}
    for i in range(1,len(dics)+1):
        dic_length[i]=0
        index_best_v_tab[i-1]=0
    for ID in best_dic.keys():
        best_value=best_dic[ID]
        liste_ans=[dics[i][ID] for i in range(len(dics))]
        counts_dic=Counter(liste_ans)
        high=counts_dic.most_common(len(dics))
        index_best_v=0
        dic_length[len(high)]+=1
        for i in range(len(high)):
            if high[i][0] == best_value:
                index_best_v=i
        index_best_v_tab[index_best_v]+=1 
        
        if len(high)==1 or len(high)==len(dics) or index_best_v==0:
            dic[ID]=best_value
            
            
        elif high[0][1]>high[1][1] and best_value!='':
            dic[ID]=high[0][0]
        
        elif len(high)==2 and high[0][1]==high[1][1]:
            dic[ID]=best_value
 
        elif high[0][0] in best_value:
            dic[ID]=high[0][0]
            
        elif high[1][0] in best_value and high[0][1]==high[1][1]:
            dic[ID]=high[1][0]
         
        elif high[index_best_v][1]>1:
            dic[ID]=best_value 
              
        elif len(high)==3:
            if high[2][1]==high[1][1]:
                dic[ID]=best_value
            elif index_best_v==3:
                dic[ID]=best_value
            else:
                dic[ID]=high[0][0]
        #elif len(high)>=4:
           # dic[ID]=high[0][0]
        else:
            dic[ID]=best_value
        
    return dic,dic_length,index_best_v_tab #,count,ids

def eval_dicts(gold_dict, pred_dict, no_answer=True):
    avna = f1 = em = total = 0
    for key, value in pred_dict.items():
        total += 1
        ground_truths = gold_dict[key]
        prediction = value
        em += metric_max_over_ground_truths(compute_em, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(compute_f1, prediction, ground_truths)
        if no_answer:
            avna += compute_avna(prediction, ground_truths)

    eval_dict = {'EM': 100. * em / total,
                 'F1': 100. * f1 / total}

    if no_answer:
        eval_dict['AvNA'] = 100. * avna / total

    return eval_dict

def write_csv(sub_dict):
    with open('./save/test_submission.csv', 'w', newline='', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for uuid in sorted(sub_dict):
            csv_writer.writerow([uuid, sub_dict[uuid]])

dic_qanet2_t=preprocess_results(qanet2)
dic_with_char_t=preprocess_results(bidaf_with_char)
dic_fusion_t=preprocess_results(bidaf_fusion)
dic_output_t=preprocess_results(qanet_dif_output)
dic_output_2_t=preprocess_results(qanet_dif_output_2)
dic_weights_t=preprocess_results(qanet_unshared_weights)
dic_weights_2_t=preprocess_results(qanet_unshared_weights_2)
dic_qanet1_t=preprocess_results(qanet1)
dic_qanet3_t=preprocess_results(qanet3)
dic_guill_t=preprocess_results(test_guill)

gold_dict=preprocess_gold_dict(id_file)


dics_t=[dic_qanet2_t,dic_qanet3_t,dic_output_2_t,dic_with_char_t,dic_fusion_t,dic_output_t,dic_weights_t,dic_qanet1_t,dic_weights_2_t]



