#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:34:54 2020

@author: pabloveyrat
"""

import googletrans
import numpy as np
import os
#import spacy
import ujson as json
import urllib.request

from args import get_setup_args
from codecs import open
from collections import Counter
from subprocess import run
from tqdm import tqdm
from zipfile import ZipFile

import torch


id_file='./data/dev-v2.0.json'
def write_contexts(filename):
    result=[]
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in source["data"]:
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                result.append(context)
    return result


def process_file_augmentation_2(filename, data_type, word_counter, char_counter):
    print(f"Pre-processing {data_type} examples...")
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                context_tokens = word_tokenize(context)
                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                    example = {"context_tokens": context_tokens,
                               "context_chars": context_chars,
                               "ques_tokens": ques_tokens,
                               "ques_chars": ques_chars,
                               "y1s": y1s,
                               "y2s": y2s,
                               "id": total}
                    examples.append(example)
                    eval_examples[str(total)] = {"context": context,
                                                 "question": ques,
                                                 "spans": spans,
                                                 "answers": answer_texts,
                                                 "uuid": qa["id"]}
                    
                    context2,new_y1s,new_y2s,transformed=transform_context(context,y1s,y2s)
                    if (transformed):
                        context_tokens2 = word_tokenize(context2)
                        context_chars2 = [list(token) for token in context_tokens2]
                        spans2 = convert_idx(context2, context_tokens2)
                        for token in context_tokens2:
                            word_counter[token] += len(para["qas"])
                            for char in token:
                                char_counter[char] += len(para["qas"])
                        total+=1
                        example = {"context_tokens": context_tokens2,
                                   "context_chars": context_chars2,
                                   "ques_tokens": ques_tokens,
                                   "ques_chars": ques_chars,
                                   "y1s": new_y1s,
                                   "y2s": new_y2s,
                                   "id": total}
                        examples.append(example)
                        eval_examples[str(total)] = {"context": context2,
                                                     "question": ques,
                                                     "spans": spans2,
                                                     "answers": answer_texts,
                                                     "uuid": qa["id"]}                                       
        print(f"{len(examples)} questions in total")
    return examples, eval_examples

def transform_context_v2(context,y1s,y2s):
    transformed=False
    if (min(y1s)!=-1 and min(y2s)!=-1):
       index_end_translation=get_index_end_translation(context,y1s)
       index_start_translation=get_index_start_translation(context,y2s)

       if (index_end_translation==0):
           new_y1s=y1s
           new_y2s=y2s
           if (index_start_translation+1<len(context)-1):
               begin=context[index_start_translation+1:len(context)]
               last_s=backtranslation(begin)
               ans=context[index_end_translation:index_start_translation+2]+last_s
               transformed=True
           else:
               ans=context
       else: 
           end=context[0:index_end_translation]
           first_s=backtranslation(end)
           new_y1s=[0]*len(y1s)
           new_y2s=[0]*len(y2s)
           for i in range(len(y1s)):
               new_y1s[i]=y1s[i]+len(first_s)-len(end)
           for i in range(len(y2s)):
               new_y2s[i]=y2s[i]+len(first_s)-len(end)
           if (index_start_translation+1<len(context)-1):
               begin=context[index_start_translation+1:len(context)]
               last_s=backtranslation(begin)
               ans=first_s+context[index_end_translation:index_start_translation+2]+last_s
               transformed=True
           else:
               ans=first_s+context[index_end_translation:len(context)]
               transformed=True
    else:
        ans=context
        new_y1s=y1s
        new_y2s=y2s
    return ans,new_y1s,new_y2s,transformed

def get_index_start_translation(context,answer_end_tab):
    i=max(answer_end_tab)
    first_index=0
    result=False
    while i<len(context) and result==False:
        i+=1
        if (context[i]=='.'):
            first_index=i
            result=True
    return first_index

def get_index_end_translation(context,answer_start_tab):
    i=0
    answer_start=min(answer_start_tab)
    last_index=0
    while (i<answer_start):
        if (context[i]=='.'):
            last_index=i
        i+=1
    return last_index


def backtranslation_tab(tab):
    translator=googletrans.Translator()
    translations=translator.translate(tab,dest='fr',src='en')
    intermediate=[]
    for translation in translations:
        intermediate.append(translation.text)
    results=translator.translate(intermediate,dest='en',src='fr')
    final=[]
    for translation in results:
        final.append(translation.text)
    return final

def backtranslation_tab_segmented(tab):
    length=len(tab)
    result=[]
    i=0
    while (i<length-100):
        result.append(backtranslation_tab(tab[i:i+100]))
        print(i+100, "Over")
        i+=100
    result.append(backtranslation_tab(tab[i:len(tab)]))
    return result

def backtranslation_tab_segmented_2(tab):
    length=len(tab)
    result=[]
    i=0
    while (i<length):
        result.append(backtranslation(tab[i]))
        print(i, "Over")
        i+=1
    return result
