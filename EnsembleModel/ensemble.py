# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 13:45:07 2020

@author: guill
"""
import csv
import numpy as np
import torch
import torch.utils.data as data
import util

from args import get_test_args
from collections import OrderedDict
from json import dumps
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD
from pickle import load


split='test'

args=get_test_args()
args.split=split

eval_file = vars(args)[f'{args.split}_eval_file']
with open(eval_file, 'r') as fh:
    gold_dict = json_load(fh)

#forgot to take the id of the question, normally ok to do it afterwards 
#as the dataset is not shuffle in test and here
record_file = vars(args)[f'{args.split}_record_file']
dataset = SQuAD(record_file, args.use_squad_v2)
data_loader = data.DataLoader(dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              collate_fn=collate_fn)

id_quest=[]
with torch.no_grad(), \
    tqdm(total=len(dataset)) as progress_bar:
    for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
        id_quest+=[ids]

p_start=[]
p_end=[]

save_dir_bidaf="../save/test/"+split+"/bidaf_original_char_embed-01/"
with open(save_dir_bidaf+"/probs_start", "rb") as fp:   #Pickling
    p_start+=[load(fp)]
with open(save_dir_bidaf+"/probs_end", "rb") as fp:   #Pickling
    p_end+=[load(fp)]    
  
save_dir_bidaf_fusion="../save/test/"+split+"/bidaf_fusion-01/"
with open(save_dir_bidaf_fusion+"/probs_start", "rb") as fp:   #Pickling
    p_start+=[load(fp)]
with open(save_dir_bidaf_fusion+"/probs_end", "rb") as fp:   #Pickling
    p_end+=[load(fp)]  

save_dir_qanet_old_s="../save/test/"+split+"/qanet_old_1_small-01/"
with open(save_dir_qanet_old_s+"/probs_start", "rb") as fp:   #Pickling
    p_start+=[load(fp)]
with open(save_dir_qanet_old_s+"/probs_end", "rb") as fp:   #Pickling
    p_end+=[load(fp)]     
    
save_dir_qanet_old_2="../save/test/"+split+"/qanet_old_2-01/"
with open(save_dir_qanet_old_2+"/probs_start", "rb") as fp:   #Pickling
    p_start+=[load(fp)]
with open(save_dir_qanet_old_2+"/probs_end", "rb") as fp:   #Pickling
    p_end+=[load(fp)]  

save_dir_qanet_output="../save/test/"+split+"/QANet_outputprob_merge_start_end-01/"
with open(save_dir_qanet_output+"/probs_start", "rb") as fp:   #Pickling
    p_start+=[load(fp)]
with open(save_dir_qanet_output+"/probs_end", "rb") as fp:   #Pickling
    p_end+=[load(fp)]   

save_dir_qanet_inde="../save/test/"+split+"/QANet_independance_encoder-01/"
with open(save_dir_qanet_inde+"/probs_start", "rb") as fp:   #Pickling
    p_start+=[load(fp)]
with open(save_dir_qanet_inde+"/probs_end", "rb") as fp:   #Pickling
    p_end+=[load(fp)]  
 
save_dir_qanet_inde_lower="../save/test/"+split+"/QANet_independance_encoder_not_best-01/"
with open(save_dir_qanet_inde_lower+"/probs_start", "rb") as fp:   #Pickling
    p_start+=[load(fp)]
with open(save_dir_qanet_inde_lower+"/probs_end", "rb") as fp:   #Pickling
    p_end+=[load(fp)] 
 
save_dir_qanet_large="../save/test/"+split+"/QANet_large-01/"
with open(save_dir_qanet_inde+"/probs_start", "rb") as fp:   #Pickling
    p_start+=[load(fp)]
with open(save_dir_qanet_inde+"/probs_end", "rb") as fp:   #Pickling
    p_end+=[load(fp)]  
 
save_dir_qanet_inde_output_large="../save/test/"+split+"/QANet_output_large-01/"
with open(save_dir_qanet_inde_lower+"/probs_start", "rb") as fp:   #Pickling
    p_start+=[load(fp)]
with open(save_dir_qanet_inde_lower+"/probs_end", "rb") as fp:   #Pickling
    p_end+=[load(fp)]     

    
assert len(p_start)==len(p_end)
for i in range(len(p_start)):
    for j in range(len(p_start[i])):        
        assert len(p_start[i][j].shape)==len(p_end[i][j].shape)



def AverageProbs(args , p_start ,p_end , id_quest, gold_dict):
    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}   # Predictions for submission
    nbr_model=len(p_start)
    nbr_batch=len(p_start[0])
    for j in range(nbr_batch):
        p1=np.zeros(p_start[0][j].shape)
        p2=np.zeros(p_start[0][j].shape)
        for model in range(nbr_model):
            p1+=p_start[model][j]
            p2+=p_end[model][j]
        
        p1/=nbr_model
        p2/=nbr_model
        ids=id_quest[j]
        
        p1=torch.from_numpy(p1).float().cuda()
        p2=torch.from_numpy(p2).float().cuda()
        ids=ids.cuda()
    
        
        starts, ends = util.discretize(p1, p2, args.max_ans_len, args.use_squad_v2)
        
        idx2pred, uuid2pred = util.convert_tokens(gold_dict,ids.tolist(),
                                                  starts.tolist(),ends.tolist(),
                                                  args.use_squad_v2)
        pred_dict.update(idx2pred)
        sub_dict.update(uuid2pred)
    
    
    
    results = util.eval_dicts(gold_dict, pred_dict, args.use_squad_v2)
    results_list = [('F1', results['F1']),
                     ('EM', results['EM'])]
    if args.use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)
    
    # Log to console
    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
    print(f'{args.split.title()} {results_str}')
    
    # Write submission file
    sub_path = join("../save/test/"+split+"/submission.csv")
    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for uuid in sorted(sub_dict):
            csv_writer.writerow([uuid, sub_dict[uuid]])

weigths=np.array([62.92,65.76,66.04,66.43,67.03,66.87,64.43,67.95,66.87])
weigths/=np.sum(weigths)

def WeightedAverage(args , p_start ,p_end , weigths, id_quest, gold_dict):
    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}   # Predictions for submission
    nbr_model=len(p_start)
    nbr_batch=len(p_start[0])
    for j in range(nbr_batch):
        p1=np.zeros(p_start[0][j].shape)
        p2=np.zeros(p_start[0][j].shape)
        for model in range(nbr_model):
            p1+=weigths[model]*p_start[model][j]
            p2+=weigths[model]*p_end[model][j]
        
        ids=id_quest[j]
        
        p1=torch.from_numpy(p1).float().cuda()
        p2=torch.from_numpy(p2).float().cuda()
        ids=ids.cuda()
    
        
        starts, ends = util.discretize(p1, p2, args.max_ans_len, args.use_squad_v2)
        
        idx2pred, uuid2pred = util.convert_tokens(gold_dict,ids.tolist(),
                                                  starts.tolist(),ends.tolist(),
                                                  args.use_squad_v2)
        pred_dict.update(idx2pred)
        sub_dict.update(uuid2pred)
    
    
    
    results = util.eval_dicts(gold_dict, pred_dict, args.use_squad_v2)
    results_list = [('F1', results['F1']),
                     ('EM', results['EM'])]
    if args.use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)
    
    # Log to console
    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
    print(f'{args.split.title()} {results_str}')


def discretizeAndMax(p_start, p_end, max_len=15, no_answer=False):
    """Discretize soft predictions to get start and end indices.

    Choose the pair `(i, j)` of indices that maximizes `p1[i] * p2[j]`
    subject to `i <= j` and `j - i + 1 <= max_len`.

    Args:
        p_start (torch.Tensor): Soft predictions for start index.
            Shape (batch_size, context_len).
        p_end (torch.Tensor): Soft predictions for end index.
            Shape (batch_size, context_len).
        max_len (int): Maximum length of the discretized prediction.
            I.e., enforce that `preds[i, 1] - preds[i, 0] + 1 <= max_len`.
        no_answer (bool): Treat 0-index as the no-answer prediction. Consider
            a prediction no-answer if `preds[0, 0] * preds[0, 1]` is greater
            than the probability assigned to the max-probability span.

    Returns:
        start_idxs (torch.Tensor): Hard predictions for start index.
            Shape (batch_size,)
        end_idxs (torch.Tensor): Hard predictions for end index.
            Shape (batch_size,)
    """
    if p_start.min() < 0 or p_start.max() > 1 \
            or p_end.min() < 0 or p_end.max() > 1:
        raise ValueError('Expected p_start and p_end to have values in [0, 1]')

    # Compute pairwise probabilities
    p_start = p_start.unsqueeze(dim=2)
    p_end = p_end.unsqueeze(dim=1)
    p_joint = torch.matmul(p_start, p_end)  # (batch_size, c_len, c_len)

    # Restrict to pairs (i, j) such that i <= j <= i + max_len - 1
    c_len, device = p_start.size(1), p_start.device
    is_legal_pair = torch.triu(torch.ones((c_len, c_len), device=device))
    is_legal_pair -= torch.triu(torch.ones((c_len, c_len), device=device),
                                diagonal=max_len)
    if no_answer:
        # Index 0 is no-answer
        p_no_answer = p_joint[:, 0, 0].clone()
        is_legal_pair[0, :] = 0
        is_legal_pair[:, 0] = 0
    else:
        p_no_answer = None
    p_joint *= is_legal_pair

    # Take pair (i, j) that maximizes p_joint
    max_in_row, _ = torch.max(p_joint, dim=2)
    max_in_col, _ = torch.max(p_joint, dim=1)
    start_idxs = torch.argmax(max_in_row, dim=-1)
    end_idxs = torch.argmax(max_in_col, dim=-1)

    if no_answer:
        # Predict no-answer whenever p_no_answer > max_prob
        max_prob, _ = torch.max(max_in_col, dim=-1)
        start_idxs[p_no_answer > max_prob] = 0
        end_idxs[p_no_answer > max_prob] = 0

    return start_idxs, end_idxs, torch.max(max_in_row, dim=-1)[0]



def MaxProbs(args , p_start ,p_end , id_quest, gold_dict):
    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}   # Predictions for submission
    nbr_model=len(p_start)
    nbr_batch=len(p_start[0])
    for j in range(nbr_batch):
        prob=torch.zeros(nbr_model,p_start[0][j].shape[0])
        start=-1*torch.ones((nbr_model,p_start[0][j].shape[0]),dtype=torch.long)
        end=-1*torch.ones((nbr_model,p_start[0][j].shape[0]),dtype=torch.long)
        
        ids=id_quest[j]
        
        for model in range(nbr_model):
            cur_p1=torch.from_numpy(p_start[model][j]).float().cuda()
            cur_p2=torch.from_numpy(p_end[model][j]).float().cuda()
            ids=ids.cuda()
    
            s_model, e_model, probs = discretizeAndMax(cur_p1, cur_p2, args.max_ans_len, args.use_squad_v2)
            start[model,:]=s_model
            end[model,:]=e_model
            prob[model,:]=probs
    
        index_s_e=torch.argmax(prob,dim=0)
        starts=torch.gather(start, 0, index_s_e.unsqueeze(0)).squeeze()
        ends=torch.gather(end, 0, index_s_e.unsqueeze(0)).squeeze()
        
        idx2pred, uuid2pred = util.convert_tokens(gold_dict,ids.tolist(),
                                                  starts.tolist(),ends.tolist(),
                                                  args.use_squad_v2)
        pred_dict.update(idx2pred)
        sub_dict.update(uuid2pred)
    
    
    
    results = util.eval_dicts(gold_dict, pred_dict, args.use_squad_v2)
    results_list = [('F1', results['F1']),
                     ('EM', results['EM'])]
    if args.use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)
    
    # Log to console
    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
    print(f'{args.split.title()} {results_str}')
    
    
AverageProbs(args, p_start, p_end, id_quest, gold_dict)
#WeightedAverage(args, p_start, p_end, weigths, id_quest, gold_dict)
#MaxProbs(args, p_start, p_end, id_quest, gold_dict)