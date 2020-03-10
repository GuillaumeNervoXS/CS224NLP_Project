"""Test a model and generate submission CSV.

Usage:
    > python test.py --split SPLIT --load_path PATH --name NAME
    where
    > SPLIT is either "dev" or "test"
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the test run

Author:
    Chris Chute (chute@stanford.edu)
"""

import csv
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import util

from args import get_test_args
from collections import OrderedDict
from json import dumps
from models import Baseline,BiDAF, BiDAF_fus ,QANet , QANet_S_E, QANet_independant_encoder, QANet_old
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD


def main(args):
    # Set up logging
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    device, gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(gpu_ids))

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)
    char_vectors = util.torch_from_json(args.char_emb_file)
    # Get model
    log.info('Building model...')
    nbr_model=0
    if(args.load_path_baseline):
        model_baseline = Baseline(word_vectors=word_vectors,hidden_size=100)
        model_baseline = nn.DataParallel(model_baseline, gpu_ids)
        log.info(f'Loading checkpoint from {args.load_path_baseline}...')
        model_baseline = util.load_model(model_baseline, args.load_path_baseline, gpu_ids, return_step=False)
        model_baseline = model_baseline.to(device)
        model_baseline.eval()
        nll_meter_baseline = util.AverageMeter()
        nbr_model+=1
        save_prob_baseline_start=[]
        save_prob_baseline_end=[]
        
    if(args.load_path_bidaf):
        model_bidaf = BiDAF(word_vectors=word_vectors,char_vectors=char_vectors,
                            char_emb_dim=args.char_emb_dim,hidden_size=args.hidden_size)
        model_bidaf = nn.DataParallel(model_bidaf, gpu_ids)
        log.info(f'Loading checkpoint from {args.load_path_bidaf}...')       
        model_bidaf = util.load_model(model_bidaf, args.load_path_bidaf, gpu_ids, return_step=False)
        model_bidaf = model_bidaf.to(device)
        model_bidaf.eval()
        nll_meter_bidaf = util.AverageMeter()
        nbr_model+=1
        save_prob_bidaf_start=[]
        save_prob_bidaf_end=[]

    if(args.load_path_bidaf_fusion):
        model_bidaf_fu = BiDAF_fus(word_vectors=word_vectors,char_vectors=char_vectors,
                                   char_emb_dim=args.char_emb_dim,hidden_size=args.hidden_size)
        model_bidaf_fu = nn.DataParallel(model_bidaf_fu, gpu_ids)
        log.info(f'Loading checkpoint from {args.load_path_bidaf_fusion}...')       
        model_bidaf_fu = util.load_model(model_bidaf_fu, args.load_path_bidaf_fusion, gpu_ids, return_step=False)
        model_bidaf_fu = model_bidaf_fu.to(device)
        model_bidaf_fu.eval()
        nll_meter_bidaf_fu = util.AverageMeter()
        nbr_model+=1
        save_prob_bidaf_fu_start=[]
        save_prob_bidaf_fu_end=[]

    if(args.load_path_qanet):
        model_qanet = QANet(word_vectors=word_vectors,char_vectors=char_vectors, char_emb_dim=args.char_emb_dim,
                            hidden_size=args.hidden_size,n_heads=args.n_heads, n_conv_emb_enc=args.n_conv_emb,
                            n_conv_mod_enc=args.n_conv_mod, n_emb_enc_blocks=args.n_emb_blocks,
                            n_mod_enc_blocks=args.n_mod_blocks, divisor_dim_kqv=args.divisor_dim_kqv)
        
        model_qanet = nn.DataParallel(model_qanet, gpu_ids)
        log.info(f'Loading checkpoint from {args.load_path_qanet}...')
        model_qanet = util.load_model(model_qanet, args.load_path_qanet, gpu_ids, return_step=False)
        model_qanet = model_qanet.to(device)
        model_qanet.eval()
        nll_meter_qanet = util.AverageMeter()
        nbr_model+=1
        save_prob_qanet_start=[]
        save_prob_qanet_end=[]
    
    if(args.load_path_qanet_old):
        model_qanet_old = QANet_old(word_vectors=word_vectors,char_vectors=char_vectors,device=device,char_emb_dim=args.char_emb_dim,
                            hidden_size=args.hidden_size,n_heads=args.n_heads, n_conv_emb_enc=args.n_conv_emb,
                            n_conv_mod_enc=args.n_conv_mod, n_emb_enc_blocks=args.n_emb_blocks,
                            n_mod_enc_blocks=args.n_mod_blocks)
        
        model_qanet_old = nn.DataParallel(model_qanet_old, gpu_ids)
        log.info(f'Loading checkpoint from {args.load_path_qanet_old}...')
        model_qanet_old = util.load_model(model_qanet_old, args.load_path_qanet_old, gpu_ids, return_step=False)
        model_qanet_old = model_qanet_old.to(device)
        model_qanet_old.eval()
        nll_meter_qanet_old = util.AverageMeter()
        nbr_model+=1
        save_prob_qanet_old_start=[]
        save_prob_qanet_old_end=[]
    
    if(args.load_path_qanet_inde):
        model_qanet_inde = QANet_independant_encoder(word_vectors=word_vectors,char_vectors=char_vectors, char_emb_dim=args.char_emb_dim,
                            hidden_size=args.hidden_size,n_heads=args.n_heads, n_conv_emb_enc=args.n_conv_emb,
                            n_conv_mod_enc=args.n_conv_mod, n_emb_enc_blocks=args.n_emb_blocks,
                            n_mod_enc_blocks=args.n_mod_blocks, divisor_dim_kqv=args.divisor_dim_kqv)
        
        model_qanet_inde = nn.DataParallel(model_qanet_inde, gpu_ids)
        log.info(f'Loading checkpoint from {args.load_path_qanet_inde}...')
        model_qanet_inde = util.load_model(model_qanet_inde, args.load_path_qanet_inde, gpu_ids, return_step=False)
        model_qanet_inde = model_qanet_inde.to(device)
        model_qanet_inde.eval()
        nll_meter_qanet_inde = util.AverageMeter()
        nbr_model+=1
        save_prob_qanet_inde_start=[]
        save_prob_qanet_inde_end=[]
    
    
    if(args.load_path_qanet_s_e):
        model_qanet_s_e = QANet_S_E(word_vectors=word_vectors,char_vectors=char_vectors, char_emb_dim=args.char_emb_dim,
                            hidden_size=args.hidden_size, n_heads=args.n_heads, n_conv_emb_enc=args.n_conv_emb,
                            n_conv_mod_enc=args.n_conv_mod, n_emb_enc_blocks=args.n_emb_blocks,
                            n_mod_enc_blocks=args.n_mod_blocks, divisor_dim_kqv=args.divisor_dim_kqv)
        
        model_qanet_s_e = nn.DataParallel(model_qanet_s_e, gpu_ids)
        log.info(f'Loading checkpoint from {args.load_path_qanet_s_e}...')
        model_qanet_s_e = util.load_model(model_qanet_s_e, args.load_path_qanet_s_e, gpu_ids, return_step=False)
        model_qanet_s_e = model_qanet_s_e.to(device)
        model_qanet_s_e.eval()
        nll_meter_qanet_s_e = util.AverageMeter()
        nbr_model+=1
        save_prob_qanet_s_e_start=[]
        save_prob_qanet_s_e_end=[]
    
    
    # Get data loader
    log.info('Building dataset...')
    record_file = vars(args)[f'{args.split}_record_file']
    dataset = SQuAD(record_file, args.use_squad_v2)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn)
        
    # Evaluate
    log.info(f'Evaluating on {args.split} split...')
    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}   # Predictions for submission
    eval_file = vars(args)[f'{args.split}_eval_file']
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            cc_idxs = cc_idxs.to(device)
            qc_idxs = qc_idxs.to(device)
            batch_size = cw_idxs.size(0)

            y1, y2 = y1.to(device), y2.to(device)
            l_p1,l_p2=[],[]
            # Forward
            if(args.load_path_baseline):
                log_p1_baseline, log_p2_baseline = model_baseline(cw_idxs, cc_idxs)
                loss_baseline = F.nll_loss(log_p1_baseline, y1) + F.nll_loss(log_p2_baseline, y2)
                nll_meter_baseline.update(loss_baseline.item(), batch_size)
                l_p1+=[log_p1_baseline.exp()]
                l_p2+=[log_p2_baseline.exp()]
                if(args.save_probabilities):
                    save_prob_baseline_start+=[log_p1_baseline.exp().detach().cpu().numpy()]
                    save_prob_baseline_end+=[log_p2_baseline.exp().detach().cpu().numpy()]
                    
            if(args.load_path_qanet):
                log_p1_qanet, log_p2_qanet = model_qanet(cw_idxs, cc_idxs, qw_idxs, qc_idxs)
                loss_qanet = F.nll_loss(log_p1_qanet, y1) + F.nll_loss(log_p2_qanet, y2)
                nll_meter_qanet.update(loss_qanet.item(), batch_size)
                # Get F1 and EM scores
                l_p1+=[log_p1_qanet.exp()]
                l_p2+=[log_p2_qanet.exp()]
                if(args.save_probabilities):
                    save_prob_qanet_start+=[log_p1_qanet.exp().detach().cpu().numpy()]
                    save_prob_qanet_end+=[log_p2_qanet.exp().detach().cpu().numpy()]
            
            if(args.load_path_qanet_old):
                log_p1_qanet_old, log_p2_qanet_old = model_qanet_old(cw_idxs, cc_idxs, qw_idxs, qc_idxs)
                loss_qanet_old = F.nll_loss(log_p1_qanet_old, y1) + F.nll_loss(log_p2_qanet_old, y2)
                nll_meter_qanet_old.update(loss_qanet_old.item(), batch_size)
                # Get F1 and EM scores
                l_p1+=[log_p1_qanet_old.exp()]
                l_p2+=[log_p2_qanet_old.exp()]
                if(args.save_probabilities):
                    save_prob_qanet_old_start+=[log_p1_qanet_old.exp().detach().cpu().numpy()]
                    save_prob_qanet_old_end+=[log_p2_qanet_old.exp().detach().cpu().numpy()]
            
            if(args.load_path_qanet_inde):
                log_p1_qanet_inde, log_p2_qanet_inde = model_qanet_inde(cw_idxs, cc_idxs, qw_idxs, qc_idxs)
                loss_qanet_inde = F.nll_loss(log_p1_qanet_inde, y1) + F.nll_loss(log_p2_qanet_inde, y2)
                nll_meter_qanet_inde.update(loss_qanet_inde.item(), batch_size)
                # Get F1 and EM scores
                l_p1+=[log_p1_qanet_inde.exp()]
                l_p2+=[log_p2_qanet_inde.exp()]
                if(args.save_probabilities):
                    save_prob_qanet_inde_start+=[log_p1_qanet_inde.exp().detach().cpu().numpy()]
                    save_prob_qanet_inde_end+=[log_p2_qanet_inde.exp().detach().cpu().numpy()]
            
            if(args.load_path_qanet_s_e):
                log_p1_qanet_s_e, log_p2_qanet_s_e = model_qanet_s_e(cw_idxs, cc_idxs, qw_idxs, qc_idxs)
                loss_qanet_s_e = F.nll_loss(log_p1_qanet_s_e, y1) + F.nll_loss(log_p2_qanet_s_e, y2)
                nll_meter_qanet_s_e.update(loss_qanet_s_e.item(), batch_size)
                # Get F1 and EM scores
                l_p1+=[log_p1_qanet_s_e.exp()]
                l_p2+=[log_p2_qanet_s_e.exp()]
                if(args.save_probabilities):
                    save_prob_qanet_s_e_start+=[log_p1_qanet_s_e.exp().detach().cpu().numpy()]
                    save_prob_qanet_s_e_end+=[log_p2_qanet_s_e.exp().detach().cpu().numpy()]
            
            if(args.load_path_bidaf):
                log_p1_bidaf, log_p2_bidaf = model_bidaf(cw_idxs, cc_idxs, qw_idxs, qc_idxs)
                loss_bidaf = F.nll_loss(log_p1_bidaf, y1) + F.nll_loss(log_p2_bidaf, y2)
                nll_meter_bidaf.update(loss_bidaf.item(), batch_size)
                l_p1+=[log_p1_bidaf.exp()]
                l_p2+=[log_p2_bidaf.exp()]
                if(args.save_probabilities):
                    save_prob_bidaf_start+=[log_p1_bidaf.exp().detach().cpu().numpy()]
                    save_prob_bidaf_end+=[log_p2_bidaf.exp().detach().cpu().numpy()]
                
            
            if(args.load_path_bidaf_fusion):
                log_p1_bidaf_fu, log_p2_bidaf_fu = model_bidaf_fu(cw_idxs, cc_idxs, qw_idxs, qc_idxs)
                loss_bidaf_fu = F.nll_loss(log_p1_bidaf_fu, y1) + F.nll_loss(log_p2_bidaf_fu, y2)
                nll_meter_bidaf_fu.update(loss_bidaf_fu.item(), batch_size)
                l_p1+=[log_p1_bidaf_fu.exp()]
                l_p2+=[log_p2_bidaf_fu.exp()]
                if(args.save_probabilities):
                    save_prob_bidaf_fu_start+=[log_p1_bidaf_fu.exp().detach().cpu().numpy()]
                    save_prob_bidaf_fu_end+=[log_p2_bidaf_fu.exp().detach().cpu().numpy()]
            
            
            p1,p2=l_p1[0],l_p2[0]
            for i in range(1,nbr_model):
                p1+=l_p1[i]
                p2+=l_p2[i]
            p1/=nbr_model
            p2/=nbr_model
            
            starts, ends = util.discretize(p1, p2, args.max_ans_len, args.use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            if args.split != 'test':
                # No labels for the test set, so NLL would be invalid
                if(args.load_path_qanet):
                    progress_bar.set_postfix(NLL=nll_meter_qanet.avg)
                elif(args.load_path_bidaf):
                    progress_bar.set_postfix(NLL=nll_meter_bidaf.avg)
                elif(args.load_path_bidaf_fusion):
                    progress_bar.set_postfix(NLL=nll_meter_bidaf_fu.avg)
                elif(args.load_path_qanet_old):
                    progress_bar.set_postfix(NLL=nll_meter_qanet_old.avg)
                elif(args.load_path_qanet_inde):
                    progress_bar.set_postfix(NLL=nll_meter_qanet_inde.avg)
                elif(args.load_path_qanet_s_e):
                    progress_bar.set_postfix(NLL=nll_meter_qanet_s_e.avg)
                else:
                    progress_bar.set_postfix(NLL=nll_meter_baseline.avg)

            idx2pred, uuid2pred = util.convert_tokens(gold_dict,
                                                      ids.tolist(),
                                                      starts.tolist(),
                                                      ends.tolist(),
                                                      args.use_squad_v2)
            pred_dict.update(idx2pred)
            sub_dict.update(uuid2pred)
    
    
    
    if(args.save_probabilities):
        if(args.load_path_baseline):
            with open(args.save_dir+"/probs_start", "wb") as fp:   #Pickling
                pickle.dump(save_prob_baseline_start, fp)
            with open(args.save_dir+"/probs_end", "wb") as fp:   #Pickling
                pickle.dump(save_prob_baseline_end, fp)    
                
        if(args.load_path_bidaf):
            with open(args.save_dir+"/probs_start", "wb") as fp:   #Pickling
                pickle.dump(save_prob_bidaf_start, fp)
            with open(args.save_dir+"/probs_end", "wb") as fp:   #Pickling
                pickle.dump(save_prob_bidaf_end, fp) 
            
        if(args.load_path_bidaf_fusion):
            with open(args.save_dir+"/probs_start", "wb") as fp:   #Pickling
                pickle.dump(save_prob_bidaf_fu_start, fp)
            with open(args.save_dir+"/probs_end", "wb") as fp:   #Pickling
                pickle.dump(save_prob_bidaf_fu_end, fp)    
                
        if(args.load_path_qanet):
            with open(args.save_dir+"/probs_start", "wb") as fp:   #Pickling
                pickle.dump(save_prob_qanet_start, fp)
            with open(args.save_dir+"/probs_end", "wb") as fp:   #Pickling
                pickle.dump(save_prob_qanet_end, fp) 
        
        if(args.load_path_qanet_old):
            with open(args.save_dir+"/probs_start", "wb") as fp:   #Pickling
                pickle.dump(save_prob_qanet_old_start, fp)
            with open(args.save_dir+"/probs_end", "wb") as fp:   #Pickling
                pickle.dump(save_prob_qanet_old_end, fp) 
        
        if(args.load_path_qanet_inde):
            with open(args.save_dir+"/probs_start", "wb") as fp:   #Pickling
                pickle.dump(save_prob_qanet_inde_start, fp)
            with open(args.save_dir+"/probs_end", "wb") as fp:   #Pickling
                pickle.dump(save_prob_qanet_inde_end, fp) 
       
        if(args.load_path_qanet_s_e):
            with open(args.save_dir+"/probs_start", "wb") as fp:   #Pickling
                pickle.dump(save_prob_qanet_s_e_start, fp)
            with open(args.save_dir+"/probs_end", "wb") as fp:   #Pickling
                pickle.dump(save_prob_qanet_s_e_end, fp) 
        
    # Log results (except for test set, since it does not come with labels)
    if args.split != 'test':
        results = util.eval_dicts(gold_dict, pred_dict, args.use_squad_v2)
        if(args.load_path_qanet):
            meter_avg=nll_meter_qanet.avg
        elif(args.load_path_bidaf):
            meter_avg=nll_meter_bidaf.avg
        elif(args.load_path_bidaf_fusion):
            meter_avg=nll_meter_bidaf_fu.avg
        elif(args.load_path_qanet_inde):
            meter_avg=nll_meter_qanet_inde.avg
        elif(args.load_path_qanet_s_e):
            meter_avg=nll_meter_qanet_s_e.avg
        elif(args.load_path_qanet_old):
            meter_avg=nll_meter_qanet_old.avg
        else:
            meter_avg=nll_meter_baseline.avg
        results_list = [('NLL', meter_avg),
                        ('F1', results['F1']),
                        ('EM', results['EM'])]
        if args.use_squad_v2:
            results_list.append(('AvNA', results['AvNA']))
        results = OrderedDict(results_list)

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        log.info(f'{args.split.title()} {results_str}')

        # Log to TensorBoard
        tbx = SummaryWriter(args.save_dir)
        util.visualize(tbx,
                       pred_dict=pred_dict,
                       eval_path=eval_file,
                       step=0,
                       split=args.split,
                       num_visuals=args.num_visuals)

    # Write submission file
    sub_path = join(args.save_dir, args.split + '_' + args.sub_file)
    log.info(f'Writing submission file to {sub_path}...')
    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for uuid in sorted(sub_dict):
            csv_writer.writerow([uuid, sub_dict[uuid]])


if __name__ == '__main__':
    main(get_test_args())
