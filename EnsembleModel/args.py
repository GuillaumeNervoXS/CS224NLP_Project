"""Command-line arguments for setup.py, train.py, test.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import argparse


def get_setup_args():
    """Get arguments needed in setup.py."""
    parser = argparse.ArgumentParser('Download and pre-process SQuAD')

    add_common_args(parser)

    parser.add_argument('--train_url',
                        type=str,
                        default='https://github.com/chrischute/squad/data/train-v2.0.json')
    parser.add_argument('--dev_url',
                        type=str,
                        default='https://github.com/chrischute/squad/data/dev-v2.0.json')
    parser.add_argument('--test_url',
                        type=str,
                        default='https://github.com/chrischute/squad/data/test-v2.0.json')
    parser.add_argument('--glove_url',
                        type=str,
                        default='http://nlp.stanford.edu/data/glove.840B.300d.zip')
    parser.add_argument('--dev_meta_file',
                        type=str,
                        default='../data/dev_meta.json')
    parser.add_argument('--test_meta_file',
                        type=str,
                        default='../data/test_meta.json')
    parser.add_argument('--word2idx_file',
                        type=str,
                        default='../data/word2idx.json')
    parser.add_argument('--char2idx_file',
                        type=str,
                        default='../data/char2idx.json')
    parser.add_argument('--answer_file',
                        type=str,
                        default='../data/answer.json')
    parser.add_argument('--para_limit',
                        type=int,
                        default=400,
                        help='Max number of words in a paragraph')
    parser.add_argument('--ques_limit',
                        type=int,
                        default=50,
                        help='Max number of words to keep from a question')
    parser.add_argument('--test_para_limit',
                        type=int,
                        default=1000,
                        help='Max number of words in a paragraph at test time')
    parser.add_argument('--test_ques_limit',
                        type=int,
                        default=100,
                        help='Max number of words in a question at test time')
    parser.add_argument('--char_dim',
                        type=int,
                        default=64,
                        help='Size of char vectors (char-level embeddings)')
    parser.add_argument('--glove_dim',
                        type=int,
                        default=300,
                        help='Size of GloVe word vectors to use')
    parser.add_argument('--glove_num_vecs',
                        type=int,
                        default=2196017,
                        help='Number of GloVe vectors')
    parser.add_argument('--ans_limit',
                        type=int,
                        default=30,
                        help='Max number of words in a training example answer')
    parser.add_argument('--char_limit',
                        type=int,
                        default=16,
                        help='Max number of chars to keep from a word')
    parser.add_argument('--include_test_examples',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Process examples from the test set')

    args = parser.parse_args()

    return args


def get_train_args():
    """Get arguments needed in train.py."""
    parser = argparse.ArgumentParser('Train a model on SQuAD')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--model',
                        type=str,
                        default='qanet',
                        choices=('baseline', 'bidaf', 'qanet','qanet_out'),
                        help='Which type of model you want to train')
    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load a model checkpoint.')
    parser.add_argument('--eval_steps',
                        type=int,
                        default=50000,
                        help='Number of steps between successive evaluations.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.5,
                        help='Learning rate.')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=0,
                        help='L2 weight decay.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs for which to train. Negative means forever.')
    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.2,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--metric_name',
                        type=str,
                        default='F1',
                        choices=('NLL', 'EM', 'F1'),
                        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')
    parser.add_argument('--seed',
                        type=int,
                        default=224,
                        help='Random seed for reproducibility.')
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help='Decay rate for exponential moving average of parameters.')
    parser.add_argument('--beta_1', 
                        type=float,
                        default=0.8, 
                        help='Beta_1 for the Adam Optimizer.')
    parser.add_argument('--beta_2',
                        type=float,
                        default=0.999,
                        help='Beta_2 for the Adam Optimizer.')
    parser.add_argument('--epsilon',
                        type=float,
                        default=10e-7,
                        help='Epsilon for the Adam Optimizer.')
    
    args = parser.parse_args()

    if args.metric_name == 'NLL':
        # Best checkpoint is the one that minimizes negative log-likelihood
        args.maximize_metric = False
    elif args.metric_name in ('EM', 'F1'):
        # Best checkpoint is the one that maximizes EM or F1
        args.maximize_metric = True
    else:
        raise ValueError(f'Unrecognized metric name: "{args.metric_name}"')

    return args


def get_test_args():
    """Get arguments needed in test.py."""
    parser = argparse.ArgumentParser('Test a trained model on SQuAD')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--split',
                        type=str,
                        default='dev',
                        choices=('train', 'dev', 'test'),
                        help='Split to use for testing.')
    parser.add_argument('--sub_file',
                        type=str,
                        default='submission.csv',
                        help='Name for submission file.')
    parser.add_argument('--load_path_baseline',
                        type=str,
                        default=None,
                        help='Path to load baseline as a model checkpoint.')
    parser.add_argument('--load_path_bidaf',
                        type=str,
                        default=None,
                        help='Path to load BiDAF as a model checkpoint.')
    parser.add_argument('--load_path_bidaf_fusion',
                        type=str,
                        default=None,
                        help='Path to load BiDAF-fusion as a model checkpoint.')
    
    parser.add_argument('--load_path_qanet',
                        type=str,
                        default=None,
                        help='Path to load QANet as a model checkpoint.')
    parser.add_argument('--load_path_qanet_old',
                        type=str,
                        default=None,
                        help='Path to load the old version QANet as a model checkpoint.')

    parser.add_argument('--load_path_qanet_inde',
                        type=str,
                        default=None,
                        help='Path to load QANet with the 3 independant decoder as a model checkpoint.')
    parser.add_argument('--load_path_qanet_s_e',
                        type=str,
                        default=None,
                        help='Path to load QANet, where we feed the start probs as in\
                              as input to the end output path, a model checkpoint.')
    parser.add_argument('--save_probabilities',
                        type=bool,
                        default=False,
                        help='Whether to save the probabilities p_start and p_end')
    

    # Require load_path for test.py
    args = parser.parse_args()
    """
    if not (args.load_path_baseline or args.load_path_bidaf or args.load_path_bidaf_fusion 
              or args.load_path_qanet or args.load_path_qanet_inde or args.load_path_qanet_s_e
              or args.load_path_qanet_old):
        raise argparse.ArgumentError('Missing required argument --load_path_{model}')
    """
    return args


def add_common_args(parser):
    """Add arguments common to all 3 scripts: setup.py, train.py, test.py"""
    parser.add_argument('--train_record_file',
                        type=str,
                        default='../data/train.npz')
    parser.add_argument('--dev_record_file',
                        type=str,
                        default='../data/dev.npz')
    parser.add_argument('--test_record_file',
                        type=str,
                        default='../data/test.npz')
    parser.add_argument('--word_emb_file',
                        type=str,
                        default='../data/word_emb.json')
    parser.add_argument('--char_emb_file',
                        type=str,
                        default='../data/char_emb.json')
    parser.add_argument('--train_eval_file',
                        type=str,
                        default='../data/train_eval.json')
    parser.add_argument('--dev_eval_file',
                        type=str,
                        default='../data/dev_eval.json')
    parser.add_argument('--test_eval_file',
                        type=str,
                        default='../data/test_eval.json')


def add_train_test_args(parser):
    """Add arguments common to train.py and test.py"""
    parser.add_argument('--name',
                        '-n',
                        type=str,
                        #required=True,
                        help='Name to identify training or test run.')
    parser.add_argument('--max_ans_len',
                        type=int,
                        default=15,
                        help='Maximum length of a predicted answer.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of sub-processes to use per data loader.')
    parser.add_argument('--save_dir',
                        type=str,
                        default='../save/',
                        help='Base directory for saving information.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size per GPU. Scales automatically when \
                              multiple GPUs are available.')
    parser.add_argument('--use_squad_v2',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether to use SQuAD 2.0 (unanswerable) questions.')
    parser.add_argument('--char_emb_dim',
                        type=int,
                        default=200,
                        help='Dimension char embeddings')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=128,
                        help='Number of features in encoder hidden layers.')
    parser.add_argument('--num_visuals',
                        type=int,
                        default=10,
                        help='Number of examples to visualize in TensorBoard.')
    parser.add_argument('--n_emb_blocks',
                        type=int,
                        default=1,
                        help='Number of embedding encoder blocks.')
    parser.add_argument('--n_mod_blocks',
                        type=int,
                        default=4,
                        help='Number of model encoder blocks.')
    parser.add_argument('--n_heads',
                        type=int,
                        default=4,
                        help='Number of heads.')
    parser.add_argument('--n_conv_emb',
                        type=int,
                        default=4,
                        help='Number of convolutional layers in the embedding encoder blocks.')
    parser.add_argument('--n_conv_mod',
                        type=int,
                        default=2,
                        help='Number of convolutional layers in the modeling encoder blocks.')
    parser.add_argument('--divisor_dim_kqv',
                        type=int,
                        default=2,
                        help='Divisor of the hidden_size to represent queries,keys and values')