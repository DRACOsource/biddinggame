# -*- coding: utf-8 -*-

import os
import torch
torch.set_num_threads(1)

ROOT_DIR         = os.path.normpath(os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), '../..'))
LOGS_DIR         = os.path.join(ROOT_DIR, 'logs')
RESOURCES_DIR    = os.path.join(ROOT_DIR, 'resources')
MODEL_DIR        = os.path.join(ROOT_DIR, 'models')
GRAPH_DIR        = os.path.join(ROOT_DIR, 'graphs')
PSTATS_FILE      = os.path.join(LOGS_DIR, 'profile.stats')
COUNTDOWN_FILE   = os.path.join(MODEL_DIR, 'countdownHistory')
BATCHCREATE_FILE = os.path.join(MODEL_DIR, 'batchCreateHistory')

DEVICE = torch.device('cpu')
DTYPE = torch.FloatTensor

class FILES():
    PERF_FILE =('perfile',os.path.join(LOGS_DIR,'performance'))
    PLM_FILE = ('plmfile',os.path.join(LOGS_DIR,'priceLearningModel'))
    BID_FILE = ('bidfile',os.path.join(LOGS_DIR,'finishedBids'))
    SL_FILE = ('supervisefile',os.path.join(LOGS_DIR,'supervisedLearningModel'))
    INVMDL_FILE = ('invmodelfile',os.path.join(LOGS_DIR,'invModel'))
    FWDMDL_FILE = ('forwardmodelfile',os.path.join(LOGS_DIR,'forwardModel'))
    REWARD_FILE = ('rewardfile',os.path.join(LOGS_DIR,'reward'))

class BIDDER_PARAMS():
    biddercost = (-1,-1)                # cost per time step
    priceFactor = 10                         # max bidding price per time step
                                            # as factor of biddercosts
    initSavings = 20
    competitorDataThres = 0                 # Nr bids to collect before 
                                            # creating transitionTbl
    plmTrainingThres = 3                    # Interval (in terms of new 
                                            # priceLearningModel inputs)
    stagingThreshold = [0.1]                # An indicator of higher than 
                                            # threshold will be submitted.
    stagingPeriod = 5                       # Max duration to bid.
    curiosityTrainingThres = 3              #
    extRewardInterval = 1             # external reward interval
    minValue = 0.01                         # minimum positive number
    loseGameCost = -10
                                            
class MDL_PARAMS():
    initBidders = 6                 # Initial number of bidders to create.
                                    # When lam==0, no new vehicles will be 
                                    # created after initialization, to give
                                    # the vehicles the chance to learn.
    recent = 350                    # Performance indicator for only
                                    # the recent periods
    auctioncost = -5                # cost to join the auction (if lost)
    priceWeight = 2.3               # weight of low price in sortbid

class PRICE_MODEL_PARAMS():
    maxReward = 100                 # to normalize reward for RL
    maxExtReward = 1                # to normalize reward for attention
    evaluation_start = 1000000
    batch_size = 30
    history_record = 62             # total size of input, need to be longer
                                    # than twice the length of batch_size
    epoch = 1
    epoch_supervise = 5
    train_all_records = 62          # Before this nr. inputs, train on all 
                                    # records. after this, train on the 
                                    # most recent history_record number of
                                    # records every 
                                    # VEHICLE_PARAMS.plmTrainingThres
    
    critic_type = 'ConvCritic'      # 'ConvCritic' or 'Critic' class
    critic_num_filter = 32
    critic_hidden_size1 = 128
    critic_hidden_size2 = 128
    critic_lr_min = 0.1
    critic_lr_reduce_rate = 0.99
    critic_learning_rate = 0.9
    critic_dropout_rate = 0.0
    reward_rate = 0.99              # Continuing tasks with function estimator 
                                    # should not use discount. 
                                    # Use average reward instead.
    reward_min = 0.01
    reward_reduce_rate = 1
    critic_pretrain_nr_record = 32 # no training until this nr. inputs
    
    actor_type = 'ConvActor'        # 'ConvActor' or 'Actor' class
    actor_num_filter = 64
    actor_hidden_size1 = 128
    actor_hidden_size2 = 128
    actor_lr_min = 0.1
    actor_lr_reduce_rate = 0.99
    actor_learning_rate = 0.9
    actor_dropout_rate = 0.0
    actor_pretrain_nr_record = 32  # No training until this nr. inputs.
    sharedlayer_output_dim = 128    # If no compression in sharedLayers, 
                                   # set the value to -1.
    
    add_randomness = 0              # Valued between 0 and 1. if greater 
                                    # than zero, then in inference function,
                                    # action is randomly chosen
                                    # when generated random number is smaller
                                    # than add_randomness * learning rate
    exploration = 128                # Before the model accumulated this 
                                    # number of records, the 
                                    # learning rate does not reduce.
    supervise_learning_rate = 0.1   # learning rate for supervised learning 
    supervise_hidden_size1 = 64     # MLP hidden layer 
    supervise_hidden_size2 = 128    # MLP hidden layer 
    fsp_rl_weight_constant = exploration * 2
    
    reward_ext_weight = 0.5 # Balance between ext. and int. reward. 
                            # Higher weight means more important ex. reward.
    reward_int_weight = 0.5 # Balance between utility and curiosity prediction 
                            # loss in intrinsic reward.
                            # Higher weight means more important utility.
    
class CURIOUS_MODEL_PARAMS():

    feature_hidden_size1 = 64
    feature_hidden_size2 = 128
    feature_outputDim = 32
    
    inv_hidden_size1 = 64
    inv_hidden_size2 = 128
    inv_learning_rate = 0.1

    forward_hidden_size1 = 64
    forward_hidden_size2 = 128
    forward_learning_rate = 0.1
    
    batch_size = 30
    epoch = 5
    
    invLoss_weight = 0.5