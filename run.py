# -*- coding: utf-8 -*-
from biddinggame.config.config import LOGS_DIR
from biddinggame.utils.common_utils import openFiles,closeFiles
from biddinggame.solutions.biddinggame import BiddingModel
import os,sys,re
os.chdir(os.path.dirname(os.path.realpath(__file__)))

if __name__ == '__main__':  
    path = LOGS_DIR
    train = True
    evaluation = False
    loadModel = True
    curious = True
    ca = True # use attention if True
    nrEpisodes = 1
    steps = 1000000
    interval = 150
    isFirstPrice = True # second price if False
    fairness = False # if to use fairness index as external signal
    if 'interval' in sys.argv[1]:
        location = re.search('=',sys.argv[1]).span()[1]
        try:
            end = re.search('_',sys.argv[1]).span()[1]
            interval = int(sys.argv[1][location:end-1])
        except:
            interval = int(sys.argv[1][location:])
    if 'eval' in sys.argv[1]:
        evaluation = True
    if 'secondPrice' in sys.argv[1]:
        isFirstPrice = False
    if 'fair' in sys.argv[1]:
        fairness = True

    for i in range(nrEpisodes):
        filedict = openFiles(additional=[interval,str(train)],
                                 trainable=(train and not evaluation))
        mdl = BiddingModel(filedict,train=train,evaluation=evaluation,
                loadModel=loadModel,curious=curious,extRewardInterval=interval,
                cumstep=i*steps,endOfGame=steps,ca=ca,
                isFirstPrice=isFirstPrice,fairness=fairness)
        for j in range(steps):
            mdl.step()
            
            if j%100==0:
                actor_learning_rate = []
                for v in mdl.bidders.values():
                    actor_learning_rate.append(v.priceMdl.actor_learning_rate)
                print('actor learning rate: {},{}.'.format(
                     min(actor_learning_rate),max(actor_learning_rate)))
        
        closeFiles(filedict)
    



