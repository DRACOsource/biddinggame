# -*- coding: utf-8 -*-
from ..config.config import (BIDDER_PARAMS as vp,MDL_PARAMS as mp,
                             PRICE_MODEL_PARAMS as pmp,
                             FILES,MODEL_DIR)
from ..solutions.price_model_curious import (PriceLearningModel as plm, 
                SupervisedModel as spv, Curiosity as cur)
from ..utils.name_utils import ColumnHead as ch

import numpy as np
import os,sys
from scipy.stats import truncnorm
from mesa import Agent, Model
from mesa.time import SimultaneousActivation
import weakref
import pickle
        
class Broker(Agent):
    '''
    '''
    auctioncost = mp.auctioncost
    priceWeight = mp.priceWeight
    _instances = set()
    def __init__(self,uniqueId,model,isFirstPrice=True):
        super().__init__(uniqueId, model)
        self.unique_id = uniqueId
        self.bids = list() # ids of submitted bids
        self.activeBids = list() # ids of active bids 
        self.winnerPrice = 0
        self.isFirstPrice = isFirstPrice # if False, then second price auction
        self.winnerDuration = 0
        self.secondPrice = -1
        self.nrSubmittedBids = 0 # nr of submitted bids
        self.nrActiveBids = 0 # nr of active bids
        self.paymentHistory = {}
        self._instances.add(weakref.ref(self))

    def _update_winner(self,bid):
        return (bid.price,bid.duration,bid.unique_id)

    def _sortBids_secondPrice(self):
        '''return bid with highest price and fixed duration'''
        self.secondPrice = -1
        bids = [self.model.bidlist[x] for x in self.bids]
        if len(bids)==0:
            return
        highestPrice,fixedDuration,winner = self._update_winner(bids[0])
        maxPriceDuration =  highestPrice
        if len(bids)==1:
            self.secondPrice = maxPriceDuration
            return self.bids[0]        
        for bid in bids[1:]:
            newPriceDuration = bid.price
            if newPriceDuration<maxPriceDuration:
                if self.secondPrice<0 or newPriceDuration>self.secondPrice:
                    self.secondPrice = newPriceDuration
                continue
            if newPriceDuration==maxPriceDuration:
                randomNr = np.random.random()
                if randomNr<0.5: # break ties randomly
                    self.secondPrice=maxPriceDuration
                    continue
            else:
                self.secondPrice = maxPriceDuration
                maxPriceDuration = newPriceDuration
            winner = bid.unique_id
        return winner
                
    def _sortBids_firstPrice(self):
        '''return bid with lowest price and longest duration'''
        bids = [self.model.bidlist[x] for x in self.bids]
        if len(bids)==0:
            return
        lowestPrice,longestDuration,winner = self._update_winner(bids[0])
        if len(bids)==1:
            return self.bids[0]
        for bid in bids[1:]:
            if bid.price>=lowestPrice and bid.duration<=longestDuration:
                continue
            if bid.price<=lowestPrice and bid.duration>=longestDuration:
                if bid.price==lowestPrice and bid.duration==longestDuration:
                    randomNr = np.random.random()
                    if randomNr<0.5: # break ties randomly
                        continue
                lowestPrice,longestDuration,winner = self._update_winner(bid)
                continue
            x = min(lowestPrice,bid.price) / max(lowestPrice,bid.price)
            y = min(longestDuration,bid.duration) / max(longestDuration,
                                                               bid.duration)
            opcost = (1-x)*y-(1-y)*self.priceWeight
            if ((opcost<0 and bid.duration>longestDuration) 
                    or (opcost>0 and bid.price<lowestPrice)):
                lowestPrice,longestDuration,winner = self._update_winner(bid)
        return winner
    
    def _calculatePayment(self,bid,result='lost'):
        ''' return fixed cost to lost bids and payoff to won bid. 
            if first price bidding game, payoff = price * duration, 
            broker pays seller/bidder. if second price bidding game, 
            buyer/bidder pays second price to broker and function returns 
            payoff = (first price - second price) * duration
        '''
        if result=='lost':
            return self.auctioncost
        else:
            if self.isFirstPrice:
                return bid.price * bid.duration
            else:
                if self.secondPrice<0:
                    self.secondPrice = bid.price * bid.duration
                return (bid.price-self.secondPrice) * bid.duration

    def _recordPayment(self,userId,bidPayment):
        if userId not in self.paymentHistory.keys():
            self.paymentHistory[userId] = [
                    (self.model.schedule.steps,bidPayment)]
        else:
            threshold = self.model.schedule.steps-self.model.extRewardInterval
            records = self.paymentHistory[userId]
            records = [(s,p) for (s,p) in records if s>=threshold]
            records.append((self.model.schedule.steps,bidPayment))
            self.paymentHistory[userId] = records

    def _rejectBid(self,bid):
        ''' rejected bids '''
        bid.payment = self._calculatePayment(bid,'lost')
        self.model.bidders[bid.user].removeBid(bid.unique_id)

    def _activateBid(self,bid):
        ''' accept bid '''
        bid.payment = self._calculatePayment(bid,'won')
        self._recordPayment(bid.user,bid.payment)
        self.activeBids.append(bid.unique_id)
        self.model.bidders[bid.user].activateBid(bid.unique_id)
        
    def getFairnessIndex(self):
        num = 0
        denom = 0
        for k,v in self.paymentHistory.items():
            x = sum([p for (s,p) in v])
            num += x
            denom += x**2
        if denom==0:
            return 0.5
        num = num**2/self.model.initBidders
        return num/denom
        
    def deactivateBid(self,bidId):
        ''' finished bid '''
        self.activeBids.remove(bidId)
    
    def getFeedback(self):
        return (self.nrSubmittedBids,self.nrActiveBids,self.winnerPrice,
                self.winnerDuration)
            
    def step(self):
        return
            
    def estimate(self):
        pass
    
    def allocate(self):
        self.nrSubmittedBids = len(self.bids)
        self.nrActiveBids = len(self.activeBids)
        
        bidfile = self.model.filedict[FILES.BID_FILE[0]]
        if self.isFirstPrice:
            winner = self._sortBids_firstPrice()
        else:
            winner = self._sortBids_secondPrice()
        if winner is None:
            return
        
        for bidId in self.bids:
            bid = self.model.bidlist[bidId]

            if bid.unique_id==winner:
                self.winnerPrice = bid.price
                self.winnerDuration = bid.duration
                self._activateBid(bid)
            else:
                self._rejectBid(bid)
            self.bids.remove(bidId)
            
            bidfile.write(
            '{};{};{};{};{};{};{};{};{};{};{};{};{}\n'.format(
            self.model.schedule.steps+self.model.cumstep,
            self.nrSubmittedBids,self.nrActiveBids,bid.user,
            bidId,bid.price,bid.payment,bid.duration,bid.biddercost,
            bid.batchCreateTime,bid.createTime,bid.isSuccess,
            self.model.extRewardInterval))
        
    def advance(self):
        pass
    
        
    @classmethod
    def getInstances(cls):
        nonexistent = set()
        for ref in cls._instances:
            obj = ref()
            if obj is not None:
                yield obj
            else:
                nonexistent.add(ref)
        cls._instances -= nonexistent
        

class Bidder(Agent):
    _instances = set()
    
    competitorDataThres = vp.competitorDataThres
    plmTrainingThres = vp.plmTrainingThres
    stagingThreshold = np.random.choice(vp.stagingThreshold)
    stagingPeriod = vp.stagingPeriod
    priceFactor = vp.priceFactor 
    loseGameCost = vp.loseGameCost
    dimOutput = 2
    def __init__(self,uniqueId,brokerId,createTime,model,biddercost=None,
                 bidderinitsavings=None,curiosity=True,ca=True,
                 extRewardInterval=1,curiosityTrainingThres=None):
        super().__init__(uniqueId, model)
        self.unique_id = uniqueId
        self.curiosity = curiosity
        self.ca = ca
        self.extRewardInterval = extRewardInterval
        if curiosityTrainingThres is None:
            self.curiosityTrainingThres = vp.curiosityTrainingThres
        else:
            self.curiosityTrainingThres = curiosityTrainingThres
        if biddercost is None:
            self.biddercost = np.random.choice([vp.biddercost[0],
                                                 vp.biddercost[1]])
            self.initSavings = vp.initSavings
        else:
            self.biddercost = biddercost
            self.initSavings = bidderinitsavings
        self.maxSavings = self.initSavings + (
                            self.priceFactor * (-self.biddercost)
                            * self.extRewardInterval
                            + self.biddercost * self.extRewardInterval)
        self.currentSavings = self.initSavings
#        self.savingsHistory = [(0,self.currentSavings)]
        self.auctioncost = 0 # cost of joining auction if lost
        self.auctionpayment = 0 # payment if won
        self.createTime = createTime
        self.bidCount = 0 # for generating unique ids for the bids
        self.activeBid = list() # bids which are active
        # temporary list to record rewards in self._activateBids
        self.bidHistory = list() # history of finished bids
        self.broker = brokerId
        self.readyToBid = 1
        self.readyToCollectInfo = 0
        self.stepcounter = 0 # for ext. reward
        self.episodeSteps = 0 # for ext. reward
        # for normalization of environment variable (total nr.bids)
        self.benchmarkNrBid = mp.initBidders
        
        # output of the price learning model: prediction of best response
        # 1st digit is whether to 
        # activate bid in the current time unit. rest of output is 
        # proportion of allocated budget to each bid.
        self.priceMdl = plm(uniqueId=self.unique_id,
                            dimOutput=self.dimOutput,
                            evaluation=self.model.evaluation,
                            loadModel=self.model.loadModel,
                            curiosity=self.curiosity,
                            cumstep=self.model.cumstep,
                            endOfGame=self.model.endOfGame,
                            ca=self.ca,#maxReward=self.maxSavings)
                            maxReward=None,fairness=self.model.fairness)
        
        # output of the supervised learning: prediction of own behavior
        self.superviseMdl = spv(uniqueId=self.unique_id,
                                dimOutput=self.dimOutput,
                                evaluation=self.model.evaluation,
                                loadModel=self.model.loadModel,
                                curiosity=self.curiosity)
        # connection key to link input records to rewards and outputs
        self.learningDataPos = str(0)
        self.learningDataPosSL = str(0) # for supervised learning model
        self.prevRecordNr = 0 # used to determine if to re-train the priceMdl.

        self.curiosityMdl = None
        self.prevRecordNr_curiosity = sys.maxsize
        if self.curiosity:
            self.curiosityMdl = cur(uniqueId=self.unique_id,
                            dimAction=self.dimOutput,
                            evaluation=self.model.evaluation,
                            loadModel=self.model.loadModel,
                            #maxReward=self.maxSavings)
                            maxReward=None)
            self.prevRecordNr_curiosity = 0 # to re-train curiosity model.  
        self.extRandom = int(np.random.rand()*self.extRewardInterval)
        self.extRewardSignal = False # train attention model when True
        self.extSignal = 0
        self._instances.add(weakref.ref(self))
    
    def _fsp(self,bestResponse,behavior,const=None):
        ''' choose best response with probability '''
        if const is None:
            const = pmp.fsp_rl_weight_constant
        mixingParam = const / (self.model.schedule.steps + const)
        randomNr = np.random.rand()
        if randomNr <= mixingParam:
            return bestResponse
        else:
            return behavior

    def _getAction(self):
        lower = 0
        upper = 1
        mu = self.stagingThreshold
        sigma = max(mu-lower,upper-mu)/2
        return truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,
                             loc=mu,scale=sigma)

    def _collectInfo(self):
        bidVec = [self.currentSavings,self.biddercost]
        (nrSubmittedBids,nrActiveBids,winnerPrice,
         winnerDuration) = self.model.broker[self.broker].getFeedback()
        observation = [winnerPrice/(self.priceFactor*-self.biddercost),
                  winnerDuration/self.stagingPeriod]
        envVec = [nrSubmittedBids/self.model.initBidders,
                  nrActiveBids/self.model.initBidders]
        return bidVec, envVec, observation

    def _submitBidOrWait(self):
        '''
        '''
        bidVec, envVec, observation = self._collectInfo()
        # for supervised learning
        inputVecSL = bidVec + envVec
        # RL
        inputVec = inputVecSL + observation            
        self.learningDataPos = self.priceMdl.collectInput(inputVec)
        if self.model.evaluation:
            randomness = 0
        else:
            randomness = None
        bestResp = self.priceMdl.inference(inputVec=inputVec,
                            randomness=randomness).tolist()
            
        # average behavior record for supervised learning
        self.learningDataPosSL = self.superviseMdl.collectInput(
                inputVecSL,model='supervisedModel')
        behavior = self.superviseMdl.inference(inputVecSL,
                            pmdlReward=self.priceMdl.reward).tolist()
        result = self._fsp(bestResp,behavior)
        if result[1]==0:
            result[1] = vp.minValue
            
        # best response behavior record for supervised learning
        self.superviseMdl.collectOutput(result,self.learningDataPosSL)
        self.learningDataPosSL = self.superviseMdl.collectInput(
                inputVecSL,model='supervisedModel')
        self.superviseMdl.collectOutput(bestResp,self.learningDataPosSL)

        self.priceMdl.collectOutput(result,self.learningDataPos)
        self.priceMdl.collectNextState(inputVec,self.learningDataPos)
        
        if self.model.trainable:
            act = result[0]
        else: 
            act = np.random.uniform()
        stagingRandom = self._getAction()
        if act>=stagingRandom:
            # submit bid:
            self.readyToBid = 0
            self.readyToCollectInfo = 1 # collect next state information
            if self.model.isFirstPrice:
                duration = int(result[0]*self.stagingPeriod)
            else:
                duration = self.stagingPeriod                
            newbid = Bid(bidId=self.unique_id+'_'+str(self.bidCount),
                         user=self.unique_id,cost=self.biddercost,
                         createTime=self.model.schedule.steps,
                         model=self.model,
                         price=result[1]*self.priceFactor*(-self.biddercost),
                         duration=duration,
                         priceLearningDataPos=self.learningDataPos)
            self.bidCount += 1
            self.model.schedule.add(newbid)
            self.model.bidlist[newbid.unique_id] = newbid
            self.model.broker[self.broker].bids.append(newbid.unique_id)
        else:
            # wait:
            self.priceMdl.collectReward(self.biddercost,self.learningDataPos)

    def _trainModels(self,signal):
        self.auctionpayment = 0 # for output
        self.auctioncost = 0 # for output
        self.gameoverReward = 0 # for output
        self.episodeSteps = 0 # for output
        self.stepcounter += 1
        self.priceMdl.trainWeightvector = (
                (int(self.learningDataPos)-self.prevRecordNr
                 >=self.plmTrainingThres),False)
        reward_add = self.currentSavings - vp.initSavings
        if (self.currentSavings<=0 or 
            self.stepcounter>self.extRewardInterval): # game over
            if self.model.fairness:
                self.priceMdl.collectReward(reward=reward_add,
                                idx=self.learningDataPos,rewardType='in')            
            else:
                if not self.ca:
                    self.priceMdl.collectReward(reward=reward_add,
                                    idx=self.learningDataPos,rewardType='in')
            self.gameoverReward = reward_add
            self.episodeSteps = self.stepcounter # for output
            # reset budget pool until the next extRewardInterval
            self.currentSavings = self.initSavings
            
        if self.extRewardSignal: # trigger attention model learning
            self.priceMdl.collectReward(reward=signal,
                                    idx=self.learningDataPos,rewardType='ex')
            self.gameoverReward = reward_add
            self.episodeSteps = self.stepcounter
            self.priceMdl.trainWeightvector = (True,True)

        if self.stepcounter>self.extRewardInterval: # restart game
            self.stepcounter = 0

        if ((not self.model.trainable) or (self.model.evaluation)
            or (self.model.schedule.steps>=pmp.evaluation_start)):
            return

        trainPlm = ((int(self.learningDataPos)-self.prevRecordNr
           >=self.plmTrainingThres) or self.priceMdl.trainWeightvector[0])
        trainCur = (int(self.learningDataPos)-self.prevRecordNr_curiosity
                                            >=self.curiosityTrainingThres)
        if trainCur:
            self.priceMdl.trainingdata = self.priceMdl.prep_data(
                            self.model.schedule.steps+self.model.cumstep,
                            self.model.filedict[FILES.FWDMDL_FILE[0]],
                            curious=self.curiosity,
                            model='curiosity')
            if self.priceMdl.actor is not None:
                _,_ = self.curiosityMdl.train(
                    self.model.schedule.steps+self.model.cumstep,
                    trainingdata=self.priceMdl.trainingdata,
                    invmodelfile=self.model.filedict[FILES.INVMDL_FILE[0]],
                    forwardmodelfile=self.model.filedict[FILES.FWDMDL_FILE[0]],
                    pretrain=True,sharedLayers=self.priceMdl.actor.sharedLayers)
                self.prevRecordNr_curiosity = int(self.learningDataPos)
            self.priceMdl.update_curiousReward(self.curiosityMdl.reward)
        if trainPlm:
            self.priceMdl.train(
                    time=self.model.schedule.steps+self.model.cumstep,
                    plmfile=self.model.filedict[FILES.PLM_FILE[0]],
                    rewardfile=self.model.filedict[FILES.REWARD_FILE[0]],
                    invfile=self.model.filedict[FILES.INVMDL_FILE[0]],
                    fwdfile=self.model.filedict[FILES.FWDMDL_FILE[0]],
                    curMdl=self.curiosityMdl,
                    target='maximize')
            self.superviseMdl.train(
                    self.model.schedule.steps+self.model.cumstep,
                    self.model.filedict[FILES.SL_FILE[0]],
                    pmdlReward=self.priceMdl.reward)
            
            self.prevRecordNr = int(self.learningDataPos)   

    def removeBid(self,bidId):
        self.readyToBid = 1
        bid = self.model.bidlist[bidId]
        # one time step of biddercost, plus cost to join the auction
        self.auctioncost = bid.payment
        self.priceMdl.collectReward(bid.biddercost+bid.payment,
                                            bid.priceLearningDataPos)
        self.currentSavings += bid.payment
        bid.remove()
    
    def activateBid(self,bidId):
        bid = self.model.bidlist[bidId]
        self.auctionpayment = bid.payment
        self.priceMdl.collectReward(bid.biddercost*bid.duration+bid.payment,
                                            bid.priceLearningDataPos)
        self.currentSavings += bid.payment
        bid.activate()
        self.activeBid.append(bidId)
    
    def deactivateBid(self,bidId):
        ''' deactivate won bid when finished '''
        self.activeBid.remove(bidId)
        self.readyToBid = 1
        self.model.broker[self.broker].deactivateBid(bidId)
        self.bidHistory.append(bidId) 
    
    def step(self):
        self.extRewardSignal = False
        signal = 0
        self.extSignal = 0
        if self.stepcounter>=self.extRewardInterval:
            self.extSignal = self.model.broker[self.broker].getFairnessIndex()
            
        if self.model.fairness:
            if self.stepcounter>=self.extRewardInterval:
                signal = self.model.broker[self.broker].getFairnessIndex()
                self.extRewardSignal = True            
        else: # cum. payoff as ext. signal only for attention model
            if (self.ca and (self.currentSavings<=0 or 
                             self.stepcounter>=self.extRewardInterval)):
                signal = self.currentSavings - vp.initSavings
                self.extRewardSignal = True
        self._trainModels(signal=signal)

    def estimate(self):
        if self.readyToBid==1:
            self._submitBidOrWait()
    
    def allocate(self):
        pass
    
    def advance(self):
        if self.readyToCollectInfo==1:
            # observation only if bid is submitted
            bidVec, envVec, observation = self._collectInfo()
            state = bidVec + envVec + observation
            self.priceMdl.collectNextState(state,self.learningDataPos)
            self.readyToCollectInfo=0
        self.currentSavings += self.biddercost
            
    @classmethod
    def getInstances(cls):
        nonexistent = set()
        for ref in cls._instances:
            obj = ref()
            if obj is not None:
                yield obj
            else:
                nonexistent.add(ref)
        cls._instances -= nonexistent


class Bid(Agent):
    def __init__(self,bidId,user,cost,createTime,model,price,duration,
                 priceLearningDataPos):
        self.unique_id = bidId
        super().__init__(self.unique_id, model)
        self.user = user # associated vehicle.
        self.biddercost = cost
        self.batchCreateTime = createTime # bid create time
        self.finishTime = -1 # time bid is done
        self.timeToGo = -1 # time left till finish
        self.createTime = createTime # bid start time
        self.price = price # unit bidding price
        self.duration = duration # time steps
        self.isSuccess = -1 # indicator if won
        self.isActive = 0 # indicator if still active (in queue or process)
        self.payment = -1 # payment
        self.environment = 0 # e.g. nr bids at broker at time of creation
        self.competitorEnv = 0 # e.g. nr bids at broker at time of allocation
        # to help find position for reward data in model
        self.priceLearningDataPos = priceLearningDataPos 
                                       # key to connect input, output and 
                                       # reward in PriceLearningModel

    def _deactivate(self):
        ''' if bid is accepted and finished '''
        self.isActive = 0
        self.model.bidders[self.user].deactivateBid(self.unique_id)
    
    def activate(self):
        ''' if bid is accepted '''
        self.isActive = 1
        self.isSuccess = 1
        self.createTime = self.model.schedule.steps
        self.finishTime = self.createTime + self.duration
        self.timeToGo = self.duration
    
    def remove(self):
        ''' if bid is rejected '''
        self.isSuccess = 0
        
    def step(self):
        # bid is being processed:
        if self.isActive and self.isSuccess:
            self.timeToGo -= 1
            if self.timeToGo<0:
                self._deactivate()

    def estimate(self):
        pass
    
    def allocate(self):
        pass
    
    def advance(self):
        pass        

class BiddingModel(Model):
    def __init__(self,filedict,initBidders=mp.initBidders,
                 train=True,evaluation=False,repeatable=True,
                 loadModel=False,curious=False,extRewardInterval=None,
                 cumstep=0,endOfGame=5000,ca=True,isFirstPrice=True,
                 fairness=False):
        self.totalNrBidders = 0 # for naming bidders
        self.initBidders = initBidders
        self.bidders = dict() # list of Bidder objects
        self.broker = dict() # list of brokers
        self.bidlist = dict() # list of Bid objects
        self.isFirstPrice = isFirstPrice # if False, then second price
        self.fairness = fairness # if true, use fairness as ext. reward
        self.filedict = filedict # all paths and names of output files
        self.trainable = train # if to run the training algorithms
        self.evaluation = evaluation # if only to run inference with already
                                     # trained model
        self.repeatable = repeatable # if create same bids with same interval
        self.loadModel = loadModel # if to use pre-trained models
        self.cumstep = cumstep
        self.endOfGame = endOfGame
        self.curiosity = curious
        self.ca = ca # if true: run attentionRNN module for credit assignment
        if extRewardInterval is None and self.curiosity:
            self.extRewardInterval = vp.extRewardInterval
        else:
            self.extRewardInterval = extRewardInterval # for curiosity model
        self.maxActive = initBidders # max. nr vehicles in graph at any time
        self.acceptedBid = dict() # record of nr. accepted bids
        self.rejectedBid = dict() # record of nr. rejected bids
        self._printHeader() # print all file headers
        # requires each agent to implement all methods 
        # step, estimate, allocate, advance:
        self.schedule = SimultaneousActivation(self)
        # create broker
        brokerAgent = Broker(uniqueId='broker',model=self,
                             isFirstPrice=isFirstPrice)
        self.broker[brokerAgent.unique_id] = brokerAgent
        self.schedule.add(brokerAgent)
        # create bidders
        self._createNewBidders(self.initBidders,brokerAgent.unique_id)

    def _prepNewBidder(self):
        self.addId = '_' + str(self.trainable) # for identifying different models
        self.bidderCostPath = os.path.join(MODEL_DIR,'biddercost.pkl')
        self.bidderInitSavingsPath = os.path.join(
                                        MODEL_DIR,'bidderinitsavings.pkl')
        self.saveBidderCosts = False
        self.saveBidderInitSavings = False
        try:
            self.biddercost = pickle.load(open(self.bidderCostPath,'rb'))
            self.bidderinitsavings = pickle.load(
                                        open(self.bidderInitSavingsPath,'rb'))
        except:
            self.biddercost = None
            self.bidderinitsavings = None
    
    def _createSingleNewBidder(self,biddercost,bidderinitsavings,brokerId): 
        curiosity = self.curiosity
        ca = self.ca
        extRewardInterval = self.extRewardInterval
        curiosityTrainingThres = None
        if self.totalNrBidders%3==0:# greedy algorithm with no curiosity model
            curiosity = False
            curiosityTrainingThres = sys.maxsize
            ca = False
        elif self.totalNrBidders%3==1: # curiosity model without attention
            ca = False
        agent = Bidder(
                uniqueId='bidder'+str(self.totalNrBidders)+self.addId,
                brokerId=brokerId,createTime=self.schedule.steps,model=self,
                biddercost=biddercost,bidderinitsavings=bidderinitsavings,
                curiosity=curiosity,ca=ca,extRewardInterval=extRewardInterval,
                curiosityTrainingThres=curiosityTrainingThres)
        self.bidders[agent.unique_id] = agent
        self.schedule.add(agent)
        self.totalNrBidders += 1      
        return agent.unique_id    
    
    def _createNewBidders(self,nrArrivals,brokerId):
        self._prepNewBidder()
        for i in range(nrArrivals):
            try:
                biddercost = self.biddercost[i]
                bidderinitsavings = self.bidderinitsavings[i]
            except:
                biddercost = None
                bidderinitsavings = None
            _ = self._createSingleNewBidder(biddercost=biddercost,
                                        bidderinitsavings=bidderinitsavings,
                                        brokerId=brokerId)
        
        if self.saveBidderCosts:
            biddercost = [x.biddercost for x in self.bidders.values()]
            pickle.dump(biddercost,open(self.bidderCostPath,'wb'))
    
    def _printHeader(self):
        perfile = self.filedict[FILES.PERF_FILE[0]]
        perfTitle = ch.STEP+';'+ch.VEHICLE+';'+ch.BIDDERCOST+';'
        perfTitle += ch.ACTIVEBID+';'+ch.BIDID+';'+ch.BIDPRICE+';'
        perfTitle += ch.AUCTIONPAYMENT+';'+ch.AUCTIONCOST+';'
        perfTitle += ch.INITSAVINGS+';'+ch.CURRENTSAVINGS+';'
        perfTitle += ch.GAMEOVERREWARD+';'+ch.EXTREWARDINTERVAL+';'
        perfTitle += ch.EPISODESTEPS+';'+ch.CURIOSITYMDL+';'
        perfTitle += ch.ATTENTIONMDL+';'+ch.EXTSIGNAL+';'+ch.FAIRNESS+'\n'
        if os.path.getsize(perfile.name) == 0:
            perfile.write(perfTitle)

        bidfile = self.filedict[FILES.BID_FILE[0]]
        bidTitle = ch.STEP+';'+ch.NRSUBMITTEDBIDS+';'+ch.NRACTIVEBIDS+';'
        bidTitle += ch.VEHICLE+';'+ch.BIDID+';'+ch.BIDPRICE+';'
        bidTitle += ch.BIDPAYMENT+';'+ch.BIDDURATION+';'+ch.BIDDERCOST+';'
        bidTitle += ch.BATCHCREATE+';'+ch.CREATETIME+';'+ch.ISSUCCESS+';'
        bidTitle += ch.EXTREWARDINTERVAL+'\n'
        if os.path.getsize(bidfile.name) == 0:
            bidfile.write(bidTitle)
        
        if self.trainable and not self.evaluation:
            plmfile = self.filedict[FILES.PLM_FILE[0]]
            plmTitle = ch.STEP+';'+ch.VEHICLE+';'+ch.EPOCH+';'+ch.NRINPUT+';'
            plmTitle += ch.AVGREWARD+';'+ch.CRITICLOSS+';'+ch.ACTORLOSS+';'
            plmTitle += ch.ATTENTIONLOSS+';'+ ch.ATTENTIONTARGET +';'
            plmTitle += ch.RESTART+'\n'
            if os.path.getsize(plmfile.name)==0:
                plmfile.write(plmTitle)
    
            slfile = self.filedict[FILES.SL_FILE[0]]
            slTitle = ch.STEP+';'+ch.VEHICLE+';'+ch.EPOCH+';'+ch.NRINPUT+';'
            slTitle += ch.AVGREWARD+'\n'
            if os.path.getsize(slfile.name)==0:
                slfile.write(slTitle)
            
            invmodelfile = self.filedict[FILES.INVMDL_FILE[0]]
            invTitle = ch.STEP+';'+ch.VEHICLE+';'+ch.EPOCH+';'+ch.NRINPUT+';'
            invTitle += ch.AVGINVLOSS+'\n'
            if os.path.getsize(invmodelfile.name)==0:
                invmodelfile.write(invTitle)
                
            forwardmodelfile = self.filedict[FILES.FWDMDL_FILE[0]]
            fwdTitle = ch.STEP+';'+ch.VEHICLE+';'+ch.EPOCH+';'+ch.NRINPUT+';'
            fwdTitle += ch.AVGFWDLOSS+'\n'
            if os.path.getsize(forwardmodelfile.name)==0:
                forwardmodelfile.write(fwdTitle)
                
            rewardfile = self.filedict[FILES.REWARD_FILE[0]]
            rewardTitle = ch.STEP+';'+ch.VEHICLE+';'+ch.EPOCH+';'
            rewardTitle += ch.INTREWARD+';'+ch.EXTREWARD+'\n'
            if os.path.getsize(rewardfile.name)==0:
                rewardfile.write(rewardTitle)

    def _print(self,perfile):
        if np.mod(self.schedule.steps,100)==0:
            print("step:{}".format(self.schedule.steps+self.cumstep))

        for bidder in self.bidders.values():
            activebidId = None
            bidPrice = 0
            nrActivebids = len(bidder.activeBid)
            if nrActivebids>0:
                activebidId = bidder.activeBid[0]
                bidPrice = self.bidlist[activebidId].price
            ca = bidder.ca
            curiosity = (False if bidder.curiosityTrainingThres
                         >bidder.priceMdl.history_record else True)
            perfile.write('{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}\n'.format(
              self.schedule.steps+self.cumstep,
              bidder.unique_id,bidder.biddercost,nrActivebids,
              activebidId,bidPrice,bidder.auctionpayment,bidder.auctioncost,
              bidder.initSavings,bidder.currentSavings,bidder.gameoverReward,
              bidder.extRewardInterval,bidder.episodeSteps,curiosity,ca,
              bidder.extSignal,self.fairness))

    def step(self):
        for agent in self.schedule.agents[:]:
            agent.step()

        for agent in self.schedule.agents[:]:
            agent.estimate()
        
        for agent in self.schedule.agents[:]:
            agent.allocate()
        
        for agent in self.schedule.agents[:]:
            agent.advance()    
        
        self._print(self.filedict[FILES.PERF_FILE[0]])
        
        self.schedule.steps += 1
        self.schedule.time += 1

