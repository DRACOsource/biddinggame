# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch import tensor,nn
from torch.autograd import Variable
torch.autograd.set_detect_anomaly(True)
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from ..config.config import (PRICE_MODEL_PARAMS as pmp, 
                             CURIOUS_MODEL_PARAMS as cmp,
                             DEVICE,DTYPE,MODEL_DIR)
from ..solutions.attention import Attention

class Model():
    train_all_records = pmp.train_all_records
    history_record = pmp.history_record
    critic_pretrain_nr_record = pmp.critic_pretrain_nr_record
    batch_size = pmp.batch_size
    epoch = pmp.epoch
    epoch_supervise = pmp.epoch_supervise
    
    def __init__(self,evaluation=False,loadModel=False,curiosity=False,
                 maxReward=None):
        self.unique_id = None
        self.evaluation = evaluation
        self.loadModel = loadModel
        self.curiosity = curiosity
        self.reward = dict()
        self.reward_normalized = dict()
        self.reward_curious = dict()
        self.inputVec = dict()
        self.nextStateVec = dict()
        # price model output: first digit is whether to delay. 
        # rest of output is proportion of allocated budget to each bid
        self.output = dict()
        if maxReward is None:
            self.maxReward = pmp.maxReward
        else:
            self.maxReward = maxReward
        self.inputCounter = 0
        self.firstInput = 0
        
    def _prepInput(self,inputVec,pos=None,var=True):
        if pos is None:
            pos = list(range(len(inputVec)))
        x = []
        if isinstance(inputVec,dict):
            for k in pos:
                x.append(inputVec[str(k)])
            x = np.array(x)
        else:
            x = np.array([x for i,x in enumerate(inputVec) if i in pos])
        if var:
            return Variable(tensor(x,device=DEVICE).type(DTYPE))
        else:
            return tensor(x,device=DEVICE).type(DTYPE)    

    def _removeRecord(self,number=1,model='priceLearningModel'):
        pos = self.firstInput
        for i in range(number):
            try:
                _ = self.inputVec.pop(str(pos))
            except KeyError:
                pass
            try:
                _ = self.output.pop(str(pos))
            except KeyError:
                pass
            if model=='priceLearningModel':
                try:
                    _ = self.reward.pop(str(pos))
                except KeyError:
                    pass
                try:
                    _ = self.nextStateVec.pop(str(pos))
                except KeyError:
                    pass
                if self.curiosity:
                    for j in range(pos+1):
                        try:
                            _ = self.reward_curious.pop(str(j))
                        except KeyError:
                            continue
                        try:
                            _ = self.reward_normalized.pop(str(j))
                        except KeyError:
                            continue
                        
            pos += 1
        self.firstInput += number

    def _prep_reward(self,rewardVec,randomize=True,normalize=True,
                     curiosity=False):
        newMax = max([abs(x) for x in rewardVec.values()])
        if newMax>self.maxReward:
            self.maxReward = newMax
        extraPos = 0
        if curiosity: 
            # add one more position for reward from previous round
            # state vector of curiosity model is the concatenation 
            # of current state and previous reward
            extraPos = 1
        try:
            length = len([v for k,v in rewardVec.items() 
                            if v is not None and not np.isnan(v)])
        except ValueError:
            length = len([v for k,v in rewardVec.items() 
                            if v[0] is not None and not np.isnan(v[0])])
        if length>=self.train_all_records:
            length = min(length,self.history_record+extraPos)
        try:
            pos = [int(k) for k,v in rewardVec.items() 
                            if v is not None and not np.isnan(v)]
        except ValueError:
            pos = [int(k) for k,v in rewardVec.items() 
                            if v[0] is not None and not np.isnan(v[0])]        
        pos.sort()
        r = []
        for k in pos:
            r.append(rewardVec[str(k)])
        # reward from curiosity model
        r = tensor(r[-length+extraPos:],device=DEVICE).type(DTYPE)
        if randomize:
            r = (r + (torch.abs(r+1E-7)/100)**0.5 * torch.randn(len(r))
                ).type(DTYPE)
        if normalize:
            r = r / self.maxReward
        # position for previous reward from MEC, to be concatenated to 
        # the state vector of curiosity model
        pos_tm1 = pos[-length:-extraPos]
        # position for input and output vectors
        pos = pos[-length+extraPos:]
        return r,pos,pos_tm1
    
    def _create_batch(self,x):
        x_batch = []
        for idx in np.arange(self.batch_size,len(x)+1):
            x_batch.append(x[idx-self.batch_size:idx])
        try:
            x_batch = torch.stack(x_batch,dim=0)
            return x_batch
        except:
            return

    def prep_data(self,time,plmfile,reward=None,reward_curious=None,
                   inputVec=None,nextStateVec=None,output=None,
                   curious=False,model='priceLearningModel'):
        if reward is None:
            reward = self.reward.copy()
        if reward_curious is None:
            if model=='priceLearningModel':
                reward_curious = self.reward_curious
            else:
                reward_curious = reward
        if inputVec is None:
            inputVec = self.inputVec
        if nextStateVec is None:
            nextStateVec = self.nextStateVec
        if output is None:
            output = self.output
        try:
            if not curious:
                currentIdx = max([int(k) for k in reward.keys()])
            else:
                currentIdx = max([int(k) for k in reward_curious.keys()])
        except: # if no reward is recorded yet
            return

        if (currentIdx<max(self.critic_pretrain_nr_record,self.batch_size)):
            plmfile.write('{};{};{};too few data points.\n'.format(
                                                    time,self.unique_id,0))
            plmfile.flush()
            return
        
        if not curious: # prepare data for RL without curiosity model
            r,pos,_ = self._prep_reward(rewardVec=reward)
            if model=='supervisedModel':
                r = r.repeat_interleave(2)
                pos = [y for z in [[x*2,x*2+1] for x in pos] for y in z]
            r = r.view(-1,1)        
            x = self._prepInput(inputVec,pos,var=True)
            y = self._prepInput(nextStateVec,pos,var=False)
            a = self._prepInput(output,pos,var=False)
            assert len(x)==len(y)==len(r)
            if model=='supervisedModel':
                return (currentIdx,x,y,r,a,pos)
            
            r_batch = r[self.batch_size-1:]
            a_batch = a[self.batch_size-1:]
            pos_batch = pos[self.batch_size-1:]
            x_batch = self._create_batch(x)
            y_batch = self._create_batch(y)
            return (currentIdx,x_batch,y_batch,r_batch,a_batch,pos_batch)
        
        # prepare data for curiosity model
        interval = 1
        r_curious,pos,pos_tm1 = self._prep_reward(
                            rewardVec=reward_curious,curiosity=True)
        if model=='curiosity':
            r_copy = r_curious.detach().numpy()
            for i,k in enumerate(pos):
                # to be added to interval reward in update_curiousReward()
                self.reward_normalized[str(k)] = r_copy[i]
        
        if model=='supervisedModel':
            # supervised model has two positions for each round, one is
            # from behavior strategy, the other one from best response
            r_curious = r_curious.repeat_interleave(2)
            pos = [y for z in [[x*2,x*2+1] for x in pos] for y in z]
            interval = 2
            
        r_curious = r_curious[:-interval].view(-1,1)
        r = self._prepInput(reward,pos_tm1,var=True)
        if model=='supervisedModel':
            r = r.repeat_interleave(2)
        r_x = r.view(-1,1)
        r_y = r[interval:].view(-1,1)
        x = self._prepInput(inputVec,pos,var=True)
        y = self._prepInput(nextStateVec,pos[:-interval],var=True)
        x = torch.cat([x,r_x],dim=1)
        y = torch.cat([y,r_y],dim=1)
        a_inv = self._prepInput(output,pos,var=False)
        a_fwd = self._prepInput(output,pos,var=True)
        assert len(x)==len(y)+interval==len(r_curious)+interval==len(a_inv)
        
        # create data batches of batchsize
        r_curious_batch = r_curious[self.batch_size-1:]
        a_inv_batch = a_inv[self.batch_size-1:]
        a_fwd_batch = a_fwd[self.batch_size-1:]
        pos_batch = pos[self.batch_size-1:]
        x_batch = self._create_batch(x)
        y_batch = self._create_batch(y)
        if x_batch is None or y_batch is None: # not enough data for one batch
            plmfile.write('{};{};{};too few data points.\n'.format(
                                                    time,self.unique_id,0))
            plmfile.flush()
            return
        return (currentIdx,x_batch,y_batch,r_curious_batch,a_inv_batch,
                a_fwd_batch,pos_batch)

    def collectInput(self,inputVec,model='priceLearningModel',
                     buffer=None):
        if buffer is None:
            buffer = self.history_record * 2
        self.inputVec[str(self.inputCounter)] = inputVec
        self.inputCounter += 1
        if len(self.inputVec)>max(self.history_record+buffer,
                                  self.train_all_records+buffer):
            self._removeRecord(model=model)
        
        return str(self.inputCounter-1) # id for matching output and reward

    def collectOutput(self,output,idx):
        self.output[idx] = output


class PriceLearningModel(Model):
    actor_learning_rate = pmp.actor_learning_rate
    actor_lr_min = pmp.actor_lr_min
    actor_lr_reduce_rate = pmp.actor_lr_reduce_rate
    critic_learning_rate = pmp.critic_learning_rate
    critic_lr_min = pmp.critic_lr_min
    critic_lr_reduce_rate = pmp.critic_lr_reduce_rate
    actor_pretrain_nr_record = pmp.actor_pretrain_nr_record
    reward_rate = pmp.reward_rate
    reward_min = pmp.reward_min
    reward_reduce_rate = pmp.reward_reduce_rate
    add_randomness = pmp.add_randomness
    exploration = pmp.exploration
    
    actor_type = pmp.actor_type
    critic_type = pmp.critic_type
    
    def __init__(self,uniqueId,dimOutput=1,evaluation=False,loadModel=False,
                 curiosity=False,cumstep=0,endOfGame=5000,ca=True,
                 maxReward=None,fairness=False):
        super().__init__(evaluation,loadModel,curiosity,maxReward)
        self.unique_id = uniqueId + '_plm'
        self.dimOutput = dimOutput
        self.actor = None
        self.critic = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.attention = None
        self.ca = ca # ignore attention net(aka credit assignment) if False. 
        if self.ca:
            self.attentionpath = os.path.join(MODEL_DIR,
                                      self.unique_id+'_attention.pkl')
        self.avg_reward = 0
        self.avg_reward_ext = 0
        self.maxExtReward = pmp.maxExtReward
        # fictitious self play fsp
        self.criticpath = os.path.join(MODEL_DIR,
                                       self.unique_id+'_critic_train_fsp.pkl')
        self.actorpath = os.path.join(MODEL_DIR,
                                      self.unique_id+'_actor_train_fsp.pkl')

        self.trainingdata = None # vectors of state, next state, reward, action
        self.reward_ext = dict() # external reward for curiosity model
        # control training of atttention net: 
        # (to train, to save attention.trainingdata)
        self.fairness = fairness # if use fairness as ext. reward
        self.trainWeightvector = (False,False)
        self.exploration = max(
                int(self.exploration / 2**(cumstep / endOfGame)),16)
    
    def _initBudget(self):
        ''' random budget split if model is not available '''
        return list(np.random.rand(self.dimOutput))

    def _initActor(self,inputDim,outputDim,sharedLayers=None):
        paramDim = int(outputDim + outputDim + (outputDim**2 - outputDim) / 2)
        if self.actor_type=='Actor':
            self.actor = MLP_Wrapper(inputDim,paramDim,
                            pmp.actor_hidden_size1,pmp.actor_hidden_size2)
        else:
            self.actor = CNNHighway(inputDim,paramDim,pmp.actor_num_filter,
                            pmp.actor_dropout_rate,pmp.actor_hidden_size1,
                            pmp.actor_hidden_size2,pmp.sharedlayer_output_dim,
                            sharedLayers)
        if DEVICE!=torch.device('cpu'):
            self.actor = nn.DataParallel(self.actor)
        self.actor.to(DEVICE)       
        self.actor_params = list(filter(lambda p: p.requires_grad, 
                                    self.actor.parameters()))
        self.actor_optimizer = torch.optim.SGD(self.actor_params, 
                                           lr=self.actor_learning_rate)
        if self.evaluation or self.loadModel:
            # evaluation: only run inference with previously trained model
            # loadModel: load pre-trained model
            self._reload(self.actorpath)
    
    def _initCritic(self,inputDim,outputDim,sharedLayers=None):
        if self.critic_type=='Critic':
            self.critic = MLP_Wrapper(inputDim,outputDim,
                            pmp.critic_hidden_size1,pmp.critic_hidden_size2)
        else:
            self.critic = CNNHighway(inputDim,outputDim,pmp.critic_num_filter,
                            pmp.critic_dropout_rate,pmp.critic_hidden_size1,
                            pmp.critic_hidden_size2,pmp.sharedlayer_output_dim,
                            sharedLayers)
        if DEVICE!=torch.device('cpu'):
            self.critic = nn.DataParallel(self.critic)
        self.critic.to(DEVICE)
        self.critic_params = list(filter(lambda p: p.requires_grad, 
                                    self.critic.parameters()))
        self.critic_optimizer = torch.optim.SGD(self.critic_params, 
                                           lr=self.critic_learning_rate)
        
        if self.evaluation or self.loadModel:
            # evaluation: only run inference with previously trained model
            # loadModel: load pre-trained model
            self._reload(self.criticpath)
        
            
    def _initAttention(self,inputDim,outputDim):
        self.attention = Attention(self.unique_id,input_size=inputDim,
                                   output_size=outputDim,maxReward=None)
        if DEVICE!=torch.device('cpu'):
            self.attention = nn.DataParallel(self.attention)
        self.attention.to(DEVICE)
        self.attention.setOptim(lr=0.01)
        if self.evaluation or self.loadModel:
            # evaluation: only run inference with previously trained model
            # loadModel: load pre-trained model
            self._reload(self.attentionpath)
        
    def _reload(self,path):
        try:
            checkpoint = torch.load(path)
            if path==self.criticpath:
                self.critic.load_state_dict(checkpoint)
            elif path==self.actorpath:
                self.actor.load_state_dict(checkpoint)
            elif path==self.attentionpath:
                self.attention.load_state_dict(checkpoint)
            else:
                pass
        except:
            pass
        
    def _updateLearningRate(self):
        currentIdx = max([int(k) for k in self.reward.keys()])
        if currentIdx < self.exploration:
            critic_lr_reduce_rate = 1
            actor_lr_reduce_rate = 1
            reward_reduce_rate = 1
        else:
            critic_lr_reduce_rate = self.critic_lr_reduce_rate
            actor_lr_reduce_rate = self.actor_lr_reduce_rate
            reward_reduce_rate = self.reward_reduce_rate
        
        if self.critic is not None:
            self.critic_learning_rate = max(self.critic_lr_min,
                        self.critic_learning_rate * critic_lr_reduce_rate)
            self.critic_optimizer = torch.optim.SGD(self.critic_params, 
                                           lr=self.critic_learning_rate)
        if self.actor is not None:            
            self.actor_learning_rate = max(self.actor_lr_min,
                        self.actor_learning_rate * actor_lr_reduce_rate)
            self.actor_optimizer = torch.optim.SGD(self.actor_params, 
                                               lr=self.actor_learning_rate)
            self.reward_rate = max(self.reward_min,
                                   self.reward_rate * reward_reduce_rate)
                
    def _critic_loss_func(self,value,next_value,reward,avg_reward,rate,
                          invloss,fwdloss):
        if invloss is None:
            invloss = torch.zeros(len(reward))
            fwdloss = torch.zeros(len(reward))
        reward_int = (reward.view(1,-1) 
                      - cmp.invLoss_weight * invloss.view(1,-1) 
                      - (1-cmp.invLoss_weight) * fwdloss.view(1,-1)).mean()
        reward_int = reward_int.detach()
        advantage = (reward + next_value - value 
                     - cmp.invLoss_weight * invloss.view(-1,1)
                     - (1-cmp.invLoss_weight) * fwdloss.view(-1,1))
        for i in range(len(advantage)):
            advantage[i] -= avg_reward
            if not torch.isnan(advantage[i]):
                avg_reward += rate * advantage[i].item()
        return advantage.pow(2).mean(),advantage,avg_reward,reward_int

    def _createCovMat(self,diag,tril):
        # with batchsize
        z = torch.zeros(size=[diag.size(0)],device=DEVICE).type(DTYPE)
        diag = 1E-7 + diag # strictly positive
        elements = []
        trilPointer = 0
        for i in range(diag.shape[1]):
            for j in range(diag.shape[1]):
                if j<i:
                    elements.append(tril[:,trilPointer])
                    trilPointer += 1
                elif j==i:
                    elements.append(diag[:,i])
                else:
                    elements.append(z)
        scale_tril = torch.stack(elements,dim=-1).view(-1,self.dimOutput,
                                                            self.dimOutput)
        return scale_tril

    def _actor_loss_func(self,log_prob_actions,advantage):
        return (advantage.detach() * -log_prob_actions).mean()

    def _chooseAction(self,params):
        ''' 
        action space is multi-dimentional continuous variables. therefore use
            parameterized action estimators, and a multivariate gaussian 
            distribution to output joint probability of the actions. 
            parameters in this case includes N means and N*N covariance 
            matrix elements. Therefore this solution is not scalable when N
            increases. Another solution would be to use a RNN, such as in 
            https://arxiv.org/pdf/1806.00589.pdf
            or http://papers.nips.cc/paper/6398-learning-multiagent-communication-with-backpropagation.pdf
            or https://arxiv.org/pdf/1705.05035.pdf
        params are split into mean, covariance matrix diagonal, 
            cov matrix triangle lower half (since cov matrix is symmetric). 
            also make sure cov is positive definite. 
        '''
        mean,diag,tril = params.split([self.dimOutput,self.dimOutput,
                         params.shape[1]-2*self.dimOutput],dim=-1)
        scale_tril = self._createCovMat(diag,tril)
        dist = MultivariateNormal(loc=mean,scale_tril=scale_tril)
        actions = dist.rsample()
        log_prob_actions = dist.log_prob(actions)
        return actions,log_prob_actions,mean

    def _saveModel(self):
        if self.critic is not None:
            torch.save(self.critic.state_dict(),self.criticpath)
        if self.actor is not None:
            torch.save(self.actor.state_dict(),self.actorpath)
        if self.attention is not None:
            torch.save(self.attention.state_dict(),self.attentionpath)

    def _inference_prepInput(self,inputVec,randomness=None,output_r=False):
        if self.critic is None or self.actor is None:
            x = tensor(self._initBudget(),device=DEVICE).type(DTYPE)
            r = torch.rand(len(x))
            return (False,x,r)
        if randomness is None:
            randomness = self.add_randomness * self.actor_learning_rate
        nr = np.random.rand()
        if nr<randomness:
            x = tensor(self._initBudget(),device=DEVICE).type(DTYPE)
            r = torch.rand(len(x))
            return (False,x,r)
        
        fullInput = list(self.inputVec.values())[
                                        -(self.batch_size-1):] + [inputVec]
        x = self._prepInput(inputVec=fullInput)
            
        if self.curiosity: # add latest MEC reward to the state vector
            r,_,_ = self._prep_reward(self.reward,randomize=True,
                                      curiosity=False)
            r = Variable(r[-len(x):]).view(-1,1)
            x = torch.cat([x,r],dim=1)
                
            if not output_r:
                return (True,x,None)
            else:
                r_output,_,_ = self._prep_reward(self.reward_curious,
                                randomize=True,curiosity=False)
                r_output = r_output[-len(x):].view(1,-1)
                return (True,x,r_output)
        else:
            if not output_r:
                return (True,x,None)
            else:
                r,_,_ = self._prep_reward(self.reward,randomize=True,
                                          curiosity=False)
                r_output = r[-len(x):].view(1,-1)
                return (True,x,r_output)

    def inference(self,inputVec,randomness=None):
        exist,x,_ = self._inference_prepInput(inputVec,randomness)
        if not exist:
            return x
        
        self.actor.eval()
        with torch.no_grad():
            x = x[None, :, :]
            params = self.actor(x)
            actions,_,_ = self._chooseAction(params)
        return torch.clamp(actions[0],0,1)
    
    def inference_weightVec(self,phi=None,r=None,target=None):
        output = self.attention.inference(input_tensor=phi,target_tensor=r,
                                          target=target)
        output = output * len(output)
        return output

    def train(self,time,plmfile,rewardfile,invfile,fwdfile,curMdl=None,
              target=None):
        self.trainingdata = self.prep_data(time,plmfile,
                                                curious=self.curiosity)
        if self.trainingdata is None:
            return

        if not self.curiosity:
            currentIdx,x,y,r,_,pos = self.trainingdata
        else:
            currentIdx,s_t,s_tp1,r,_,_,pos = self.trainingdata
            s_t = s_t[:-1]
            x = s_t
            y = s_tp1
        
        if self.critic is None:
            self._initCritic(x.shape[-1],1)        
        if self.actor is None and currentIdx>=self.actor_pretrain_nr_record:
            self._initActor(x.shape[-1],self.dimOutput,
                            self.critic.sharedLayers)
        if self.attention is None and self.ca:
            self._initAttention(
                inputDim=self.actor.sharedLayers.outputDim,outputDim=1)
        self._saveModel()

        for epoch in range(self.epoch): # price learning model epoch=1
            pointer = 0
            epoch_loss_critic = []
            epoch_loss_actor = []
            epoch_loss_attention = []
            epoch_reward_int = []
            
            while pointer+1<len(x):
                idx = range(pointer,min(pointer+self.batch_size,len(x)))
                invloss_vec = torch.zeros(len(idx))
                fwdloss_vec = torch.zeros(len(idx))
                if self.curiosity:
                    invloss_vec,fwdloss_vec = curMdl.train(time,
                        trainingdata=self.trainingdata,invmodelfile=invfile,
                        forwardmodelfile=fwdfile,pretrain=False,idx=idx,
                        sharedLayers=self.actor.sharedLayers)

                x_batch = x[idx]
                y_batch = y[idx]
                r_batch = r[idx]
                reward = np.nan
                get_r_weight = (self.actor is not None and self.ca 
                                and pointer+self.batch_size<=len(x))
                if get_r_weight:
                    r_weight = self.inference_weightVec(
                            phi=self.actor.sharedLayers(x_batch), 
                            r=r_batch.type(DTYPE),
                            target=target)
                    r_weight = tensor(r_weight)
                    
                    if r_weight.sum()==0 or torch.isnan(r_weight.sum()):
                        r_weight = torch.ones(len(r_batch))
                    r_batch = r_batch.view(-1,1) * r_weight.view(-1,1)
                values = self.critic(x_batch)
                next_values = self.critic(y_batch)   
                
                (critic_loss,advantage,self.avg_reward,
                 reward_int) = self._critic_loss_func(values,next_values,
                           r_batch,self.avg_reward,self.reward_rate,
                           invloss_vec,fwdloss_vec)                

                epoch_loss_critic.append(critic_loss)
                epoch_reward_int.append(reward_int)
                loss = critic_loss
                
                if self.actor is not None:
                    action_params = self.actor(x_batch)
                    actions,log_prob_actions,mean = self._chooseAction(
                                                       action_params)
                    actor_loss = self._actor_loss_func(log_prob_actions,
                                                       advantage)
                    epoch_loss_actor.append(actor_loss)
                    loss += actor_loss
                    self.actor_optimizer.zero_grad()

                self.critic_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # record intrinsic and extrinsic rewards
                if self.trainWeightvector[1]:
                    # new ext. reward
                    try:
                        reward_ext_key = str(max([int(k) 
                                         for k in self.reward_ext.keys()]))
                        reward = self.reward_ext[reward_ext_key]
                    except: 
                        pass
                rewardfile.write('{};{};{};{};{}\n'.format(time,
                                self.unique_id,epoch,reward_int,reward))
                rewardfile.flush()
                
                # trainWeightVector[0]: true when price learning model 
                #     is trained, or when there is new external reward.
                # trainWeightVector[1]: true when there is new external reward
                # train weight vector on the last batch before ext. reward
                if (self.curiosity and pointer+self.batch_size==len(x) 
                    and self.trainWeightvector[0]):
                    try:
                        reward_ext_key = str(max([int(k) 
                                         for k in self.reward_ext.keys()]))
                        reward = self.reward_ext[reward_ext_key]
                    except:
                        break
                    
                    if self.ca: # if attention net:
                        if self.trainWeightvector[1]: # new ext. reward
                            input_tensor = self.actor.sharedLayersOutput.detach().clone()
                            attention_loss = self.attention.train(
                                input_tensor=input_tensor,
                                target_tensor=r_batch.type(DTYPE),
                                end_value=reward)                                
                            self.attention.trainingdata = (currentIdx,s_t,
                                                           s_tp1,r)
                        # retrain on new features and past ext. reward
                        else: 
                            try:
                                (currentIdx,s_t,
                                     s_tp1,r) = self.attention.trainingdata
                                x = s_t
                                x_batch = x[idx]
                                r_batch = r[idx]
                            except:
                                break
                            input_tensor = self.actor.sharedLayers(x_batch).detach().clone()
                            attention_loss = self.attention.train(
                                input_tensor=input_tensor,
                                target_tensor=r_batch.type(DTYPE),
                                end_value=reward)                              
                            
                        if (attention_loss==np.inf or attention_loss==np.nan 
                            or torch.isinf(attention_loss) 
                            or torch.isnan(attention_loss)):
                            self._initAttention(
                             inputDim=self.actor.sharedLayers.outputDim,
                             outputDim=1)    
                            self._reload(self.attentionpath)
                        else:
                            epoch_loss_attention.append(attention_loss)

                pointer += 1
                if pointer+self.batch_size>len(x):
                    break
                
            
            avgLossCritic = sum(epoch_loss_critic)
            if len(epoch_loss_critic) > 0:
                avgLossCritic /= len(epoch_loss_critic)
            avgLossActor = sum(epoch_loss_actor)
            if len(epoch_loss_actor) > 0:
                avgLossActor /= len(epoch_loss_actor)
            avgLossAttention = sum(epoch_loss_attention)
            if len(epoch_loss_attention) > 0:
                avgLossAttention /= len(epoch_loss_attention)
            else:
                avgLossAttention = np.nan
                
            plmfile.write('{};{};{};{};{};{};{};{};{}\n'.format(time,
                    self.unique_id,epoch,len(x),self.avg_reward,
                    avgLossCritic, avgLossActor, avgLossAttention,target))
            
            if avgLossCritic!=0 and torch.isnan(avgLossCritic):
                plmfile.write(
                    '{};{};{};{};{};{};{};{};{};critic restarted.\n'.format(
                    time,self.unique_id,epoch,len(x),self.avg_reward,
                    avgLossCritic,avgLossActor,avgLossAttention,target))
                self._initCritic(x.shape[-1],1)
                self._reload(self.criticpath)
            
            if avgLossActor!=0 and torch.isnan(avgLossActor):
                plmfile.write(
                    '{};{};{};{};{};{};{};{};{};actor restarted.\n'.format(
                    time,self.unique_id,epoch,len(x),self.avg_reward,
                    avgLossCritic,avgLossActor,avgLossAttention,target))
                self._initActor(x.shape[-1],self.dimOutput,
                                self.critic.sharedLayers)
                self._reload(self.actorpath)     
            plmfile.flush()
        
        self._updateLearningRate()
        self.trainingdata = None
        
    
    def collectNextState(self,stateVec,idx):
        self.nextStateVec[idx] = stateVec
        
    def collectReward(self,reward,idx,rewardType='in'):
        if rewardType=='in':
            if (idx not in self.reward.keys() or self.reward[idx] is None 
                                              or np.isnan(self.reward[idx])):
                if idx in self.inputVec.keys():
                    self.reward[idx] = reward
            else:
               self.reward[idx] += reward
        else:
            self.reward_ext[idx] = reward
            if reward>self.maxExtReward:
                self.maxExtReward = reward

    def update_curiousReward(self,rewardVec):
        if rewardVec is None:
            for k in self.reward_normalized.keys(): # add bidding payoff
                self.reward_curious[k] = (pmp.reward_int_weight 
                                           * self.reward_normalized[k])
            self.trainingdata = None
            return
        for (k,v) in rewardVec.items():
            # add reward from curiosity model. v is forward model loss and 
            # is weighted by (1-reward_int_weight) in Curiosity._calc_reward
            self.reward_curious[k] = v
            if k in self.reward_normalized.keys(): # add bidding payoff
                self.reward_curious[k] += (pmp.reward_int_weight 
                                           * self.reward_normalized[k])
            # if there is ext.reward signal but no attention model:
            if self.fairness and not self.ca:
                self.reward_curious[k] = ((1-pmp.reward_ext_weight) 
                                            * self.reward_curious[k])
                if k in self.reward_ext.keys():  
                    self.reward_curious[k] += (pmp.reward_ext_weight 
                                       * self.reward_ext[k]/self.maxExtReward)

        self.trainingdata = None
        

class MLP(nn.Module):
    '''multilayer perceptron as another form of highway'''    
    def __init__(self,inputDim,outputDim,hidden_size1,hidden_size2):
        super().__init__()
        self.batchNorm = nn.BatchNorm1d(inputDim)
        self.hidden1 = nn.Linear(inputDim,hidden_size1)
        self.hidden2 = nn.Linear(hidden_size1,hidden_size2)
        self.batchNorm2  = nn.BatchNorm1d(hidden_size2)
        self.hidden3 = nn.Linear(hidden_size2,outputDim)
        
    def forward(self,x):
        batchnorm = self.batchNorm(x)
        hidden1 = F.relu(self.hidden1(batchnorm))
        hidden2 = F.relu(self.batchNorm2(self.hidden2(hidden1)))
        hidden3 = self.hidden3(hidden2)
        return hidden3


class MLP_Wrapper(nn.Module):
    '''value function estimator. sigmoid layer is used for output to
            control the output range.'''    
    def __init__(self,inputDim,outputDim,hidden_size1,hidden_size2):
        super().__init__()
        self.mlp = MLP(inputDim,outputDim,hidden_size1,hidden_size2)
        self.predict = nn.Sigmoid()
        
    def forward(self,x):
        mlp = self.mlp(x)
        predict = self.predict(mlp)
        return predict


class Highway(nn.Module):
    def __init__(self,in_features,out_features,num_layers=1,bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.bias = bias
        self.cells = nn.ModuleList()
        for idx in range(self.num_layers):
            g = nn.Sequential(
                    nn.Linear(self.in_features, self.out_features),
                    nn.ReLU(inplace=True)
                    )
            t = nn.Sequential(
                    nn.Linear(self.in_features, self.out_features),
                    nn.Sigmoid()
                    )
            self.cells.append(g)
            self.cells.append(t)
        
    def forward(self,x):
        for i in range(0,len(self.cells),2):
            g = self.cells[i]
            t = self.cells[i+1]
            nonlinearity = g(x)
            transformGate = t(x) + self.bias
            x = nonlinearity * transformGate + (1-transformGate) * x
        return x
    

class SharedLayers(nn.Module):
    filter_size = list(np.arange(1,pmp.batch_size,step=2,dtype=int))
    def __init__(self,inputDim,outputDim,num_filter):
        super().__init__()

        self.num_filter = ([num_filter] 
              + [num_filter * 2] * int(len(self.filter_size)/2)
              + [num_filter] * len(self.filter_size))[0:len(self.filter_size)]

        self.num_filter_total = sum(self.num_filter)
        self.inputDim = inputDim
        self.outputDim = outputDim if outputDim>0 else self.num_filter_total
        self.seqLength = pmp.batch_size

        self.batchNorm = nn.BatchNorm1d(pmp.batch_size)
        self.convs = nn.ModuleList()
        for fsize, fnum in zip(self.filter_size, self.num_filter):
            conv = nn.Sequential(
                nn.Conv2d(in_channels=1,out_channels=fnum,
                         kernel_size=(fsize,inputDim),
                         padding=0,stride=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(fnum),
                nn.MaxPool2d(kernel_size=(self.seqLength-fsize+1,1),stride=1)
                )
            self.convs.append(conv)
        
        self.highway = Highway(self.num_filter_total,self.num_filter_total,
                               num_layers=1, bias=0)
        self.compress = nn.Linear(self.num_filter_total,self.outputDim)

    def forward(self,x):
        batchnorm = self.batchNorm(x)
        xs = list()
        for i,conv in enumerate(self.convs):
            x0 = conv(batchnorm.view(-1,1,self.seqLength,self.inputDim))
            x0 = x0.view((x0.shape[0],x0.shape[1]))
            xs.append(x0)
        cats = torch.cat(xs,1)
        highwayOutput = self.highway(cats)
        sharedLayersOutput = nn.Sigmoid()(self.compress(highwayOutput))
        return sharedLayersOutput
        

class CNNHighway(nn.Module):
    def __init__(self,inputDim,outputDim,num_filter,dropout_rate,
                 hidden_size1,hidden_size2,sharedlayer_output_dim,
                 sharedLayers):
        super().__init__()
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.dropout_rate = dropout_rate
        if sharedLayers is None:
            self.sharedLayers = SharedLayers(inputDim,sharedlayer_output_dim,
                                             num_filter)
        else:
            self.sharedLayers = sharedLayers
        self.sharedLayersOutputDim = self.sharedLayers.outputDim
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc_conv = nn.Linear(self.sharedLayersOutputDim,outputDim)
        self.predict = nn.Sigmoid()

    def forward(self,x):
        self.sharedLayersOutput = self.sharedLayers(x)
        dropout = F.relu(self.dropout(self.sharedLayersOutput))
        fc_conv = F.relu(self.fc_conv(dropout))
        predict = self.predict(fc_conv)
        return predict


class SupervisedModel(Model):
    supervise_learning_rate = pmp.supervise_learning_rate
    supervise_hidden_size1 = pmp.supervise_hidden_size1
    supervise_hidden_size2 = pmp.supervise_hidden_size2    
    def __init__(self,uniqueId,dimOutput=1,evaluation=False,loadModel=False,
                 curiosity=False,maxReward=None):
        super().__init__(evaluation,loadModel,curiosity,maxReward)
        self.unique_id = uniqueId + '_supervised'
        self.dimOutput = dimOutput
        self.supervise = None # the model
        self.supervisepath = os.path.join(MODEL_DIR,
                                       self.unique_id+'_train_fsp.pkl')

    def _initBudget(self):
        ''' random budget split if model is not available '''
        return list(np.random.rand(self.dimOutput))

    def _reload(self,path):
        try:
            checkpoint = torch.load(path)
            if path==self.supervisepath:
                self.supervise.load_state_dict(checkpoint)
        except:
            pass
        
    def _saveModel(self,supervise=True):
        if supervise:
            torch.save(self.supervise.state_dict(),self.supervisepath)

    def _initSupervise(self,inputDim,outputDim):
        self.supervise = MLP_Wrapper(inputDim,outputDim,
                    self.supervise_hidden_size1,self.supervise_hidden_size2)
        if DEVICE!=torch.device('cpu'):
            self.supervise = nn.DataParallel(self.supervise)
        self.supervise.to(DEVICE)
        self.supervise_params = list(filter(lambda p: p.requires_grad, 
                                    self.supervise.parameters()))
        self.supervise_optimizer = torch.optim.SGD(self.supervise_params, 
                                           lr=self.supervise_learning_rate)
        self.loss_func = torch.nn.MSELoss()
        
        if self.evaluation or self.loadModel:
            # evaluation: only run inference with previously trained model
            # loadModel: load pre-trained model
            self._reload(self.supervisepath)
 
    def inference(self,inputVec,pmdlReward=None):
        if self.supervise is None:
            return tensor(self._initBudget(),device=DEVICE).type(DTYPE)
        
        if self.curiosity: # add latest MEC reward to the state vector
            fullInput = list(self.inputVec.values())[
                                        -(self.batch_size-1):] + [inputVec]
            x = self._prepInput(inputVec=fullInput)
            r,_,_ = self._prep_reward(pmdlReward,randomize=False,
                                      curiosity=False)
            r = Variable(r[-len(x):]).view(x.shape[0],1)
            x = torch.cat([x,r],dim=1)
        else:
            x = self._prepInput(inputVec=inputVec)
        x = x.reshape(1,-1)
        self.supervise.eval()
        actions = self.supervise(x)
        return torch.clamp(actions[0],0,1)

    def train(self,time,supervisefile,pmdlReward):
        trainingdata = self.prep_data(time,supervisefile,reward=pmdlReward,
                            reward_curious=pmdlReward,inputVec=self.inputVec,
                            nextStateVec=self.inputVec,output=self.output,
                            curious=self.curiosity,model='supervisedModel')
        if trainingdata is not None:
            if not self.curiosity:
                currentIdx,x,_,_,y,_ = trainingdata
            else:
                currentIdx,x,_,_,y,_,_ = trainingdata
        else:
            return
        x = x.view(x.size()[0],-1)
        if self.supervise is None:
            self._initSupervise(x.shape[1],self.dimOutput)
        self._saveModel()

        if len(x)<self.batch_size:
            replace = True
        else:
            replace = False
        for epoch in range(self.epoch_supervise):
            epoch_loss_supervise = []
            idx = np.random.choice(len(x),size=self.batch_size,
                                   replace=replace)
            x_batch = x[idx]
            y_batch = y[idx]
            prediction = self.supervise(x_batch)
            supervise_loss = self.loss_func(prediction, y_batch)
            
            self.supervise_optimizer.zero_grad()
            supervise_loss.backward()
            self.supervise_optimizer.step()
            epoch_loss_supervise.append(supervise_loss)
            
            avgLossSupervise = sum(epoch_loss_supervise)
            if len(epoch_loss_supervise) > 0:
                avgLossSupervise /= len(epoch_loss_supervise)
                
            supervisefile.write('{};{};{};{};{}\n'.format(time,
                    self.unique_id,epoch,len(x),avgLossSupervise))
                    
            if avgLossSupervise!=0 and torch.isnan(avgLossSupervise):
                supervisefile.write(
                    '{};{};{};{};{};supervised learning restarted.\n'.format(
                    time,self.unique_id,epoch,len(x),avgLossSupervise))
                self._initSupervise(x.shape[1],self.dimOutput)
                self._reload(self.supervisepath)
            supervisefile.flush()


class InverseModel(nn.Module):
    '''The inverse module in curiosity model '''    
    def __init__(self,feature_hidden_size1,feature_hidden_size2,
                 inv_hidden_size1,inv_hidden_size2,outputDim,
                 sharedLayers=None):
        if sharedLayers is None:
            return
        super().__init__()
        
        self.features = sharedLayers
        self.feature_outputDim = sharedLayers.outputDim
        
        inv_inputDim = 2 * self.feature_outputDim
        self.batchNorm = nn.BatchNorm1d(inv_inputDim)
        self.hidden1 = nn.Linear(inv_inputDim,inv_hidden_size1)
        self.hidden2 = nn.Linear(inv_hidden_size1,inv_hidden_size2)
        self.batchNorm2  = nn.BatchNorm1d(inv_hidden_size2)
        self.hidden3 = nn.Linear(inv_hidden_size2,outputDim)
        self.predict = nn.Sigmoid()
        
    def forward(self,oldstate,newstate):
        oldfeatures = self.features(oldstate)
        self.oldfeatures = oldfeatures.detach().clone()
        newfeatures = self.features(newstate)
        self.newfeatures = newfeatures.detach().clone()
        x = torch.cat((oldfeatures,newfeatures),-1)
        batchnorm = self.batchNorm(x)
        hidden1 = F.relu(self.hidden1(batchnorm))
        hidden2 = F.relu(self.batchNorm2(self.hidden2(hidden1)))
        hidden3 = self.hidden3(hidden2)
        predict = self.predict(hidden3)
        return predict


class ForwardModel(nn.Module):
    '''The forward module in curiosity model '''    
    feature_hidden_size1 = cmp.feature_hidden_size1
    feature_hidden_size2 = cmp.feature_hidden_size2
    
    def __init__(self,inputDim_action,
                 forward_hidden_size1,forward_hidden_size2,outputDim,
                 sharedLayers=None):
        if sharedLayers is None:
            return
        super().__init__()
        self.features = sharedLayers
        self.feature_outputDim = sharedLayers.outputDim
        
        forward_inputDim = inputDim_action + self.feature_outputDim
        self.batchNorm = nn.BatchNorm1d(forward_inputDim)
        self.hidden1 = nn.Linear(forward_inputDim,forward_hidden_size1)
        self.hidden2 = nn.Linear(forward_hidden_size1,forward_hidden_size2)
        self.batchNorm2  = nn.BatchNorm1d(forward_hidden_size2)
        self.hidden3 = nn.Linear(forward_hidden_size2,outputDim)
        self.predict = nn.Sigmoid()
        
    def forward(self,action,oldstate):
        oldfeatures = self.features(oldstate)
        x = torch.cat((action,oldfeatures),-1)
        batchnorm = self.batchNorm(x)
        hidden1 = F.relu(self.hidden1(batchnorm))
        hidden2 = F.relu(self.batchNorm2(self.hidden2(hidden1)))
        hidden3 = self.hidden3(hidden2)
        predict = self.predict(hidden3)
        return predict
    
    
class Curiosity(Model):

    feature_hidden_size1 = cmp.feature_hidden_size1
    feature_hidden_size2 = cmp.feature_hidden_size2
    
    inv_hidden_size1 = cmp.inv_hidden_size1
    inv_hidden_size2 = cmp.inv_hidden_size2
    inv_learning_rate = cmp.inv_learning_rate

    forward_hidden_size1 = cmp.forward_hidden_size1
    forward_hidden_size2 = cmp.forward_hidden_size2
    forward_learning_rate = cmp.forward_learning_rate
    
    batch_size = cmp.batch_size
    epoch = cmp.epoch
        
    def __init__(self,uniqueId,dimAction=1,evaluation=False,
                 loadModel=False,maxReward=None):
        super().__init__(evaluation,loadModel,curiosity=True,
                         maxReward=maxReward)
        self.unique_id = uniqueId + '_curious'
        self.dimOutput_action = dimAction
        self.invmodel = None # the inverse model
        self.invmodelpath = os.path.join(MODEL_DIR,
                                self.unique_id+'_train_inv.pkl')
        self.forwardmodel = None # the forward model
        self.forwardmodelpath = os.path.join(MODEL_DIR,
                                self.unique_id+'_train_forward.pkl')
        self.reward = None
        self.sharedLayers = None
        
    def _initSharedLayers(self,sharedLayers=None):
        if self.sharedLayers is None:
            self.sharedLayers = sharedLayers
            self.feature_outputDim = sharedLayers.outputDim

    def _initOutput_action(self):
        ''' random output if inverse model is not available '''
        return list(np.random.rand(self.dimOutput_action))
    
    def _initOutput_features(self):
        ''' random output if forward model is not available '''
        return list(np.random.rand(self.feature_outputDim))

    def _reload_invmodel(self,path):
        try:
            checkpoint = torch.load(path)
            if path==self.invmodelpath:
                self.invmodel.load_state_dict(checkpoint)
        except:
            pass

    def _reload_forwardmodel(self,path):
        try:
            checkpoint = torch.load(path)
            if path==self.forwardmodelpath:
                self.forwardmodel.load_state_dict(checkpoint)
        except:
            pass    
        
    def _saveModel(self,invmodel=True,forwardmodel=True):
        if invmodel and self.invmodel is not None:
            torch.save(self.invmodel.state_dict(),self.invmodelpath)
        if forwardmodel and self.forwardmodel is not None:
            torch.save(self.forwardmodel.state_dict(),self.forwardmodelpath)

    def _initInvmodel(self,sharedLayers=None):
        self._initSharedLayers(sharedLayers)
        self.invmodel = InverseModel(
                 self.feature_hidden_size1,self.feature_hidden_size2,
                 self.inv_hidden_size1,self.inv_hidden_size2,
                 self.dimOutput_action,self.sharedLayers)        
        
        if DEVICE!=torch.device('cpu'):
            self.invmodel = nn.DataParallel(self.invmodel)
        self.invmodel.to(DEVICE)
        self.invmodel_params = list(filter(lambda p: p.requires_grad, 
                                    self.invmodel.parameters()))
        self.invmodel_optimizer = torch.optim.SGD(self.invmodel_params, 
                                           lr=self.inv_learning_rate)
        self.invmodel_loss_func = torch.nn.MSELoss()
        
        if self.evaluation or self.loadModel:
            # evaluation: only run inference with previously trained model
            # loadModel: load pre-trained model
            self._reload_invmodel(self.invmodelpath)

    def _initForwardmodel(self):
        if self.invmodel is None:
            return
            
        self.forwardmodel = ForwardModel(self.dimOutput_action,
            self.forward_hidden_size1,self.forward_hidden_size2,
            self.feature_outputDim,self.invmodel.features)        
        
        if DEVICE!=torch.device('cpu'):
            self.forwardmodel = nn.DataParallel(self.forwardmodel)
        self.forwardmodel.to(DEVICE)
        self.forwardmodel_params = list(filter(lambda p: p.requires_grad, 
                                    self.forwardmodel.parameters()))
        self.forwardmodel_optimizer = torch.optim.SGD(self.forwardmodel_params, 
                                           lr=self.forward_learning_rate)
        self.forwardmodel_loss_func = torch.nn.MSELoss()
        
        if self.evaluation or self.loadModel:
            # evaluation: only run inference with previously trained model
            # loadModel: load pre-trained model
            self._reload_forwardmodel(self.forwardmodelpath)

    def _calc_reward(self,pos,oldInputVec_state,newInputVec_state,
                      inputVec_actualAction):
        idx = range(len(oldInputVec_state))
        s_t_batch = oldInputVec_state[idx]
        s_tp1_batch = newInputVec_state[idx]
        a_f_batch = inputVec_actualAction[idx]
        a,phi,phi_tp1 = self.inference_invmodel(s_t_batch,s_tp1_batch)
        phi_tp1_hat = self.inference_forwardmodel(
                                    s_t_batch,a_f_batch).detach().numpy()
        phi_tp1 = phi_tp1.detach().numpy()
        a_i_batch = a.detach().numpy()
        a_f_batch = a_f_batch.detach().numpy()
        invLoss = list(((a_i_batch-a_f_batch)**2).mean(axis=1))
        fwdLoss = list(((phi_tp1-phi_tp1_hat)**2).mean(axis=1))
        predLoss = fwdLoss
        keys = [str(k) for k in pos]
        self.reward = dict([(k,(1-pmp.reward_int_weight)*v) 
                            for (k,v) in zip(keys,predLoss)])
 
    def inference_invmodel(self,oldInputVec_state,newInputVec_state):
        if self.invmodel is None:
            return (tensor(self._initOutput_action(),
                           device=DEVICE).type(DTYPE),
                    tensor(self._initOutput_features(),
                           device=DEVICE).type(DTYPE),
                    tensor(self._initOutput_features(),
                           device=DEVICE).type(DTYPE))

        self.invmodel.eval()
        actions = self.invmodel(oldInputVec_state,newInputVec_state)
        newfeatures = self.invmodel.newfeatures
        oldfeatures = self.invmodel.oldfeatures
        return torch.clamp(actions,0,1), oldfeatures, newfeatures
    
    def inference_forwardmodel(self,oldInputVec_state,actualAction):
        self.forwardmodel.eval()
        newstate = self.forwardmodel(actualAction,oldInputVec_state)       
        return newstate

    def train(self,time,trainingdata,invmodelfile,forwardmodelfile,
              pretrain=True,idx=None,sharedLayers=None):
        
        if trainingdata is None:
            return None,None
        
        (currentIdx,oldInputVec_state,newInputVec_state,_,
         actualAction_inv,actualAction_fwd,pos) = trainingdata
        
        s_t = oldInputVec_state[:-1]
        s_tp1 = newInputVec_state
        a_t_inv = actualAction_inv[:-1]
        a_t_fwd = actualAction_fwd[:-1]
        pos = pos[:-1]

        if self.invmodel is None:
            self._initInvmodel(sharedLayers)
            self._initForwardmodel()
        self._saveModel()

        if len(s_t)<self.batch_size:
            replace = True
        else:
            replace = False
        
        training_epoch = 1
        if pretrain:
            training_epoch = self.epoch
        
        for epoch in range(training_epoch):
            epoch_loss_invmodel = []
            epoch_loss_forwardmodel = []
            
            if pretrain or idx is None:
                idx = np.random.choice(len(s_t),size=self.batch_size,
                                                       replace=replace)
                
            s_t_batch = s_t[idx]
            s_tp1_batch = s_tp1[idx]
            a_i_batch = a_t_inv[idx]
            a_f_batch = a_t_fwd[idx]
            
            action_pred = self.invmodel(s_t_batch,s_tp1_batch)
            invmodel_loss = self.invmodel_loss_func(action_pred,a_i_batch)
            invloss_vec = (action_pred-a_i_batch).pow(2).mean(dim=-1)
            
            if pretrain:
                self.invmodel_optimizer.zero_grad()
                invmodel_loss.backward()
                self.invmodel_optimizer.step()
            epoch_loss_invmodel.append(invmodel_loss.detach())
            
            newfeature_actual = self.invmodel.newfeatures
            feature_pred = self.forwardmodel(a_f_batch,s_t_batch)
            forwardmodel_loss = self.forwardmodel_loss_func(feature_pred,
                                                        newfeature_actual)
            fwdloss_vec = (feature_pred-newfeature_actual).pow(2).mean(dim=-1)
            
            if pretrain:
                self.forwardmodel_optimizer.zero_grad()
                forwardmodel_loss.backward()
                self.forwardmodel_optimizer.step()
            epoch_loss_forwardmodel.append(forwardmodel_loss.detach())
            
            avgLossInvmodel = sum(epoch_loss_invmodel)
            avgLossForwardmodel = sum(epoch_loss_forwardmodel)
            if len(epoch_loss_invmodel) > 0:
                avgLossInvmodel /= len(epoch_loss_invmodel)
            if len(epoch_loss_forwardmodel) > 0:
                avgLossForwardmodel /= len(epoch_loss_forwardmodel)
            
            if pretrain:
                invmodelfile.write('{};{};{};{};{}\n'.format(time,
                        self.unique_id,epoch,len(s_t),avgLossInvmodel))
                invmodelfile.flush()
                forwardmodelfile.write('{};{};{};{};{}\n'.format(time,
                        self.unique_id,epoch,len(s_t),avgLossForwardmodel))
                forwardmodelfile.flush()
                    
            if avgLossInvmodel!=0 and torch.isnan(avgLossInvmodel):
                invmodelfile.write(
                  '{};{};{};{};{};inverse model learning restarted.\n'.format(
                  time,self.unique_id,epoch,len(s_t),avgLossInvmodel))
                invmodelfile.flush()
                self._initInvmodel(self.sharedLayers)
                self._reload_invmodel(self.invmodelpath)
            
            if avgLossForwardmodel!=0 and torch.isnan(avgLossForwardmodel):
                forwardmodelfile.write(
                  '{};{};{};{};{};forward model learning restarted.\n'.format(
                  time,self.unique_id,epoch,len(s_t),avgLossForwardmodel))
                forwardmodelfile.flush()
                self._initForwardmodel()
                self._reload_forwardmodel(self.forwardmodelpath)
            
        self._calc_reward(pos,s_t,s_tp1,a_t_fwd)
        return invloss_vec,fwdloss_vec

