# -*- coding: utf-8 -*-
from biddinggame.config.config import LOGS_DIR
from biddinggame.utils.name_utils import ColumnHead as ch
from biddinggame.utils.graphic_utils import Graphics
from biddinggame.utils.common_utils import CommonUtils as cu
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings("ignore")

def getFairnessIndex(df,col):
    values = list(df[col])
    num = sum(values)
    denom = sum([x**2 for x in values])
    if denom==0:
        result = 0.5
    else:
        result = (num**2/len(values)) / denom
    return result

def getCustomFairnessGraph(data0,col,hue,density,graph,name,legends,title,
                           yaxis='fairness score',order='fixed',
                           ylim=(None,None),threshold=0):
    try:
        data = data0[[ch.STEP,ch.VEHICLE,col,hue]]
    except:
        data = data0[[ch.STEP,ch.VEHICLE,col]]
    data[ch.STEP] = data[ch.STEP].apply(lambda x: int(x/density)*density)
    try:
        data = data.groupby([ch.STEP,ch.VEHICLE,hue],as_index=False).sum()
    except:
        data = data.groupby([ch.STEP,ch.VEHICLE],as_index=False).sum()

    try:
        data = data.groupby([ch.STEP,hue]).apply(
                                           lambda x: getFairnessIndex(x,col))
    except:
        data = data.groupby(ch.STEP).apply(lambda x: getFairnessIndex(x,col))
    data = pd.DataFrame(data,columns=[yaxis])
    data.reset_index(inplace=True)
    
    data[ch.STEP] = data[ch.STEP].apply(int)
    data = data[data[yaxis]>=threshold]
    graph._drawLineplot(df=data,x=ch.STEP,y=yaxis,hue=hue,style=hue,
                order=order,title=name+title,legends=legends,
                decimals=2,legendFontsize=14,ylim=ylim)

def myLineplot(data,col,title,dense,graph,name,legends,ci,decimals=0,hue=None,
                 order='fixed',ylabel=None,ylim=(None,None),threshold=-10000):
    try:
        df = data[set([ch.STEP,ch.VEHICLE,col,hue,ch.FILEVERSION])]
    except:
        df = data[[ch.STEP,ch.VEHICLE,col,ch.FILEVERSION]]
    df[ch.STEP] = df[ch.STEP].apply(lambda x: int(x/dense)*dense)
    try:
        df = df.groupby([ch.STEP,ch.VEHICLE,hue,ch.FILEVERSION],
                    as_index=False).sum()
    except:
        df = df.groupby([ch.STEP,ch.VEHICLE,ch.FILEVERSION],
                    as_index=False).sum()            
    if ci is None:
        try:
            df = df.groupby([ch.STEP,hue],as_index=False).mean()
        except:
            df = df.groupby(ch.STEP,as_index=False).mean()
    else:
        try:
            df = df.groupby([ch.STEP,hue,ch.FILEVERSION],
                            as_index=False).mean()
        except:
            df = df.groupby([ch.STEP,ch.FILEVERSION],as_index=False).mean()
    df = df[df[col]>=threshold]
    graph._drawLineplot(df=df,x=ch.STEP,y=col,hue=hue,style=hue,
                        order=order,title=name+title,legends=legends,
                        ci=ci,decimals=decimals,legendFontsize=14,
                        ylabel=ylabel,ylim=ylim)

def outputPerformance(path=LOGS_DIR,density=None,name=None,ci=None,
                      outputSingleBidder=True,outputRange=None,
                      legends='best',fairness=False,simType='heterogeneous'):
    if density is None:
        density = 300
    if name is None:
        name = 'performance'
    ylabel = 'avg. cum. payoff'
    
    graph = Graphics(path)
    data = graph._collectData(name=name)
    data = cu.changeColumnName(data,ch.GAMEOVERREWARD,ch.REWARD)
    if outputRange is not None:
        data = data[(data[ch.STEP]>=outputRange[0]) 
                            & (data[ch.STEP]<=outputRange[1])]
    def myMdlType(row):
        if not row[ch.CURIOSITYMDL]:
            return 'SHT'
        if not row[ch.ATTENTIONMDL]:
            return 'CUR'
        return 'DRA'
    data[ch.MODELTYPE] = data.apply(lambda row: myMdlType(row),axis=1)

    data0 = data.copy()
    
    # original simulation, reward comparison by model type
    data = data[(data[ch.FAIRNESS]==fairness) & 
                (data[ch.SIMULATIONTYPE]==simType)]
    for bidder in set(data[ch.VEHICLE]):
        hue = 'savings type'
        tmp0 = data.loc[data[ch.VEHICLE]==bidder,[ch.STEP,ch.CURRENTSAVINGS]]
        tmp0[hue] = 'savings'
        tmp1 = data.loc[data[ch.VEHICLE]==bidder,[ch.STEP,ch.REWARD]]
        tmp1[hue] = 'reward' 
        tmp1.columns = [ch.STEP,ch.CURRENTSAVINGS,hue]
        tmp = pd.concat([tmp0,tmp1],axis=0)
        if outputSingleBidder:
            graph._drawLineplot(df=tmp,x=ch.STEP,y=ch.CURRENTSAVINGS,hue=hue,
                            title=name+'_'+bidder,legends='best')    
    myLineplot(data=data,col=ch.REWARD,title='_gameoverReward',dense=density,
               graph=graph,name=name,legends=legends,ci=ci,decimals=0,
               hue=ch.MODELTYPE,order='flex',ylabel=ylabel)

    data[ch.VEHICLE] = data[ch.VEHICLE].apply(lambda x: x[:7])
    myLineplot(data=data,col=ch.REWARD,title='_gameoverReward_peragent',
       dense=density,graph=graph,name=name,legends=legends,ci=ci,decimals=0,
       hue=ch.VEHICLE,order='fixed',ylabel=ylabel)
    
    # compare by simulation type
    def mySimType(row):
        if 'hetero' in row[ch.SIMULATIONTYPE]:
            return 'HETERO'
        return 'All ' + row[ch.MODELTYPE]
    if len(set(data0[ch.SIMULATIONTYPE]))>1:
        data = data0[data0[ch.FAIRNESS]==fairness]
        data[ch.SIMULATIONTYPE] = data.apply(lambda row: mySimType(row),axis=1)
        myLineplot(data=data,col=ch.REWARD,
            title='_gameoverRewardComparison_simType',
            dense=density,graph=graph,name=name,legends='best',ci=ci,
            decimals=0,hue=ch.SIMULATIONTYPE,order='flex',
            ylabel=ylabel)

    # compare by ext. reward type
    fi = 'fairness-signal'
    cp = 'payoff-signal'
    if len(set(data0[ch.FAIRNESS]))>1:
        data = data0[data0[ch.SIMULATIONTYPE]==simType]
        data[ch.FAIRNESS] = np.where(data[ch.FAIRNESS],fi,cp)
        myLineplot(data=data,col=ch.REWARD,
               title='_gameoverRewardComparison_rewardType',
               dense=density,graph=graph,name=name,legends='best',ci=ci,
               decimals=0,hue=ch.FAIRNESS,ylabel=ylabel
               #,threshold=500
               )
        
    # compare by both simulation and ext. reward type
    if len(set(data0[ch.SIMULATIONTYPE]))>1 and len(set(data0[ch.FAIRNESS]))>1:
        coln = 'alltypes'
        data = data0.copy()
        data[ch.FAIRNESS] = np.where(data[ch.FAIRNESS],fi,cp)
        data[ch.SIMULATIONTYPE] = data.apply(lambda row: mySimType(row),axis=1)
        data[coln] = data.apply(lambda row: row[ch.SIMULATIONTYPE] + ', ' 
                                            + row[ch.FAIRNESS], axis=1)
        myLineplot(data=data,col=ch.REWARD,
               title='_gameoverRewardComparison_allTypes',
               dense=density,graph=graph,name=name,legends='best',ci=ci,
               decimals=0,hue=coln,ylabel=ylabel)        
    
    # fairness comparison
    data0 = data0[data0[ch.STEP]>=150]
    density = 300
    legends = 'best'
    yaxis = 'fairness index score'
    
    # original simulation
    hue = None
    data = data0[(data0[ch.FAIRNESS]==fairness) & 
                 (data0[ch.SIMULATIONTYPE]==simType)]
    data[ch.FAIRNESS] = np.where(data[ch.FAIRNESS],fi,cp)
    data[ch.SIMULATIONTYPE] = data.apply(lambda row: mySimType(row),axis=1)   
    getCustomFairnessGraph(data,col=ch.AUCTIONPAYMENT,hue=hue,
                density=density,graph=graph,name=name,legends=legends,
                title='_fairness_paymentComparison_original',yaxis=yaxis)
    
    # compare by simulation type:
    if len(set(data0[ch.SIMULATIONTYPE]))>1:
        data = data0[data0[ch.FAIRNESS]==fairness]
        data[ch.SIMULATIONTYPE] = data.apply(lambda row: mySimType(row),axis=1)
        hue = ch.SIMULATIONTYPE
        getCustomFairnessGraph(data,col=ch.AUCTIONPAYMENT,hue=hue,
                density=density,graph=graph,name=name,legends=legends,
                title='_fairness_paymentComparison_simType',yaxis=yaxis)

    # compare by ext. reward type
    if len(set(data0[ch.FAIRNESS]))>1:
        data = data0[data0[ch.SIMULATIONTYPE]==simType]
        data[ch.FAIRNESS] = np.where(data[ch.FAIRNESS],fi,cp)
        hue = ch.FAIRNESS
        getCustomFairnessGraph(data,col=ch.AUCTIONPAYMENT,hue=hue,
                density=density,graph=graph,name=name,legends=legends,
                title='_fairness_paymentComparison_rewardType',yaxis=yaxis)
    
    # compare by all types
    if len(set(data0[ch.SIMULATIONTYPE]))>1 and len(set(data0[ch.FAIRNESS]))>1:
        coln = 'alltypes'
        data = data0.copy()
        data[ch.FAIRNESS] = np.where(data[ch.FAIRNESS],fi,cp)
        data[ch.SIMULATIONTYPE] = data.apply(lambda row: mySimType(row),axis=1)
        data[coln] = data.apply(lambda row: row[ch.SIMULATIONTYPE] + ', ' 
                                            + row[ch.FAIRNESS], axis=1) 
        hue = coln
        getCustomFairnessGraph(data,col=ch.AUCTIONPAYMENT,hue=hue,
                density=density,graph=graph,name=name,legends=legends,
                title='_fairness_paymentComparison_allTypes',yaxis=yaxis)
    
        
def getbidders(data,bidders=None):
    if bidders is None:
        return [(x,x[0:7]) for x in set(data[ch.VEHICLE])]
    if isinstance(bidders,str):
        bidders = [bidders]
    if isinstance(bidders,list):
        bidderlist = []
        for b in bidders:
            bidderlist += [(x,b) for x in set(data[ch.VEHICLE]) if b in x]
        if len(bidderlist)==0:
            return [(x,x[0:7]) for x in set(data[ch.VEHICLE])]        
        return list(set(bidderlist))
    if isinstance(bidders,dict):
        bidderlist = []
        for k,v in bidders.items():
            bidderlist += [(x,v) for x in set(data[ch.VEHICLE]) if k in x]
        return list(set(bidderlist))

def outputPLM(path=LOGS_DIR,dataRange=None,bidders=None,threshold=0,
              density=300):
    graph = Graphics(path)    
    name = 'priceLearningModel'
    data = graph._collectData(name=name)
    try:
        bidderlist = getbidders(data=data,bidders=bidders)
    except:
        bidderlist = []
    for bidder,value in bidderlist:
        tmp3 = data.loc[data[ch.VEHICLE]==bidder,[ch.STEP,ch.ATTENTIONLOSS]]
        if dataRange is not None:
            tmp3 = tmp3.loc[(tmp3[ch.STEP]>=dataRange[0]) & 
                            (tmp3[ch.STEP]<=dataRange[1])]
        tmp3[ch.STEP] = tmp3[ch.STEP].apply(lambda x: int(x/density)*density)
        tmp30 = tmp3[[ch.STEP,ch.ATTENTIONLOSS]].groupby([ch.STEP],
                                                as_index=False).sum()
        tmp30[ch.ATTENTIONLOSS] = np.where(tmp30[ch.ATTENTIONLOSS]<threshold,
                                          threshold,tmp30[ch.ATTENTIONLOSS])
        graph._drawLineplot(df=tmp30,x=ch.STEP,y=ch.ATTENTIONLOSS,
                        title=name+'_ACattentionloss_'+bidder,legends=None,
                        ylabel='credit assignment loss')        

def myConcat(tbl,tmp,coln=None,value=None):
    if coln is not None:
        tmp[coln] = value
    if tbl is None:
        tbl = tmp
    else:
        tbl = pd.concat([tbl,tmp],axis=0)
    return tbl

def outputCuriosityLoss(path=LOGS_DIR,bidders=None,density=1,
                        dataRange=None,compare=False):
    graph = Graphics(path)
    name = 'forwardModel'
    data = graph._collectData(name=name)
    if dataRange is not None:
        data = data.loc[(data[ch.STEP]>=dataRange[0]) & 
                        (data[ch.STEP]<=dataRange[1])]
    try:
        bidderlist = getbidders(data=data,bidders=bidders)
    except:
        bidderlist = []
    tbl = None
    for bidder,value in bidderlist:
        if 'plm' in bidder:
            continue
        tmp = data.loc[data[ch.VEHICLE]==bidder,[ch.STEP,ch.AVGFWDLOSS]]
        tmp[ch.STEP] = tmp[ch.STEP].apply(lambda x: int(x/density)*density)
        tmp = tmp.groupby([ch.STEP],as_index=False).sum()
        tbl = myConcat(tbl=tbl,tmp=tmp,coln=ch.VEHICLE,value=value)
        graph._drawLineplot(df=tmp,x=ch.STEP,y=ch.AVGFWDLOSS,
                    title='fwdloss_'+bidder,legends=None,
                    ylabel='forward model loss')
    if compare and tbl is not None:
        hue_order = [value for bidder,value in bidderlist]
        graph._drawLineplot(df=tbl,x=ch.STEP,y=ch.AVGFWDLOSS,
                    hue=ch.VEHICLE,style=ch.VEHICLE,order='fixed',
                    hue_order=hue_order,decimals=3,
                    title='fwdloss_compare',legends='best',
                    ylabel='forward model loss')        
    
    name = 'reward'
    data = graph._collectData(name=name)
    if dataRange is not None:
        data = data.loc[(data[ch.STEP]>=dataRange[0]) & 
                        (data[ch.STEP]<=dataRange[1])]
    try:
        bidderlist = getbidders(data=data,bidders=bidders)
    except:
        bidderlist = []
    intTbl = None
    extTbl = None
    for bidder,value in bidderlist:
        tmp = data.loc[data[ch.VEHICLE]==bidder,[ch.STEP,ch.INTREWARD]]
        tmp[ch.STEP] = tmp[ch.STEP].apply(lambda x: int(x/density)*density)
        tmp = tmp.groupby([ch.STEP],as_index=False).sum()
        intTbl = myConcat(tbl=intTbl,tmp=tmp,coln=ch.VEHICLE,value=value)
        graph._drawLineplot(df=tmp,x=ch.STEP,y=ch.INTREWARD,
                    title='intReward_'+bidder,legends=None,
                    ylabel='int. reward')
        tmp = data.loc[data[ch.VEHICLE]==bidder,[ch.STEP,ch.EXTREWARD]]
        tmp[ch.STEP] = tmp[ch.STEP].apply(lambda x: int(x/density)*density)
        tmp = tmp.groupby([ch.STEP],as_index=False).mean()
        extTbl = myConcat(tbl=extTbl,tmp=tmp,coln=ch.VEHICLE,value=value)
        graph._drawLineplot(df=tmp,x=ch.STEP,y=ch.EXTREWARD,
                    title='extReward_'+bidder,legends=None,
                    ylabel='ext. reward')
    if compare and intTbl is not None:
        hue_order = [value for bidder,value in bidderlist]
        graph._drawLineplot(df=intTbl,x=ch.STEP,y=ch.INTREWARD,
                    hue=ch.VEHICLE,style=ch.VEHICLE,order='fixed',
                    hue_order=hue_order,
                    title='intReward_compare',legends='best',decimals=3,
                    ylabel='int. reward')
    if compare and extTbl is not None:
        hue_order = [value for bidder,value in bidderlist]
        graph._drawLineplot(df=extTbl,x=ch.STEP,y=ch.EXTREWARD,
                    hue=ch.VEHICLE,style=ch.VEHICLE,order='fixed',
                    hue_order=hue_order,
                    title='extReward_compare',legends='best',decimals=3,
                    ylabel='ext. reward')  

#%%
if __name__ == '__main__': 
    path = LOGS_DIR
    fairness = False
    simType = 'heterogeneous'
    
    if 'fair' in path:
        fairness = True
    if 'att' in path or 'cur' in path or 'greedy' in path:
        simType = 'homogeneous'
    outputPerformance(path=path,name='performance',
                outputSingleBidder=False,density=1000,legends='best',
                fairness=fairness,simType=simType,outputRange=(0,100000))
#    outputPLM(path=path,dataRange=(1000,100000),bidders=['bidder2'],
#              threshold=0.6,density=500)
#    outputCuriosityLoss(path=path,bidders={'bidder2':'DRA','bidder4':'CUR'},
#              density=500,dataRange=(1000,100000),compare=True)
