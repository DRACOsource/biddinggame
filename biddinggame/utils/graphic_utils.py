# -*- coding: utf-8 -*-
from ..config.config import (GRAPH_DIR, MDL_PARAMS as mp, 
                             BIDDER_PARAMS as vp,
                             PRICE_MODEL_PARAMS as pmp)
# from ..supports.data import TraceData
from ..utils.name_utils import ColumnHead as ch
import glob,os,re,ast
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15
plt.rc('pdf',fonttype=42)
plt.ioff()
import seaborn as sns
from scipy.stats import gaussian_kde


class OOMFormatter(ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format

class Graphics:
    learned = ch.LEARNED
    random = ch.RANDOM
    retrain = ch.RETRAIN
    
    def __init__(self,path):
        self.path = path
    
    def _collectData(self,name,density=1):
        name = name + '*'
        filename = os.path.join(self.path,name)
        perfiles = glob.glob(filename)
        data = pd.DataFrame()
        data_part = pd.DataFrame()
        for f in perfiles:
            try:
                data_part = pd.read_csv(f,sep=';')
            except:
                continue
            locations = [x.span() for x in re.finditer('_',f)]
            try:
                interval = str(f[locations[-2][1]:locations[-1][0]])
            except:
                interval = 0
            try:
                fileversion = str(f[locations[-2][0]-1
                                              :locations[-2][0]])
                if not re.match('\d',fileversion):
                    fileversion = ''
            except:
                continue
            trained = (self.learned 
                       if f[locations[-1][1]:-4]=='True' else self.random)
            if 'Retrain' in f:
                trained = trained + ' ' + self.retrain
            
            simulationType = 'heterogeneous'
            try:
                if (len(set(data_part[ch.CURIOSITYMDL]))==1 
                    and len(set(data_part[ch.ATTENTIONMDL]))==1):
                    simulationType = 'homogeneous'
            except:
                pass
            data_part[ch.SIMULATIONTYPE] = simulationType
            data_part[ch.TRAINED] = trained
            data_part[ch.INTERVAL] = interval
            data_part[ch.FILEVERSION] = fileversion
            if data.shape[0]==0:
                data = data_part
            else:
                data = pd.concat([data,data_part],axis=0)
        
        data[ch.BATCHSIZE] = pmp.batch_size
        return data
    
    def _drawLineplot(self,df,x,y,title,style=None,hue=None,order='flex',
                      hue_order=None,legends=2,legendFontsize=None,
                      tickFontsize=None,size=None,separate=None,
                      decimals=1,ci=None,vertical=True,
                      ylim=(None,None),yscale='linear',yaxisTick='left',
                      ylabel=None,xlabel=None):
        defaultFontsize = 16
        if tickFontsize is None:
            tickFontsize = defaultFontsize
        if legendFontsize is None:
            legendFontsize = defaultFontsize
        if size is None:
            length = 5
            height = 4
        else:
            length = size[0]
            height = size[1]
        if xlabel is None:
            xlabel = x
        if ylabel is None:
            ylabel = y

        if separate is None:
            if hue is not None:
                if hue_order is None:
                    if order=='flex':
                        tmp = df.groupby(hue)
                        tmp = (tmp.tail(1).drop_duplicates()
                               .sort_values(y,ascending=False))
                    else:
                        tmp = df.groupby(
                                hue,as_index=False).max().sort_values(hue,
                                                       ascending=True)
                    hue_order = tmp[hue].tolist()
            else:
                hue_order = None
            
            fig,ax = plt.subplots()
            fig.set_size_inches(length,height)
            
            try:
                dashes = [(len(hue_order)-x,x,2*x,x)
                                for x in list(range(len(hue_order)))]
            except:
                dashes = None
            try:
                ax = sns.lineplot(x=x,y=y,style=style,hue=hue,data=df,ci=ci,
                                  hue_order=hue_order,style_order=hue_order,
                                  dashes=dashes)
            except ValueError: # not enough styles
                ax = sns.lineplot(x=x,y=y,style=style,hue=hue,data=df,ci=ci,
                                  hue_order=hue_order,dashes=dashes)                
                
            tmp = df[~df[y].isnull()]
            if len(tmp)==0:
                return
            ylim_min = min(tmp[y])
            ylim_max = max(tmp[y])
            if ylim[0] is not None:
                ylim_min = ylim[0]
            if ylim[1] is not None:
                ylim_max = ylim[1]
            ax.set_ylim(ylim_min,ylim_max)
            ax.set_yscale(yscale)
            
            ax.set_xlabel(xlabel=xlabel,fontsize=tickFontsize)
            ax.set_ylabel(ylabel=ylabel,fontsize=tickFontsize)
 
            ax.tick_params(axis='both',which='major',labelsize=tickFontsize)
            ax.tick_params(axis='both',which='minor',labelsize=tickFontsize-3)
            yformatter = ScalarFormatter(useOffset=True,useMathText=True)
            xformatter = OOMFormatter(order=3,fformat='%2.0f')
            ax.yaxis.set_major_formatter(yformatter)
            ax.yaxis.set_minor_formatter(yformatter)
            ax.xaxis.set_major_formatter(xformatter)
            ax.xaxis.get_offset_text().set_fontsize(tickFontsize)          

            if decimals==0:
                ax.set_yticklabels(np.int0(ax.get_yticks()),size=tickFontsize)                
            else:
                ax.set_yticklabels(np.round(ax.get_yticks(),
                                   decimals=decimals),size=tickFontsize)  
            
            if yaxisTick=='right':
                ax.yaxis.tick_right()
                
            if legends is not None:
                handles, labels = ax.get_legend_handles_labels() 
                l = ax.legend(handles[1:],labels[1:],loc=legends,
                              fontsize=legendFontsize)
                plt.savefig(os.path.join(GRAPH_DIR,
                            title.replace(' ','')+'.pdf'),
                            bbox_extra_artists=(l,), bbox_inches='tight')
            else:
                l = ax.legend()
                l.remove()
                plt.savefig(os.path.join(GRAPH_DIR,
                            title.replace(' ','')+'.pdf'),
                            bbox_inches='tight')
            plt.clf()
        else:
            sepCol = list(set(df[separate]))
            if vertical:
                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
            else:
                fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
            fig.set_size_inches(length,height)
            
            df1 = df.loc[df[separate]==sepCol[0]]
            df2 = df.loc[df[separate]==sepCol[1]]
            
            if hue is not None:
                if hue_order is None:
                    if order=='flex':
                        tmp1 = df1.groupby(hue)
                        tmp1 = (tmp1.tail(1).drop_duplicates()
                               .sort_values(y,ascending=False))
                        tmp2 = df2.groupby(hue)
                        tmp2 = (tmp2.tail(1).drop_duplicates()
                               .sort_values(y,ascending=False))
                    else:
                        tmp1 = df1.groupby(
                                hue,as_index=False).max().sort_values(hue)
                        tmp2 = df2.groupby(
                                hue,as_index=False).max().sort_values(hue)
                    hue_order1 = tmp1[hue].tolist()
                    hue_order2 = tmp2[hue].tolist()
            else:
                hue_order1 = None
                hue_order2 = None
            
            g1 = sns.lineplot(x=x,y=y,style=style,hue=hue,data=df1,ci=ci,
                        hue_order=hue_order1,style_order=hue_order1,ax=ax1)      
            g2 = sns.lineplot(x=x,y=y,style=style,hue=hue,data=df2,ci=ci,
                        hue_order=hue_order2,style_order=hue_order2,ax=ax2)
            
            ax1.set_yscale(yscale)
            ax2.set_yscale(yscale)
            ax1.set_xticklabels(np.int0(ax1.get_xticks()),
                                size=tickFontsize)
            ax2.set_xticklabels(np.int0(ax2.get_xticks()),
                                size=tickFontsize)
            ax1.set_yticklabels(np.round(ax1.get_yticks(),
                                decimals=decimals),size=tickFontsize)
            ax2.set_yticklabels(np.round(ax2.get_yticks(),
                                decimals=decimals),size=tickFontsize)
            ax1.xaxis.label.set_size(tickFontsize)
            ax2.xaxis.label.set_size(tickFontsize)
            ax1.yaxis.label.set_size(tickFontsize)
            ax2.yaxis.label.set_size(tickFontsize)

            if yaxisTick=='right':
                ax1.yaxis.tick_right()
                ax2.yaxis.tick_right()
            
            if legends is not None:
                handles1, labels1 = ax1.get_legend_handles_labels() 
                ax1.legend(handles1[1:],labels1[1:],loc=legends,
                           fontsize=legendFontsize)
                handles2, labels2 = ax2.get_legend_handles_labels() 
                ax2.legend(handles2[1:],labels2[1:],loc=legends,
                           fontsize=legendFontsize)            
            else:
                l = ax.legend()
                l.remove()
            plt.savefig(os.path.join(GRAPH_DIR,title.replace(' ','')+'.pdf'),
                        bbox_inches='tight')
            plt.clf()
          
    def _drawCdfFromKde(self,df,hue,target,style,title,
                        col=None,xlim=(0,1),loc=4,size=(10,4)):
        if col is None:
            plt.figure(figsize=(5,4))
            hue_order = list(set(df[hue]))
            hue_order.sort()
            for grp in hue_order:
                tmp = df.loc[df[hue]==grp,target]
                tmp = np.array(tmp)
                kde = gaussian_kde(tmp)
                cdf = np.vectorize(lambda x: kde.integrate_box_1d(-np.inf,x))
                x = np.linspace(xlim[0],xlim[1])
                plt.plot(x,cdf(x),linestyle=style[grp],label=grp)
            plt.legend(loc=loc,fontsize=15)
            plt.ylabel('CDF',fontsize=15)
            plt.xlabel(target,fontsize=15)
            print(target)
            plt.savefig(os.path.join(GRAPH_DIR,title.replace(' ','')+'.pdf'),
                        bbox_inches='tight')
            plt.clf()
        else:
            x = np.linspace(xlim[0],xlim[1])
            newDf = pd.DataFrame()
            for c in set(df[col]):
                for grp in set(df[hue]):
                    tmp = df.loc[(df[hue]==grp) & (df[col]==c),target]
                    tmp = np.array(tmp)
                    kde = gaussian_kde(tmp)
                    cdf = np.vectorize(
                            lambda y:kde.integrate_box_1d(-np.inf,y))
                    tmp0 = pd.DataFrame(np.vstack([x,cdf(x)]).transpose(),
                                         columns=[target,'CDF'])
                    tmp0[hue] = grp
                    tmp0[col] = c
                    if len(newDf)==0:
                        newDf = tmp0                        
                    else:
                        newDf = pd.concat([newDf,tmp0],axis=0)
            fig,ax = plt.subplots()
            ax = sns.FacetGrid(data=newDf,col=col,)
            ax.fig.set_size_inches(size[0],size[1])
            ax.map_dataframe(sns.lineplot,target,'CDF',hue,
                    style=hue,hue_order=list(style.keys()),
                    style_order=list(style.keys()),ci=None)
            ax.set(xlim=xlim)
            for axes in ax.axes.flat:
                axes.set_ylabel('CDF', fontsize=15)
                axes.set_xlabel(target, fontsize=15)
                axes.set_title(axes.get_title(),fontsize=15)
            handles, labels = ax.axes[0][-1].get_legend_handles_labels()
            l = ax.axes[0][-1].legend(handles[1:],labels[1:],
                                           loc=loc,fontsize=15)
            plt.savefig(os.path.join(GRAPH_DIR,title.replace(' ','')+'.pdf'),
                        bbox_extra_artists=(l,),bbox_inches='tight')
            plt.clf()
    
    def _drawBoxplot(self,df,x,y,title,hue=None,legends=3,ylabel=None,
                     legendFontsize=None,figsize=None,
                     myPalette=None,hue_order=None):
        if figsize is None:
            figsize = (5,4)
        defaultFontsize = 16
        if legendFontsize is None:
            legendFontsize = defaultFontsize
        if ylabel is None:
            ylabel = y
        
        sns.set_style('white')
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize)
        if myPalette is None:
            myPalette = {self.random:'C1',self.learned:'C0'}
        sns.boxplot(data=df,x=x,y=y,ax=ax,hue=hue,
                    showfliers=False,palette=myPalette,
            showmeans=True,meanprops={'marker':'o','markerfacecolor':'white',
                            'markeredgecolor':'white'},hue_order=hue_order)
        ax.set_xlabel(xlabel=x,fontsize=defaultFontsize)
        ax.set_ylabel(ylabel=ylabel,fontsize=defaultFontsize)
        if len(set(df[x]))>12:
            for ind, label in enumerate(ax.get_xticklabels()):
                if ind % 2 == 0:
                    label.set_visible(True)
                else:
                    label.set_visible(False)
        
        if legends is not None:
            handles, labels = ax.get_legend_handles_labels()
            l = ax.legend(handles,labels,loc=legends,fontsize=legendFontsize)
            plt.savefig(os.path.join(GRAPH_DIR,title.replace(' ','')+'.pdf'),
                    bbox_extra_artists=(l,), bbox_inches='tight')
        else:
            l = ax.legend()
            l.remove()
            plt.savefig(os.path.join(GRAPH_DIR,title.replace(' ','')+'.pdf'),
                        bbox_inches='tight')
        plt.clf()
    
    def _parse(self,value,sitetype):
        a = ast.literal_eval(value)
        values = {}
        for i,x in enumerate(a):
            site = 'site' + str(i)
            stype = sitetype[site]
            for key in x.keys():
                values[stype+'_'+site+'_'+str(key[1])] = x[key]
        return pd.Series(values)
    
    def _parseColumn(self,df,target,sitetype):
        result = df[target].apply(lambda x: self._parse(x,sitetype))
        col = result.columns
        col0 = [target + '_' + x for x in col]
        result.columns = col0
        return result
    
    def _getRegData(self,data,x,y):
        model = lambda x,a1,a2,a3,a4,a5,a6,a7: a1+a2*x+a3*x**2+a4*x**3+a5*x**4     
        mdl = model
        a,b = curve_fit(mdl,data[x],data[y])
        lst = np.array(data[x])
        pts = mdl(lst,*a)
        return pts
    
    def drawRegplot(self,df,x,y,title,style=None,hue=None,order='flex',
                      hue_order=None,legends=2,legendFontsize=None,
                      tickFontsize=None,size=None,separate=None,
                      x_decimals=1,y_decimals=1,linestyle=None,
                      dataRange=None,xticklabel=None):
        defaultFontsize = 15
        if tickFontsize is None:
            tickFontsize = defaultFontsize
        if legendFontsize is None:
            legendFontsize = defaultFontsize
        if size is None:
            length = 5
            height = 4
        else:
            length = size[0]
            height = size[1]
        if linestyle is None:
            linestyle = ['-','--']
        if dataRange is not None:
            try:
                df = df[(df[x]>=min(dataRange)) & 
                        (df[x]<=max(dataRange))]
            except:
                pass

        if separate is None:
            if hue is not None:
                if hue_order is None:
                    if order=='flex':
                        tmp = df.groupby(hue)
                        tmp = (tmp.tail(1).drop_duplicates()
                               .sort_values(y,ascending=False))
                    else:
                        tmp = df.groupby(
                                hue,as_index=False).max().sort_values(hue)
                    hue_order = tmp[hue].tolist()
            else:
                hue_order = None
            
            fig,ax = plt.subplots()
            fig.set_size_inches(length,height)
            
            for i,h in enumerate(hue_order):
                tmp = df[df[hue]==h]
                regData = self._getRegData(tmp,x,y)
                ax.scatter(x=tmp[x].values,y=tmp[y].values)
                ax.plot(tmp[x].values,regData,label=h,linestyle=linestyle[i]) 
        
            ax.set_xlabel(xlabel=x,fontsize=tickFontsize)
            ax.set_ylabel(ylabel=y,fontsize=tickFontsize)
            if dataRange is not None:
                ax.set_xticks(dataRange[0::2])
            if xticklabel is not None:
                xticklabel = xticklabel[0::2]
                ax.set_xticklabels(xticklabel)
            else:
                xticklabel = ax.get_xticks()
            
            if tickFontsize!=defaultFontsize:
                if x_decimals>0:
                    ax.set_xticklabels(np.round(xticklabel,
                                                decimals=x_decimals),
                                       size=tickFontsize)
                else:
                    ax.set_xticklabels(np.int0(xticklabel),
                                       size=tickFontsize)
                if y_decimals>0:
                    ax.set_yticklabels(np.round(ax.get_yticks(),
                                                decimals=x_decimals),
                                       size=tickFontsize)
                else:
                    ax.set_yticklabels(np.int0(ax.get_yticks()),
                                       size=tickFontsize)
            if legends is not None:
                handles, labels = ax.get_legend_handles_labels() 
                l = ax.legend(handles[0:],labels[0:],loc=legends,
                              fontsize=legendFontsize)
                plt.savefig(os.path.join(GRAPH_DIR,
                            title.replace(' ','')+'.pdf'),
                            bbox_extra_artists=(l,), bbox_inches='tight')
            else:
                l = ax.legend()
                l.remove()
                plt.savefig(os.path.join(GRAPH_DIR,
                            title.replace(' ','')+'.pdf'),
                            bbox_inches='tight')
            plt.clf()
        else:
            sepCol = list(set(df[separate]))
            sepCol.sort()
            
            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
            fig.set_size_inches(length,height)
            
            df1 = df.loc[df[separate]==sepCol[0]]
            df2 = df.loc[df[separate]==sepCol[1]]
            
            if hue is not None:
                if hue_order is None:
                    if order=='flex':
                        tmp1 = df1.groupby(hue)
                        tmp1 = (tmp1.tail(1).drop_duplicates()
                               .sort_values(y,ascending=False))
                        tmp2 = df2.groupby(hue)
                        tmp2 = (tmp2.tail(1).drop_duplicates()
                               .sort_values(y,ascending=False))
                    else:
                        tmp1 = df1.groupby(
                                hue,as_index=False).max().sort_values(hue)
                        tmp2 = df2.groupby(
                                hue,as_index=False).max().sort_values(hue)
                    hue_order1 = tmp1[hue].tolist()
                    hue_order2 = tmp2[hue].tolist()
            else:
                hue_order1 = None
                hue_order2 = None
            
            for i,sep in enumerate(sepCol):
                if i==0:
                    for j,h in enumerate(hue_order1):
                        tmp = df[(df[separate]==sep) & (df[hue]==h)]
                        regData = self._getRegData(tmp,x,y)
                        ax1.scatter(x=tmp[x].values,y=tmp[y].values,s=4)
                        ax1.plot(tmp[x].values,regData,
                                 label=h+', '+separate+'='+str(sep),
                                 linestyle=linestyle[j])
                else:
                    for j,h in enumerate(hue_order2):
                        tmp = df[(df[separate]==sep) & (df[hue]==h)]
                        regData = self._getRegData(tmp,x,y)
                        ax2.scatter(x=tmp[x].values,y=tmp[y].values,s=4)
                        ax2.plot(tmp[x].values,regData,
                                 label=h+', '+separate+'='+str(sep),
                                 linestyle=linestyle[j])

            ax1.set_ylabel(ylabel=y,fontsize=tickFontsize)
            ax2.set_ylabel(ylabel=y,fontsize=tickFontsize)
            
            ax2.set_xlabel(xlabel=x,fontsize=tickFontsize)
            if dataRange is not None:
                ax2.set_xticks(dataRange[0::2])
            if xticklabel is not None:
                xticklabel = xticklabel[0::2]
                ax2.set_xticklabels(xticklabel)
            else:
                xticklabel = ax2.get_xticks()
            
            if tickFontsize!=defaultFontsize:
                if x_decimals>0:
                    ax1.set_xticklabels(np.round(xticklabel,
                                                 decimals=x_decimals),
                                        size=tickFontsize)
                    ax2.set_xticklabels(np.round(xticklabel,
                                                 decimals=x_decimals),
                                        size=tickFontsize)
                else:
                    ax1.set_xticklabels(np.int0(xticklabel),
                                        size=tickFontsize)
                    ax2.set_xticklabels(np.int0(xticklabel),
                                        size=tickFontsize)
                if y_decimals>0:
                    ax1.set_yticklabels(np.round(ax1.get_yticks(),
                                                 decimals=x_decimals),
                                        size=tickFontsize)
                    ax2.set_yticklabels(np.round(ax2.get_yticks(),
                                                 decimals=x_decimals),
                                        size=tickFontsize)
                else:
                    ax1.set_yticklabels(np.int0(ax1.get_yticks()),
                                        size=tickFontsize)
                    ax2.set_yticklabels(np.int0(ax2.get_yticks()),
                                        size=tickFontsize)

                ax1.xaxis.label.set_size(tickFontsize)
                ax2.xaxis.label.set_size(tickFontsize)
                ax1.yaxis.label.set_size(tickFontsize)
                ax2.yaxis.label.set_size(tickFontsize)
            
            if legends is not None:
                handles1, labels1 = ax1.get_legend_handles_labels() 
                ax1.legend(handles1[0:],labels1[0:],loc=legends,
                           fontsize=legendFontsize)
                handles2, labels2 = ax2.get_legend_handles_labels() 
                ax2.legend(handles2[0:],labels2[0:],loc=legends,
                           fontsize=legendFontsize)            
            else:
                l = ax.legend()
                l.remove()
            plt.savefig(os.path.join(GRAPH_DIR,title.replace(' ','')+'.pdf'),
                        bbox_inches='tight')
            plt.clf()

    
    def drawPriceModelLoss(self,name='priceLearningModel'):
        data = self._collectData(name)
        data = data.loc[(~data[ch.ACTORLOSS].isnull()) & (data[ch.ACTORLOSS]<3) 
                        & (data[ch.ACTORLOSS]>-3)]
            
        for target in [ch.AVGREWARD,ch.CRITICLOSS,ch.ACTORLOSS]:
            title = name + '_' + target
            df = data[[ch.STEP,ch.NRSITES,target]].groupby(
                            [ch.STEP,ch.NRSITES],as_index=False).mean()
            self._drawLineplot(df=df,x=ch.STEP,y=target,
                               title=title,style=ch.NRSITES,hue=ch.NRSITES,
                               order='fix')  
            


