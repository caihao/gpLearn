import time
import datetime

class LeftTime(object):
    def __init__(self):
        self.startTimeLoading=None
        self.loadLength=None
        self.currentLength=0
        
        self.startTimeTraining=None
        self.trainLength=None
        self.epoch=None
        self.currentEpoch=0
        self.currentIndex=0

        self.startTimeEpochTraining=None
        self.startTimeEpochTesting=None

        self.loadTime=0
        self.trainTimeList=[]
        self.testTimeList=[]

    def startLoading(self,loadLength:int):
        self.startTimeLoading=int(time.time())
        self.loadLength=loadLength

    def loadLeftTime(self,currentFinishLength:int,preSetLength:int):
        if self.startTimeLoading==None:
            return None
        ct=int(time.time())
        self.currentLength=self.currentLength+currentFinishLength
        self.loadLength=self.loadLength+currentFinishLength-preSetLength
        leftLength=self.loadLength-self.currentLength
        lt=(ct-self.startTimeLoading)*leftLength/self.currentLength
        completion="%.2f" % (self.currentLength/self.loadLength*100)
        return self.stamp_to_hms(lt),self.stamp_to_date(ct+lt),completion
    
    def endLoading(self):
        self.loadTime=int(time.time())-self.startTimeLoading
        self.startTimeLoading=None
    
    def startTraining(self,trainLength:int,epoch:list):
        self.startTimeTraining=int(time.time())
        self.trainLength=trainLength
        self.epoch=epoch
    
    def startEpochTraining(self):
        self.startTimeEpochTraining=int(time.time())
        self.currentEpoch=self.currentEpoch+1
        self.currentIndex=0

    def trainLeftTime(self,currentIndex:int):
        self.currentIndex=currentIndex
        ct=int(time.time())
        total_train_proportion=self.currentIndex/self.trainLength+self.currentEpoch-1
        total_train_time=ct-self.startTimeEpochTraining+sum(self.trainTimeList)
        lt_train=(self.epoch-self.currentEpoch+1-self.currentIndex/self.trainLength)*total_train_time/total_train_proportion

        if len(self.testTimeList)==0:
            lt_test=0
        else:
            total_test_proportion=self.currentEpoch-1
            total_test_time=sum(self.testTimeList)
            lt_test=(self.epoch-self.currentEpoch+1)*total_test_time/total_test_proportion

        lt=lt_train+lt_test

        total_completion="%.2f" % (total_train_proportion/self.epoch*100)
        epoch_completion="%.2f" % (self.currentIndex/self.trainLength*100)
        return 0 if lt_test!=0 else 1,self.stamp_to_hms(lt),self.stamp_to_date(ct+lt),total_completion,epoch_completion

    def endEpochTrainin(self):
        self.trainTimeList.append(int(time.time())-self.startTimeEpochTraining)
        self.startTimeEpochTraining=None
    
    def startEpochTesting(self):
        self.startTimeEpochTesting=int(time.time())

    def endEpochTesting(self):
        self.testTimeList.append(int(time.time())-self.startTimeEpochTesting)
        self.startTimeEpochTesting=None

    def endTraining(self):
        ct=int(time.time())
        total_time=ct-self.startTimeTraining+self.loadTime
        self.startTimeTraining=None
        return self.stamp_to_hms(total_time),self.stamp_to_date(ct)


    def stamp_to_hms(self,stamp:int):
        h=stamp//3600
        m=(stamp-3600*h)//60
        s=stamp-3600*h-60*m
        return str(int(h))+":"+str(int(m))+":"+str(int(s))
    def stamp_to_date(self,stamp:int):
        dt_obj=datetime.datetime.fromtimestamp(stamp)
        dt_str=dt_obj.strftime('%Y-%m-%d %H:%M:%S')
        return dt_str
