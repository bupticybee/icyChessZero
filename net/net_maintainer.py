import urllib.request, urllib.parse, urllib.error
from urllib import parse,request
import os
import sys
#url = 'http://localhost:10087/model_get?name=model_2&model_f=data-00000-of-00001'
 
#print("downloading with urllib")
#urllib.request.urlretrieve(url, "code.zip")
 
#print("downloading with urllib2")
#f = urllib.request.urlopen(url)
#data = f.read()
#with open("dat", "wb") as code:
#    code.write(data)

class NetMatainer():
    def __init__(self,server,netdir,datanames = ['data-00000-of-00001','index','meta']):
        self.server = server
        self.netdir = netdir
        self.netname = None
        self.datanames = datanames
    
    def download_model(self,model_name):
        for model_f in self.datanames:
            url = '{}/model_get?name={}&model_f={}'.format(self.server,model_name,model_f)
            print("downloading with urllib2 {} ".format(model_name))
            f = urllib.request.urlopen(url)
            data = f.read()
            with open("{}/{}.{}".format(self.netdir,model_name,model_f), "wb") as code:
                code.write(data)
            print("download complete {}.{}".format(model_name,model_f))
    
    def get_update(self):
        url='{}/best_weight'.format(self.server)
        req = request.Request(url=url)
        res = request.urlopen(req)
        remote_model = res.read().strip()
        remote_model = str(remote_model,encoding='utf-8')
        if remote_model != self.get_latest():
            self.download_model(remote_model)
        else:
            print("best model downloaded already")
        return remote_model
        
    def get_latest(self):
        filelist = os.listdir(self.netdir)
        filelist = [i[:-6] for i in filelist if '.index' in i]
        if len(filelist) == 0:
            return None
        local_latest = sorted(filelist)[-1]
        return local_latest
        
    def updated(self,name):
        self.netname = name
        
    
