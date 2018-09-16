import tornado.ioloop
import shutil
import tornado.web
import argparse
import os
import sys
import numpy as np

currentpath = os.path.dirname(os.path.realpath(__file__))
project_basedir = os.path.join(currentpath,'..')
sys.path.append(project_basedir)

from config import conf
datadir = conf.distributed_datadir


validate_dirs = os.listdir(conf.distributed_server_weight_dir)
validate_dirs = [i.replace('.index','') for i in validate_dirs if i != '_blank' and '.index' in i and conf.noup_flag in i]
validate_dirs = sorted(validate_dirs)
validate_dirs = [os.path.join(conf.distributed_server_weight_dir,i) for i in validate_dirs]


one_dir = validate_dirs[-1]
one_date = one_dir.split('/')[-1].replace("_{}".format(conf.noup_flag),"")

one_noupweight = os.path.join(conf.distributed_server_weight_dir,one_date)
one_noupweight_up = "{}".format(one_noupweight)
one_noupweight_noup = "{}_{}".format(one_noupweight,conf.noup_flag)
print(one_noupweight,one_noupweight_up,one_noupweight_noup)
print(one_noupweight_up)
print(one_noupweight_noup)
if os.path.exists(one_noupweight_noup + '.index'):
    print("up weight")
    for f in ['data-00000-of-00001','meta','index']:
        src = one_noupweight_noup + '.' + f
        dst = one_noupweight_up + '.' + f
        print("copying file from {} to {}".format(src,dst))
        shutil.copyfile(src,dst)
else:
    print("No weight to up")
