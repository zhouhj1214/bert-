#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 15:08:36 2019

@author: zhouhuanjian
"""
from bert_sim import BertSim
import sys

bs = BertSim(gpu_no=0, log_dir='log', bert_sim_dir='bert_sim_model\\', verbose=True)

def main(infile,outfile):
    fin = open(infile)
    fout = open(outfile,"wt")
    for line in fr:
        l = line.split("\t")
        lid = l[0]
        sen1 = l[1]
        sen2 = l[2]
        score = bs.predict([[text_a, text_b]])
        p1 = int(score.tolist()[0][0])
        p2 = int(score.tolist()[0][1])
        if p1>p2:
            s=1
        else:
            s=0
        out_line = lid+"\t"+str(s) +"\n"
        fout.write(out_line)
        fout.flush()
        
path1 = sys.argv[1]
path2 = sys.argv[2]

main(path1,path2)

            
        
        
