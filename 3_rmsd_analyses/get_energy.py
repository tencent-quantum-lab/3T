import os,sys, glob, subprocess
import numpy as np
import pandas as pd


def get_3T_energy(pdb, OUT_PATH, PREFIX, PDB):
    l_rmsd = sorted(list(set([x for x in glob.glob(f'{OUT_PATH}{pdb}_dock_{PDB.lower()}_*_RMSD_backbone.csv')])))
    idx = [x.split('/')[-1].split('_')[-3] for x in l_rmsd]
    l1 = [pd.read_csv(x, index_col = 0) for x in l_rmsd]
    l1_idx = [x.index.tolist() for x in l1]
    l1_idx = [[y.replace('.gro','')+'_'+str(i) for y in x] 
              for x,i in zip(l1_idx, idx)]
    
    for i,j in zip(l1, l1_idx):
        i.index = j

    energy_index = []
    l1_energy = sorted([x for x in glob.glob(f'{PREFIX}/*/') if pdb in x])
    for i in l1_energy:
        ll =[[float(y[:-1]) for y in open(x,'r')][-1] for x in glob.glob(i+'stepAll_*.txt')]
        energy_index.append(ll)
    energy_index = [pd.DataFrame(x) for x in energy_index]
    
    for x,y in zip(energy_index, l1):
        a = [z for z in y.index.tolist() if 'step2' in z]
        x.index = a
        x.columns = ['ENERGY']        
    
    l_ = pd.concat(energy_index, axis = 0)
    l_.to_csv(OUT_PATH+f'{pdb}_dock_{PDB}_energy.csv')  