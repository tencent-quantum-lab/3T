import os,sys, glob, subprocess
import numpy as np
from pymol import cmd
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

def cal_RMSD(PATH, PDB, OUT_PATH, target, pocket, cocry = 0):
    # 先计算蛋白质部分
    PDB_ref = [x for x in glob.glob(f'{target}/pdb_processed/*.pdb') if PDB.lower() in x][0]
    LIG_ref = [x for x in glob.glob(f'{target}/cocry_dataset/*.pdb') if PDB.lower() in x][0]
    cmd.load(PDB_ref, f'{PDB}_pro')
    cmd.remove('(hydro)')
    cmd.select('ref', f'{PDB}_pro and resi {pocket} and backbone')
    
    files = sorted([x for x in os.listdir(PATH) if x.endswith('gro')])
    out = []
    for i in files:
        cmd.load(PATH+i, i)
        cmd.select('step_pro', f'{i} and polymer')
        cmd.save(PATH+i.replace('.gro', '_0.pdb'), 'step_pro')       
        cmd.select('step_lig', f'{i} and resn LIG')
        cmd.align(f'{i} and resi {pocket} and backbone', f'{PDB}_pro and resi {pocket} and backbone')
        cmd.save(PATH+i.replace('.gro', '.pdb'), 'step_pro')
        cmd.select('p_', f'step_pro and resi {pocket} and backbone')
        cmd.remove('(hydro)')
        
        b = cmd.rms('ref','p_')
        
        if cocry == 1:
            cmd.load(PATH+i.replace('.gro', '.mol2'),'step_lig_')
            cmd.remove('(hydro)')
            cmd.align('step_lig_', 'step_lig')
            cmd.save(PATH+i.replace('.gro', '_nonH.mol2'), 'step_lig_')
            ob = subprocess.Popen(['python', '-m', 'spyrmsd', 
                                   LIG_ref, PATH+i.replace('.gro', '_nonH.mol2')],
                                   stdin=subprocess.PIPE, 
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
            a = ob.stdout.read()
            try:
                a = float(a.strip())
            except:
                a = None
            cmd.delete('step_lig_')
            out.append([b]+[a])
        else:
            out.append([b])

        cmd.delete('step_lig')
        cmd.delete('step_pro')
        cmd.delete(i)
        
    cmd.delete("all")    
    dt = pd.DataFrame(out)
    dt.index = files

    if cocry == 1:
        dt.columns = ['POCKET_RMSD', 'LIGAND_RMSD']
        dt.to_csv(OUT_PATH+PATH.split('/')[-2]+'_RMSD_backbone.csv')
    else:
        dt.columns = ['POCKET_RMSD']
