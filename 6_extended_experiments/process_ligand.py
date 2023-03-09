import pymol
from pymol import cmd
import sys, os
import rdkit
from rdkit import Chem
from rdkit.Chem.Lipinski import RotatableBondSmarts
import glob, subprocess

def H_adding(infile, outfile):
    cmd.load(infile, 'MOL')
    cmd.remove('hydrogens')
    cmd.h_add('MOL')
    cmd.select('NP', 'h. and (e. c extend 1)')
    cmd.remove('NP')
    cmd.save(outfile, 'MOL', format='sdf')
    cmd.delete('all')

def prelig4swiss(infile, outfile):
    outfile_0 = infile.split('.')[0]+'_0.mol2'
    cmd.load(infile, 'MOL')
    cmd.remove('hydrogens')
    cmd.h_add('MOL')
    cmd.save(outfile_0, 'MOL', format='mol2')
    cmd.delete('all')

    MOL_list = [x for x in open(outfile_0,'r')]
    idx = [i for i,x in enumerate(MOL_list) if x.startswith('@')]
    block = MOL_list[idx[1]+1:idx[2]]
    block = [x.split('\t') for x in block]

    block_new = []
    atom_count = {'C':1, 'N':1, 'O':1, 'S':1, 'P':1, 'F':1, 'Br':1, 'Cl':1, 'I': 1,   
                  'Li':1, 'Na':1, 'K':1, 'Mg':1, 'Al':1, 'Si':1,
                  'Ca':1, 'Cr':1, 'Mn':1, 'Fe':1, 'Co':1, 'Cu':1}
    for i in block:
        at = i[1].strip()
        if 'H' not in at:
            count = atom_count[at]
            atom_count[at]+=1
            at_new = at+str(count)
            at_new = at_new.rjust(4)
            block_new.append([i[0], at_new]+i[2:])
        else:
            block_new.append(i)

    block_new = ['\t'.join(x) for x in block_new]
    MOL_list_new = MOL_list[:idx[1]+1]+block_new+MOL_list[idx[2]:]
    f = open(outfile,'w')
    for i in MOL_list_new:
        f.write(i)
    f.close()

    
def get_rotatable_bond(infile, outfile):
    try:
        mol = Chem.rdmolfiles.MolFromMol2File(infile, removeHs=False)
        rot_atom_pairs = mol.GetSubstructMatches(RotatableBondSmarts)
        rot_atom_pairs = [(x[0]+1, x[1]+1) for x in rot_atom_pairs]
        f = open(outfile,'w')
        f.write('id,atom1,atom2,type\n')
        for i,(j,k) in enumerate(rot_atom_pairs):
            f.write('%d,%d,%d,1\n'%(i+1,j,k))
        f.close()
    except AttributeError:
        print(infile)

def set_temp_mol2(infile):
    l = [x for x in open(infile,'r')]
    # deal with the O.co2 problem in P=O
    l = [x.replace('O.co2', 'O.2') for x in l]
    f = open('temp.mol2','w')
    for i in l:
        f.write(i)
    f.close()
    
def batch_rotatable_bonds(PATH):
    if not PATH.endswith('/'):
        PATH+='/'
    files = sorted([x for x in glob.glob('%s*.mol2'%PATH)])
    for file in files:
        out = file.replace('.mol2','.rotbond')
        set_temp_mol2(file)
        get_rotatable_bond('temp.mol2', out)
        subprocess.call(['rm','temp.mol2'])