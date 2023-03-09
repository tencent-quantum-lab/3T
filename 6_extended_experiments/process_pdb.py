import pymol
from pymol import cmd
import itertools
from itertools import groupby

one_letter ={'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', \
'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y',    \
'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A',    \
'GLY':'G', 'PRO':'P', 'CYS':'C'}

def check_aa(l, one_letter):
    for i,j,_ in l:
        if j not in one_letter:
            print('{}{} is not in the lib.'.format(j,i))
            return False
    return True

def check_seq(l):
    try:
        start = int(l[0][0])
        end = int(l[-1][0])
        if len(l) != end -start + 1: 
            return False
        else:
            return True
    except:
        return False

def get_ss(l):
    a = [(x[0], x[2]) if x[2] != '' else 'L' for x in l]
    a_new = []
    for i,r in enumerate(a):
        if i > 1 and i < len(a):
            if (r[1] == 'S') or (r[1] == 'H'):
                if a[i-1][1] == 'L' and a[i+1][1] =='L':
                    a_new.append((r[0],'L'))
                else:
                    a_new.append(r)
            else:
                a_new.append(r)
        else:
            a_new.append(r)
    ss = [(i,len(''.join(g))) 
          for i, g in groupby([x[1] for x in a_new])]  
    a_index = [x[0] for x in a_new]
    ss_index = [0]+list(itertools.accumulate([x[1] for x in ss]))
    ss_new = []
    for  i,j in enumerate(ss):
        ss_new.append((j[0], a_index[ss_index[i]], a_index[ss_index[i+1]-1]))
    out = ['type,start,end\n']
    for i,j,k in ss_new:
        out.append('{},{},{}\n'.format(i,j,k))
    return out

def process_pdb(path, PDB, out_path):
    cmd.load(path, 'pro')
    
    cmd.save('{}{}.pdb'.format(out_path, PDB), 'pro')
    cmd.alter('pro and resn HIE', 'resn="HIS"')
    cmd.alter('pro and resn HID', 'resn="HIS"')
    cmd.alter('pro and resn CYX', 'resn="CYS"')
    cmd.dss('pro')
    ress = {'rrr':[]}
    cmd.iterate('pro and name CA',
                'rrr.append((resi, resn, ss))', space=ress)
    cmd.remove('hydro')
    if check_aa(ress['rrr'], one_letter):
        cmd.save('{}{}.pdb'.format(out_path, PDB), 'pro')
        out = get_ss(ress['rrr'])
        f = open('{}{}_secondary_fragment.txt'.format(out_path, PDB),'w')
        for i in out:
            f.write(i)
        f.close()
    cmd.delete('all')


    
