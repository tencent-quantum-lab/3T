import sys, os

def get_new_mol2(mol2_file, gro_file):
    a = [x for x in open(mol2_file,'r')]
    b = [x[:-1] for x in open(gro_file, 'r')]
    b = [x for x in b if 'LIG' in x]
    b = [[x for x in y.split(' ') if x != ''] for y in b]
    b = {x[1].rjust(4):(x[3], x[4],x[5]) for x in b}
    b = {k:['%.3f'%(float(y)*10) for y in v] for k,v in b.items()}
    idx_start = [i for i,x in enumerate(a) 
             if x.startswith('@<TRIPOS>ATOM')][0]
    idx_end = [i for i,x in enumerate(a) 
                 if x.startswith('@<TRIPOS>BOND')][0]

    block = a[idx_start+1:idx_end]
    block = [x.split('\t') for x in block]
    block = [x[:2] + b[x[1]] +x[5:] for x in block]
    block = ['\t'.join(x) for x in block]
    new_block = a[:idx_start+1]+block+a[idx_end:]
    return new_block    

def get_mol2_coord(mol2_file, gro_file, out_file):
    l = get_new_mol2(mol2_file, gro_file)
    f = open(out_file, 'w')
    for i in l:
        f.write(i)
    f.close()
