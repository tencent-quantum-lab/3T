import os, sys

class gro_parser(object):
    def __init__(self, infile):
        super(gro_parser, self).__init__()
        gro_file = [x for x in open(infile,'r')]
        self.header = gro_file[:2]
        self.coords = gro_file[2:-1]
        self.tail = gro_file[-1]
    
    def renew_coords(self, new_coords):
        try:
            assert len(new_coords) == len(self.coords)
            n_co = []
            for i, coord in enumerate(self.coords):
                x = coord[:20]
                y = new_coords[i]
                n_co.append(x+y+'\n')
            return self.header+n_co+[self.tail]
        except AssertionError: 
            print('[%d] not equals [%d]'%(len(new_coords), len(self.coords)))
            
def get_cif_coords(infile):
    cif_file = [x for x in open(infile,'r')]
    cif_file_coords = [x for x in cif_file if x.startswith(' ') and (not x.startswith('  _'))]
    cif_file_coords = [x.split(' ') for x in cif_file_coords]
    cif_file_coords = [[y for y in x if y != ''] for x in cif_file_coords]
    cif_file_coords = [x[3:6] for x in cif_file_coords]
    cif_file_coords = [['%8.3f'%(float(y)/10) for y in x] for x in cif_file_coords]
    cif_file_coords = [''.join(x) for x in cif_file_coords]
    return cif_file_coords

def cif2gro(gro_file, cif_file, outfile):
    gro = gro_parser(gro_file)
    cif_file_coords = get_cif_coords(cif_file)
    cif = gro.renew_coords(cif_file_coords)
    f = open(outfile,'w')
    for i in cif:
        f.write(i)
    f.close()