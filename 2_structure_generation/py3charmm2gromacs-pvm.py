#!/usr/bin/python
"""
Script for parsing charmm force field to gromacs format

inparameters:
            command line parameters:
            1            charmm topology file
            2            corresponding charmm parameter file
            3    opt        foldername, default cgenff.ff

outfiles:
            1            foldername/atomtypes.atp
            2            foldername/forcefield.itp
            3            foldername/forcefield.doc
            4            foldername/aminoacids.rtp
            5            foldername/ffbonded.itp
            6            foldername/ffnonbonded.itp
            7            foldername/forcefield.r2b
            8    opt        foldername/lipids.rtp    (if '!lipid section' statement in CHARMM top file)
            9    opt        foldername/cmap.itp        (if genCMAP = True)
"""

### This is the Python 3 implementation of CHARMM to GROMACS force field format conversion code from:
### https://github.com/zyxue/gromacs_top_4.5.5/blob/master/cgenff-2b7.ff/src/charmm2gromacs-pvm.py

import sys
import math
import re
import os

#------------------
# System parameters
#------------------

# infiles
parFile = open(sys.argv[2], 'r')
topFile = open(sys.argv[1], 'r')
# test to see if there's a name specified
if len(sys.argv)>3:
    ffName = sys.argv[3]
    print('Creating '+ffName+' files...')
# if not, use cgenff
else:
    ffName = 'cgenff-2b7.ff'
    print('Creating '+ffName+' files...')

# conversion constant between kcal and kJ
kcal2kJ = 4.184

#-------------------------
# User specific parameters
#-------------------------

# specification of character used for comments in charmm ff file
comment = '!'

# create folder for output files
if not os.path.isdir(ffName):
    os.mkdir(ffName)
os.chdir(ffName)

# outfiles
nbFile = open('ffnonbonded.itp', 'w') # nonbonded file 
bonFile = open('ffbonded.itp', 'w') # bonded file
atpFile = open('atomtypes.atp', 'w') # atom type file
itpFile = open('forcefield.itp', 'w') # ffcharmm**.itp file
docFile = open('forcefield.doc', 'w') # ffcharmm**.itp file
aartpFile = open('aminoacids.rtp', 'w') # aminoacids rtp file

for line in topFile:
    if line.startswith('CMAP'):
        print("\n NOTE: This force field seems to support CMAP so trying to port it!\n")
        genCMAP = True
        cmapFile = open('cmap.itp', 'w') # cmap itp file
        break
    else:
        genCMAP = False
topFile.close()

topFile = open('../'+sys.argv[1], 'r')

# set the func parameter for bonds, angles and proper/improper dihedrals
# for further information see section 5.3.2 in gromacs documentation:
# ftp://ftp.gromacs.org/pub/manual/manual-3.3.pdf
funcForBonds = '1'
funcForAngles = '5' # Urey-Bradley angle type
funcForDihedrals = '9' # special type for treating multiple entries (modification in source code)
funcForImpropers = '2'
funcFor14 = '1' # function

# particle type
ptype = 'A'

# dictionary for atom numbers, used for the nb file
element2atomNumber= {}
element2atomNumber['H']='1'
element2atomNumber['HE']='2'
element2atomNumber['C']='6'
element2atomNumber['N']='7'
element2atomNumber['O']='8'
element2atomNumber['F']='9'
element2atomNumber['NE']='10'
element2atomNumber['NA']='11'
element2atomNumber['MG']='12'
element2atomNumber['AL']='13'
element2atomNumber['P']='15'
element2atomNumber['S']='16'
element2atomNumber['CL']='17'
element2atomNumber['K']='19'
element2atomNumber['CA']='20'
element2atomNumber['Fe']='26'
element2atomNumber['ZN']='30'
element2atomNumber['BR']='35'
element2atomNumber['I']='53'
element2atomNumber['CS']='55'


#-----------------------------------------------------------------------
# parsing the charmm top file and writing to gromacs .atp and .rtp files
#-----------------------------------------------------------------------

# position flags
mass = False
postMass = False
type2element = {}
element2mass = {}
type2charge = {}
firstBond = True
firstImpr = True
presFlag = False
lipidFlag = False
lipidFlagCounter = 0

# group counter
groupCounter = 0

# initiation of rtp file, defaults etc
aartpFile.write('[ bondedtypes ] \n')
aartpFile.write('; Col 1: Type of bond \n')
aartpFile.write('; Col 2: Type of angles \n')
aartpFile.write('; Col 3: Type of proper dihedrals \n')
aartpFile.write('; Col 4: Type of improper dihedrals \n')
aartpFile.write('; Col 5: Generate all dihedrals if 1, only heavy atoms of 0. \n')
aartpFile.write('; Col 6: Number of excluded neighbors for nonbonded interactions \n')
aartpFile.write('; Col 7: Generate 1,4 interactions between pairs of hydrogens if 1 \n')
aartpFile.write('; Col 8: Remove propers over the same bond as an improper if it is 1 \n')
aartpFile.write('; bonds  angles  dihedrals  impropers all_dihedrals nrexcl HH14 RemoveDih \n')
aartpFile.write('     1       5          9        2        1           3      1     0 \n')

# parse line by line
for line in topFile:

    # parse masses and write to gromacs atp file
    # match with MASS
    if line.startswith('MASS'):
        mass = True

    # match with something after MASS
    if line.startswith('DECL') or line.startswith('DEFA') or line.startswith('AUTO'):
        postMass = True

    if line.startswith('!lipid section'):
        lipidFlagCounter += 1
        # match for lipid section in CHARMM top file and create lipids rtp file
        if lipidFlagCounter == 2:
            lipidFlag = True
            lipidrtpFile = open('lipids.rtp', 'w') # lipids rtp file
            # initiation of rtp file, defaults etc
            lipidrtpFile.write('[ bondedtypes ] \n')
            lipidrtpFile.write('; Col 1: Type of bond \n')
            lipidrtpFile.write('; Col 2: Type of angles \n')
            lipidrtpFile.write('; Col 3: Type of proper dihedrals \n')
            lipidrtpFile.write('; Col 4: Type of improper dihedrals \n')
            lipidrtpFile.write('; Col 5: Generate all dihedrals if 1, only heavy atoms of 0. \n')
            lipidrtpFile.write('; Col 6: Number of excluded neighbors for nonbonded interactions \n')
            lipidrtpFile.write('; Col 7: Generate 1,4 interactions between pairs of hydrogens if 1 \n')
            lipidrtpFile.write('; Col 8: Remove propers over the same bond as an improper if it is 1 \n')
            lipidrtpFile.write('; bonds  angles  dihedrals  impropers all_dihedrals nrexcl HH14 RemoveDih \n')
            lipidrtpFile.write('     1       5          9        2        1           3      1     0 \n')
            
        
       
    # not empty line
    line = line.strip()
    if len(line) > 1:
        # mass part
        #-----------
        if mass and not postMass:
            segments = line.split() 

            # ignore comments
            if line[0] != comment:
                type = segments[2]
                mass = segments[3]
                if type[0] == 'H':
                    element = segments[4]    
                    element = 'H'
                elif type[0] == 'B' and type[1] == 'R':
                    element = 'BR'
                elif type[0] == 'C' and type[1] == 'L':
                    element = 'CL'
                elif type[0] == 'A' and type[1] == 'L':
                    element = 'AL'
                elif type[0] == 'C':
                    element = 'C'
                elif type[0] == 'N':
                    element = 'N'
                elif type[0] == 'O':
                    element = 'O'
                elif type[0] == 'S':
                    element = 'S'
                elif type[0] == 'F':
                    element = 'F'
                elif type[0] == 'P':
                    element = 'P'
                elif type[0] == 'I':
                    element = 'I'
                atomComment = segments[6:]
                # construct a string from the rest of the elements in the list
                string = ''
                for word in atomComment:
                    string = string + word + ' '
                # write to the .atp file
                atpFile.write(type+'\t'+            
                                mass+' '+
                                ';'+'\t'+
                                string+'\n')
                # set dictionaries:
                type2element[type]=element
                element2mass[element]=mass

    # parse the topologies of the charmm top file and write to gromacs rtp file
    #--------------------------------------------------------------------------
    # save the comments if available
    comments = line.split(' ! ')
        # delete comments in the end of the lines
    segments = line.split()
    line = ''
    for seg in segments:
        if seg == comment:
            break
        else:
            line = line+str(seg)+'\t'

    # new residues to parse
    if line.startswith('RESI'):
        # reset flags and group counter because this is a new residue
        presFlag = False # reset 
        firstBond = True # reset
        firstImpr = True # reset
        groupCounter = 0 # reset group counter since new residue
        # read and print name of residue
        name = line.split()[1]
        if not lipidFlag:
            aartpFile.write('\n; '+comments[1])
            aartpFile.write('\n[ '+name+' ]'+'\n')
            aartpFile.write(' [ atoms ]\n') # write header
        else:
            lipidrtpFile.write('\n[ '+name+' ]'+'\n')
            lipidrtpFile.write(' [ atoms ]\n') # write header        
    # set flag for pres entries, except for ACE and CT2 pres residues
    if line.startswith('PRES'):
        # find ACE and CT2 PRES residues, set presFlag = False for adding to rtp file
        name = line.split()[1]
        if name == 'ACE' or name == 'CT2':
            presFlag = True # reset 
            firstBond = True # reset
            firstImpr = True # reset
            groupCounter = 0 # reset group counter since new residue
            # read and print name of residue 
            if not presFlag:
                if not lipidFlag:
                    aartpFile.write('\n[ '+name+' ]'+'\n')
                    aartpFile.write(' [ atoms ]\n') # write header
                else:
                    lipidrtpFile.write('\n[ '+name+' ]'+'\n')
                    lipidrtpFile.write(' [ atoms ]\n') # write header
        else:
            presFlag = True
    # discard if entry is a pres
    if not presFlag:
        if line.startswith('GROUP'):
            #groupCounter += 1
            None
        if line.startswith('ATOM'):
            segments = line.split()
            name = segments[1]
            type = segments[2]
            charge = segments[3]
            try:
                type2charge[type]=charge
            except KeyError:
                None
            if not lipidFlag:
                aartpFile.write('\t'+name+'\t'+type+'\t'+charge+'\t'+str(groupCounter)+'\n')
            else:
                lipidrtpFile.write('\t'+name+'\t'+type+'\t'+charge+'\t'+str(groupCounter)+'\n')            
            groupCounter += 1

        if line.startswith('BOND') or line.startswith('DOUBLE'):
            # several bond statements...
            if firstBond:
                # write header
                if not lipidFlag:
                    aartpFile.write(' [ bonds ]\n')
                else:
                    lipidrtpFile.write(' [ bonds ]\n')
                segments = line.split()
                bondNumber = len(segments[1:])/2
                for i in range(bondNumber):
                    atom1 = segments[i+i+1]
                    atom2 = segments[i+i+2]
                    if not lipidFlag:
                        aartpFile.write('\t'+atom1+'\t'+atom2+'\n')
                    else:
                        lipidrtpFile.write('\t'+atom1+'\t'+atom2+'\n')                    
                firstBond = False
            else:
                segments = line.split()
                bondNumber = len(segments[1:])/2
                for i in range(bondNumber):
                    atom1 = segments[i+i+1]
                    atom2 = segments[i+i+2]
                    if not lipidFlag:
                        aartpFile.write('\t'+atom1+'\t'+atom2+'\n')
                    else:
                        lipidrtpFile.write('\t'+atom1+'\t'+atom2+'\n')
        if line.startswith('IMPR'):
            if firstImpr:
                # write header
                if not lipidFlag:
                    aartpFile.write(' [ impropers ]\n')
                else:
                    lipidrtpFile.write(' [ impropers ]\n')                    
                segments = line.split()
                imprNumber = len(segments[1:])/4
                for i in range(imprNumber):
                    atom1 = segments[4*i+1]
                    atom2 = segments[4*i+2]
                    atom3 = segments[4*i+3]
                    atom4 = segments[4*i+4]
                    if not lipidFlag:
                        aartpFile.write('\t'+atom1+'\t'+atom2+'\t'+atom3+'\t'+atom4+'\n')
                    else:
                        lipidrtpFile.write('\t'+atom1+'\t'+atom2+'\t'+atom3+'\t'+atom4+'\n')                        
                firstImpr = False
            else:
                segments = line.split()
                imprNumber = len(segments[1:])/4
                for i in range(imprNumber):
                    atom1 = segments[4*i+1]
                    atom2 = segments[4*i+2]
                    atom3 = segments[4*i+3]
                    atom4 = segments[4*i+4]
                    if not lipidFlag:
                        aartpFile.write('\t'+atom1+'\t'+atom2+'\t'+atom3+'\t'+atom4+'\n')
                    else:
                        lipidrtpFile.write('\t'+atom1+'\t'+atom2+'\t'+atom3+'\t'+atom4+'\n')                    
        # parse cmap foursomes
        if line.startswith('CMAP'):
            segments = line.split()
            if genCMAP:
                if not lipidFlag:
                    aartpFile.write(' [ cmap ]\n')
                    aartpFile.write('\t'+segments[1]+'\t'+segments[2]+'\t'+segments[3]+'\t'+segments[4]+'\t'+segments[8]+'\n')
                else:
                    lipidrtpFile.write(' [ cmap ]\n')
                    lipidrtpFile.write('\t'+segments[1]+'\t'+segments[2]+'\t'+segments[3]+'\t'+segments[4]+'\t'+segments[8]+'\n')                

#--------------------------------------------------------------------
# parsing the charmm par file and writing to gromacs bon and nb files
#--------------------------------------------------------------------

# position flags
bonds = False
angles = False
dihedrals = False
impropers = False
cmap = False
nonbonded = False
hbond = False

# lists for saving 1-4 and LJ params
paramList = []
LJlist = []

# cmap variables
if genCMAP:
    cmapType = 1
    cmapData = []
    cmapParamCounter = 0
    # matrix (2D list) for saving cmaps
    cmapValues = []

# initiation of temporary storage for dihedrals and impropers containing wildcards
dihWilds = []
impWilds = []

### parse charmm parFile

# parse line by line
for line in parFile:

    # match with BONDS
    if line.startswith('BONDS'):
        bonds = True
        # print header
        bonFile.write('[ bondtypes ]'+ '\n')
        bonFile.write('; i'+'\t'+
                        'j'+'\t'+
                        'func'+'\t'+
                        'b0'+'\t'+
                        'kb'+'\n')

    # match with ANGLES
    if line.startswith('ANGLES'):
        angles = True
        bonds = False

        # print header
        bonFile.write('\n'+'[ angletypes ]'+'\n')
        bonFile.write('; i'+'\t'+
                        'j'+'\t'+
                        'k'+'\t'+
                        'func'+'\t'+
                        'th0'+'\t'+
                        'cth'+'\t'+
                        'ub0'+'\t'+
                        'cub'+'\n')

    # match with (proper) DIHEDRALS
    if line.startswith('DIHEDRALS'):
        dihedrals = True
        angles = False
        
        # print header
        bonFile.write('\n'+'[ dihedraltypes ]'+'\n')
        bonFile.write('; i\tj\tk\tl\t'+
                        'func'+'\t'+
                        'phi0'+'\t'+            
                        'cp'+'\t'
                        'mult'+'\n')

    # match with IMPROPER (dihedrals)
    if line.startswith('IMPROPER'):
      
        ## print the dihedrals containing wildcards before!
        for wilds in dihWilds:
            cp = wilds[4]        
            # conversion to kJ
            cp = str(float(cp)*kcal2kJ) # not a factor 2!!!! 
            bonFile.write(wilds[0]+'\t'+wilds[1]+'\t'+wilds[2]+'\t'+wilds[3]+'\t'+funcForDihedrals+'\t'+wilds[6]+'\t'+cp+'\t'+wilds[5]+'\n')

        # continue with the impropers...
        impropers = True
        dihedrals = False
        # print header
        bonFile.write('\n'+'[ dihedraltypes ]'+'\n')
        bonFile.write('; i'+'\t'+
                        'j'+'\t'+
                        'k'+'\t'+
                        'l'+'\t'+
                        'func'+'\t'+
                        'q0'+'\t'+
                        'cq'+'\n')

    # match with CMAP
    if line.startswith('CMAP'):
        cmap = True
        impropers = False
        if genCMAP:
            cmapFile.write('[ cmaptypes ]')
        
    # match with NONBONDED
    if line.startswith('NONBONDED'):

        ## print the impropers containing wildcards
        for wilds in impWilds:
            cq = wilds[4]
            # converstion to kJ
            cq = str(float(cq)*2*kcal2kJ) # Need a factor 2 here too of course!!!
            bonFile.write(wilds[0]+'\t'+wilds[1]+'\t'+wilds[2]+'\t'+wilds[3]+'\t'+funcForImpropers+'\t'+wilds[6]+'\t'+cq+'\n')

        # continue with the nonbonded...
        nonbonded = True
        impropers = False

        # write last CMAP
        if genCMAP:
            cmapParamCounter = 0
            for i in range(len(cmapData)):
                cmapParamCounter = cmapParamCounter + 1
                if cmapParamCounter < 11:
                    cmapFile.write(str(cmapData[i]))
                    if not cmapParamCounter == 10:
                        cmapFile.write(' ')
                else:
                    cmapParamCounter = 0
                    cmapFile.write('\\'+'\n'+str(cmapData[i])+' ')
                    cmapParamCounter = 1

        cmap = False
        # print header
        nbFile.write('[ atomtypes ]'+'\n')
        nbFile.write(';name'+'\t'+
                     'at.num'+'\t'+
                     'mass'+'\t'+
                     'charge'+'\t'+
                     'ptype'+'\t'+
                     'sigma'+'\t'+
                     'epsilon'+'\n')

    # match with NBFIX
    if line.startswith('NBFIX'):
        nonbonded = False

    # match with HBOND => after NONBONDED -> 1-4 parameters
    if line.startswith('HBOND'):      
        hbond = True

        # write the header for the pairwise 1-4 parameters 
        nbFile.write('\n[ pairtypes ]'+'\n')
        nbFile.write('; i'+'\t'+
                     'j'+'\t'+
                     'func'+'\t'+
                     'sigma1-4'+'\t'+
                     'epsilon1-4 ; THESE ARE 1-4 INTERACTIONS\n')

    ###
    ### Write bonFile
    ###
        
    # not empty line
    line = line.strip()
    if len(line) > 0:
        
        # bonds part
        #-----------
        if bonds and not angles:
            segments = line.split()
            # ignore comments
            if line[0] != comment and segments[0] != 'BONDS':
                typei = segments[0]
                typej = segments[1]
                Kb = segments[2]
                # converstion from kcal/mole/A**2 -> kJ/mole/nm**2 incl factor 2 (see definitions)
                Kb = str(float(Kb)*2*kcal2kJ*10*10)
                b0 = segments[3]
                # conversion from A -> nm
                b0 = str(float(b0)/10)
                bonFile.write(typei+'\t'+
                              typej+'\t'+
                              funcForBonds+'\t'+
                              b0+'\t'+
                              Kb+'\n')

        # angles part, using Urey-Bradley type on all angles (type 5)
        #------------------------------------------------------------
        if angles and not dihedrals:
            segments = line.split() 
            # ignore comments and the first ANGLES line
            if line[0] != comment and segments[0] != 'ANGLES':
                typei = segments[0]
                typej = segments[1]
                typek = segments[2]
                th0 = segments [4]
                cth = segments [3]
                cth = str(float(cth)*2*kcal2kJ) # -> kJ/mol and an factor 2 (see definitions)
                # check for Urey-Bradley parameters
                if len(segments)>6:
                    try:
                        Kub = float(segments[5])*10*10
                        Kub = Kub*2*kcal2kJ
                        S0 = float(segments[6])
                        S0 = S0/10
                        ubFlag = True
                    except ValueError:
                        ubFlag = False
                else:
                    ubFlag = False
                    
                if not ubFlag:
                    Kub = 0.0
                    S0 = 0.0
                ## add comments also
                #lineComment = segments[6:]
                # construct a string from the rest of the elements in the list
                #string = ''
                #for word in lineComment:
                #    string = string + word + ' '
                
                bonFile.write(typei+'\t'+
                              typej+'\t'+
                              typek+'\t'+
                              funcForAngles+'\t'+
                              th0+'\t'+
                              cth+'\t'+
                              str(S0)+'\t'+
                              str(Kub)+'\n') #' ;'+
                              #string+'\n')

        # dihedrals part
        #---------------
        if dihedrals and not impropers:
            segments = line.split() 
            # ignore comments and the first DIHEDRALS line
            if line[0] != comment and segments[0] != 'DIHEDRALS':
                typei = segments[0]
                typej = segments[1]
                typek = segments[2]
                typel = segments[3]

                # look for wildcards in positions 1 and 4
                if typei == 'X' and typel == 'X':
                    dihWilds.append(segments) # save them in a list
                else:
                    phi0 = segments[6]
                    cp = segments[4]
                    # conversion to kJ
                    cp = str(float(cp)*kcal2kJ)
                    mult = segments[5]
                    bonFile.write(typei+'\t'+
                                    typej+'\t'+
                                    typek+'\t'+
                                    typel+'\t'+
                                    funcForDihedrals+'\t'+
                                    phi0+'\t'+
                                    cp+'\t'+
                                    mult+'\n')

        # impropers part
        #---------------
        if impropers and not nonbonded and not cmap:
            segments = line.split()

            # ignore comments and the first IMPROPERS line
            if line[0] != comment and segments[0] != 'IMPROPER':
                typei = segments[0]
                typej = segments[1]
                typek = segments[2]
                typel = segments[3]

                # look for wildcards in positions 2 and 3
                if typej == 'X' and typek == 'X':
                    impWilds.append(segments) # save them in a list
                else: # no wildcard - write to bon file
                    q0 = segments [6]
                    cq = segments [4]
                    # converstion to kJ
                    cq = str(float(cq)*2*kcal2kJ) # factor 2 from definition difference
                    bonFile.write(typei+'\t'+
                                    typej+'\t'+
                                    typek+'\t'+
                                    typel+'\t'+
                                    funcForImpropers+'\t'+
                                    q0+'\t'+
                                    cq+'\n')

        # cmap part (new part for the new version of gromacs supporting charmm cmaps)
        #----------------------------------------------------------------------------
        if genCMAP:
            if cmap and not nonbonded:
                segments = line.split()
                # discard lines starting with a comment
                if line[0]!= comment and segments[0]!='CMAP':
                    # find start of a surface
                    try: # if float cmap parameters start
                        segments[0] = float(segments[0])
                        for i in range(len(segments)):
                            cmapData.append(float(segments[i])*kcal2kJ)
                            #cmapFile.write(line+'\n')
                    except: # if not float it's a string and defines the cmap atom types
                            # write the parameters
                        cmapParamCounter = 0
                        for i in range(len(cmapData)):
                            cmapParamCounter = cmapParamCounter + 1
                            if cmapParamCounter < 11:
                                cmapFile.write(str(cmapData[i]))
                                if not cmapParamCounter == 10:
                                    cmapFile.write(' ')
                                   
                            else:
                                cmapParamCounter = 0
                                cmapFile.write('\\'+'\n'+str(cmapData[i])+' ')
                                cmapParamCounter = 1
                        # write the next CMAP header
                        cmapData = []
                        cmapFile.write('\n\n'+segments[0]+' '+segments[1]+' '+segments[2]+' '+segments[3]+' '+segments[7]+' '+str(cmapType)+' '+segments[-1]+' '+segments[-1]+'\\'+'\n')

#   IF COMMENTS !-180, !-164, etc WANTED
#            else:
#                # find phi = 0
#                try:
#                    phi = re.search('^!\s*(0)', line).group(1)
#                except:
#                    # find the other phi values
#                    try:
#                        phi = re.search('^!\s*(-*[0-9]{2,3})', line).group(1)
#                        cmapFile.write('!'+phi+'\n')
#                    except:
#                        None
        

        ###
        ### Write nbFile
        ###

        # nonbonded part: LJ parameters
        #------------------------------
        charge = '0.000' # charge for nb definitions
        if nonbonded and not hbond:
            segments = line.split()
            # ignore comments and the first NONBONDED lines
            if line[0]!=comment and segments[0]!='NONBONDED' and segments[0]!='cutnb' and segments[0]!='CUTNB':
                type = segments[0]
                epsilon = segments[2]
                eps = str(abs(float(epsilon)*kcal2kJ)) # ->kJ and positive
                RminHalf = segments[3]
                sigma = str(2*float(RminHalf)/(10.0*2.0**(1.0/6.0))) # -> nm, double distance and rmin2sigma factor
                LJlist.append([type, eps, sigma])
                # test length to avoid IndexError
                if len(segments)> 6: 
                    try: # if possible, convert element 5 to float 
                        segments[5] = float(segments[5])
                    except:
                        None

                    # is segment 5 and 6 floats => there's 1-4 defined
                    if not isinstance(segments[5],str): # not string?
                        # read charmm epsilon
                        epsilon14 = segments[5]
                        # conversion to gromacs units
                        eps14 = str(abs(float(epsilon14)*kcal2kJ))
                        # read charmm Rmin*1/2
                        Rmin14Half = segments[6]
                        # conversion to gromacs units
                        sigma14 = str(2*float(Rmin14Half)/(10.0*2.0**(1.0/6.0)))
                        
                        # add to list
                        paramList.append([type, eps14, sigma14])
                # test if partial charge is defined
                try:
                    charge = type2charge[type]
                    noChargeComment = ''
                except KeyError:
                    noChargeComment = '; partial charge def not found'

                # add special types of TIP3p model, in both with and without HEAVY_H
                noChargeComment = ''
                if type == 'HT':
                    nbFile.write('#ifdef HEAVY_H'+'\n')
                    nbFile.write(type+'\t'+
                                 element2atomNumber[type2element[type]]+'\t'+
                                 str(4*float(element2mass[type2element[type]]))+'\t'+
                                 charge+'\t'+
                                 ptype+'\t'+
                                 sigma+'\t'+
                                 eps+' '+noChargeComment+'; CHARMM TIP3p H\n')
                    nbFile.write('#else \n')
                    nbFile.write(type+'\t'+
                                 element2atomNumber[type2element[type]]+'\t'+
                                 element2mass[type2element[type]]+'\t'+
                                 charge+'\t'+
                                 ptype+'\t'+
                                 sigma+'\t'+
                                 eps+' '+noChargeComment+'\n')
                    nbFile.write('#endif'+'\n')
                else:
                    if type in type2element:
                        print(type, " ->", type2element)
                        nbFile.write(type+'\t'+
                                 element2atomNumber[type2element[type]]+'\t'+
                                 element2mass[type2element[type]]+'\t'+
                                 charge+'\t'+
                                 ptype+'\t'+
                                 sigma+'\t'+ 
                 eps + 
                 ' ' + 
#                 noChargeComment + 
                 '\n')

                if type == 'OT':
                    nbFile.write('#ifdef HEAVY_H'+'\n')
                    nbFile.write(type+'\t'+
                                 element2atomNumber[type2element[type]]+'\t'+
                                 '9.951400'+'\t'+
                                 charge+'\t'+
                                 ptype+'\t'+
                                 sigma+'\t'+
                                 eps+' '+noChargeComment+'; CHARMM TIP3p O\n')
                    nbFile.write('#endif'+'\n')


            
# nonbonded part: 1-4 parameters, where available
# NOTE: not in the for line in par loop since the 1-4 params are saved already
#------------------------------------------------------------------------------

# loop through all atom types with 1-4 parameters specified
for i in range(len(paramList)): # outer loop
    # look for other types with 1-4 parameters
    for j in range(i, len(paramList)): # inner loop
        # use combination rule 2 for epsilon: epsij=sqrt(epsi*epsj)
        combEps14 = math.sqrt(float(paramList[i][1])*float(paramList[j][1]))
        # use combination rule 2 for sigma
        combSigma14 = (float(paramList[i][2])+float(paramList[j][2]))/2.0
        # write pairs to [ pairtypes ] section
        nbFile.write(paramList[i][0]+'\t'+
                     paramList[j][0]+'\t'+
                     funcFor14+'\t'+
                     str(combSigma14)+'\t'+
                     str(combEps14)+'\n')
    # generate 1-4 params with non-1-4 specified types also
    for k in range(len(LJlist)):
        not14 = True
        # check if atom is present in 1-4 list
        for l in range(len(paramList)):
            if LJlist[k][0] == paramList[l][0]:
                not14 = False
        # if not calculate pair 1-4 interactions according to combination rules
        if not14:
            # use combination rule 2 for epsilon: epsij=sqrt(epsi*epsj)
            combEps14 = math.sqrt(float(paramList[i][1])*float(LJlist[k][1]))
            # use combination rule 2 for sigma
            combSigma14 = (float(paramList[i][2])+float(LJlist[k][2]))/2.0
            # write pairs to [ pairtypes ] section
            nbFile.write(paramList[i][0]+'\t'+
                         LJlist[k][0]+'\t'+
                         funcFor14+'\t'+
                         str(combSigma14)+'\t'+
                         str(combEps14)+'\n')

#-----------------------------------------
# construction of the forcefield.itp file
#-----------------------------------------

# '#' is for the c-preprocessor
itpFile.write('*******************************************************************************\n')
itpFile.write('*                    CHARMM port writted by                                   *\n')
itpFile.write('*                    Par Bjelkmar, Per Larsson, Michel Cuendet,               *\n')
itpFile.write('*                    Berk Hess and Erik Lindahl.                              *\n')
itpFile.write('*                    Correspondance:                                          *\n')
itpFile.write('*                    bjelkmar@cbr.su.se or lindahl@cbr.su.se                  *\n')
itpFile.write('*******************************************************************************\n\n\n') 
itpFile.write('#define _FF_CHARMM\n')
itpFile.write('[ defaults ]\n')
itpFile.write('; nbfunc\tcomb-rule\tgen-pairs\tfudgeLJ\tfudgeQQ\n')
itpFile.write('1\t2\tyes\t1.0\t1.0\n\n')
itpFile.write('#include "ffnonbonded.itp"\n')
itpFile.write('#include "ffbonded.itp"\n') 
itpFile.write('#include "gb.itp"\n')
if genCMAP:
    itpFile.write('#include "cmap.itp"\n')
itpFile.write('; Nucleic acids nonbonded and bonded parameters"\n')
itpFile.write('#include "ffnanonbonded.itp"\n') 
itpFile.write('#include "ffnabonded.itp"\n')


#-----------------------------------------
# construction of the forcefield.doc file
#-----------------------------------------

# '#' is for the c-preprocessor
docFile.write('CHARMM27 all-atom force field (with CMAP) - version 2.0\n\n')
docFile.write('*******************************************************************************\n')
docFile.write('*                    CHARMM port writted by                                   *\n')
docFile.write('*                    Par Bjelkmar, Per Larsson, Michel Cuendet,               *\n')
docFile.write('*                    Berk Hess and Erik Lindahl.                              *\n')
docFile.write('*                    Correspondance:                                          *\n')
docFile.write('*                    bjelkmar@cbr.su.se or lindahl@cbr.su.se                  *\n')
docFile.write('*******************************************************************************\n\n\n') 
docFile.write('Parameters derived from c32b1 version of CHARMM\n')
docFile.write('NOTE: Atom-based charge groups\n\n')
docFile.write('References:\n\n')
docFile.write('Proteins\n\n')
docFile.write('MacKerell, Jr., A. D., Feig, M., Brooks, C.L., III, Extending the\n')
docFile.write('treatment of backbone energetics in protein force fields: limitations\n')
docFile.write('of gas-phase quantum mechanics in reproducing protein conformational\n')
docFile.write('distributions in molecular dynamics simulations, Journal of\n')
docFile.write('Computational Chemistry, 25: 1400-1415, 2004.\n\n')
docFile.write('and \n\n')
docFile.write('MacKerell, Jr., A. D.,  et al. All-atom\n')
docFile.write('empirical potential for molecular modeling and dynamics Studies of\n')
docFile.write('proteins.  Journal of Physical Chemistry B, 1998, 102, 3586-3616.\n\n')
docFile.write('Lipids\n\n')
docFile.write('Feller, S. and MacKerell, Jr., A.D. An Improved Empirical Potential\n')
docFile.write('Energy Function for  Molecular Simulations of Phospholipids, Journal\n')
docFile.write('of Physical Chemistry B, 2000, 104: 7510-7515.\n\n')
docFile.write('Nucleic Acids\n\n')
docFile.write('Foloppe, N. and MacKerell, Jr., A.D. "All-Atom Empirical Force Field for\n')
docFile.write('Nucleic Acids: 2) Parameter Optimization Based on Small Molecule and\n')
docFile.write('Condensed Phase Macromolecular Target Data. 2000, 21: 86-104.\n\n')
docFile.write('and \n\n')
docFile.write('MacKerell, Jr., A.D. and Banavali, N. "All-Atom Empirical Force Field for\n')
docFile.write('Nucleic Acids: 2) Application to Molecular Dynamics Simulations of DNA\n')
docFile.write('and RNA in Solution.  2000, 21: 105-120.\n\n')
docFile.write('\n\n')
docFile.write('If using there parameters for research please cite:\n')
docFile.write('Bjelkmar, P., Larsson, P., Cuendet, M. A, Bess, B., Lindahl, E.\n')
docFile.write('Implementation of the CHARMM force field in GROMACS: Analysis of protein\n')
docFile.write('stability effects from correction maps, virtual interaction sites, and\n')
docFile.write('water models., Journal of Chemical Theory and Computation, 6: 459-466, 2010.\n')


# Close all files
nbFile.close()
bonFile.close()
atpFile.close()
itpFile.close()
docFile.close()
aartpFile.close()
if genCMAP:
    cmapFile.close()


