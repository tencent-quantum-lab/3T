import data_py3 as dt

class Fix_LAMMPS:
    def __init__(self,input_filename,data_filename,out_filename):
        pair_coeffs = self.extract_pair_coeffs(input_filename)
        raw_lmp = dt.data(data_filename)
        fixed_data = dt.data()
        fixed_data.title = raw_lmp.title
        fixed_data.headers = raw_lmp.headers
        raw_sections = self.split_strip_sections(raw_lmp)
        del raw_lmp
        raw_sections = self.group_molecules(raw_sections)
        raw_sections['Pair Coeffs'] = pair_coeffs
        del pair_coeffs
        fixed_data.sections = self.group_coeffs(raw_sections)
        del raw_sections
        fixed_data = self.fix_headers(fixed_data)
        fixed_data.write(out_filename)

    def fix_headers(self,fixed_data):
        fixed_data.headers['bond types'] = len(fixed_data.sections['Bond Coeffs'])
        fixed_data.headers['angle types'] = len(fixed_data.sections['Angle Coeffs'])
        fixed_data.headers['dihedral types'] = len(fixed_data.sections['Dihedral Coeffs'])
        fixed_data.headers['improper types'] = len(fixed_data.sections['Improper Coeffs'])
        return fixed_data
        
    def group_coeffs(self,raw_sections):
        def grouping_func(atoms,connections_raw,c_coeffs_raw):
            c_table = dict()
            c_coeffs = []
            index = 1
            for connection in connections_raw:
                # Determine the types of the atoms forming the connections
                atom_indices = [atoms[int(atom_index)-1][2] for atom_index in connection[2:]]
                key1 = ' '.join(atom_indices)
                atom_indices.reverse()
                key2 = ' '.join(atom_indices)
                # Determine the values of the connection coefficients
                c_index = int(connection[0])-1
                c_coeff = c_coeffs_raw[c_index]
                key3 = ' '.join(c_coeff[1:])
                # Check if the connection type is already in the table
                key1 += ' ' + key3
                key2 += ' ' + key3
                if not(key1 in c_table):
                    index_str = str(index)
                    c_table[key1] = index_str
                    c_table[key2] = index_str
                    connection[1] = index_str     # change type to the grouped version
                    #c_coeffs.append(' '.join([index_str,key3])+'\n')
                    c_coeffs.append(' '.join([index_str,key3]))
                    index += 1
                else:
                    connection[1] = c_table[key1]
            c_coeffs_raw = c_coeffs
            for i in range(len(connections_raw)):
                #connections_raw[i] = ' '.join(connections_raw[i]) + '\n'
                connections_raw[i] = ' '.join(connections_raw[i])
            return [connections_raw,c_coeffs,c_table]

        atoms = raw_sections['Atoms']
        [bonds,b_coeffs,b_table] = grouping_func(atoms,
                                                 raw_sections['Bonds'],
                                                 raw_sections['Bond Coeffs'])
        [angles,a_coeffs,a_table] = grouping_func(atoms,
                                                  raw_sections['Angles'],
                                                  raw_sections['Angle Coeffs'])
        [dihedrals,d_coeffs,d_table] = grouping_func(atoms,
                                                     raw_sections['Dihedrals'],
                                                     raw_sections['Dihedral Coeffs'])
        [impropers,i_coeffs,i_table] = grouping_func(atoms,
                                                     raw_sections['Impropers'],
                                                     raw_sections['Improper Coeffs'])

        for i in range(len(atoms)):
            #atoms[i] = ' '.join(atoms[i]) + '\n'
            atoms[i] = ' '.join(atoms[i])
        masses = raw_sections['Masses']
        for i in range(len(masses)):
            #masses[i] = ' '.join(masses[i]) + '\n'
            masses[i] = ' '.join(masses[i])
        raw_sections['Bonds'] = bonds
        raw_sections['Angles'] = angles
        raw_sections['Dihedrals'] = dihedrals
        raw_sections['Bond Coeffs'] = b_coeffs
        raw_sections['Angle Coeffs'] = a_coeffs
        raw_sections['Dihedral Coeffs'] = d_coeffs
        raw_sections['Improper Coeffs'] = i_coeffs
        return raw_sections
        
        
    def group_molecules(self,raw_sections):
        atoms = raw_sections['Atoms']
        bonds = raw_sections['Bonds']
        molecules = []

        # New edit for fixing InterMol atom indexing bug. Assume that InterMol Atoms section is ordered.
        atom_idx = 1 # First atom index for first Atoms line
        for atom in atoms:
            atom[0] = str(atom_idx) # Manually force sequential ordering of atom index
            atom_idx += 1

        # Start with bonded atoms
        for bond in bonds:
            atom1 = bond[2]
            atom2 = bond[3]
            ungrouped = True
            for molecule in molecules:
                if (atom1 in molecule) or (atom2 in molecule):
                    molecule[atom1] = atoms[int(atom1)-1][2]
                    molecule[atom2] = atoms[int(atom2)-1][2]
                    ungrouped = False
            if (ungrouped):
                molecule = dict()
                molecule[atom1] = atoms[int(atom1)-1][2]
                molecule[atom2] = atoms[int(atom2)-1][2]
                molecules.append(molecule)
        # Merge molecules containing identical atoms
        for mol1 in molecules:
            #print( 'Current: '+str(len(molecules))+' molecules' )
            # z = 0
            found_merge = True
            while (found_merge):
                found_merge = False
                for mol2 in molecules:
                    if (mol1 == mol2):
                        pass
                    else:
                        intersect = False
                        for atom2 in mol2.keys():
                            if atom2 in mol1:
                                intersect = True
                                break
                        if (intersect):
                            for atom2 in mol2.keys():
                                mol1[atom2] = mol2[atom2]
                            molecules.remove(mol2)
                            found_merge = True
                            # z += 1
                            break
            # print 'Reduce '+str(z)+' molecules'
        # Next do the non-bonded atoms
        for atom in atoms:
            atom1 = atom[0]
            ungrouped = True
            for molecule in molecules:
                if atom1 in molecule:
                    ungrouped = False
            if (ungrouped):
                molecule = dict()
                molecule[atom1] = atoms[int(atom1)-1][2]
                molecules.append(molecule)
        # print 'Current: '+str(len(molecules))+' molecules'

        # Map each atom to a molecule index
        # Sort such that cation is first, followed by anion, then solvent
        cation = []
        anion = []
        solvent = []
        for molecule in molecules:
            charge = 0.0
            for atom in molecule.keys():
                charge += float(atoms[int(atom)-1][3])
            if (charge > 0.1):
                cation.append(molecule)
            elif (charge < -0.1):
                anion.append(molecule)
            else:
                solvent.append(molecule)
        mol_table = dict()
        mol_id = 1
        self.N0 = len(solvent)
        #if not(len(cation)==len(anion)):
        #    raise Exception('Unmatched cation-anion molecule count')
        self.Nsalt = len(cation)
        for molecule in (cation+anion+solvent):
            for atom in molecule.keys():
                mol_table[atom] = str(mol_id)
            mol_id += 1

        # Update the molecule IDs
        for atom in atoms:
            atom[1] = mol_table[atom[0]]

##        mol_id = 1
##        species_dict = dict()
##        species_dict['4'] = 'Li'
##        species_dict['5'] = 'B'
##        species_dict['6'] = 'C'
##        species_dict['7'] = 'N'
##        for molecule in [cation[9],cation[19]]:
##            print 'Molecule '+str(mol_id)
##            for atom in molecule.keys():
##                atom_species = species_dict[ molecule[atom] ]
##                print atom_species +\
##                      ' ' + str(atoms[int(atom)-1][4]) +\
##                      ' ' + str(atoms[int(atom)-1][5]) +\
##                      ' ' + str(atoms[int(atom)-1][6])
##            mol_id += 1
##        for molecule in [anion[0],anion[14]]:
##            print 'Molecule '+str(mol_id)
##            for atom in molecule.keys():
##                atom_species = species_dict[ molecule[atom] ]
##                print atom_species +\
##                      ' ' + str(atoms[int(atom)-1][4]) +\
##                      ' ' + str(atoms[int(atom)-1][5]) +\
##                      ' ' + str(atoms[int(atom)-1][6])
##            mol_id += 1
        
        return raw_sections
        

    def split_strip_sections(self,raw_lmp):
        sections = dict()
        sections['Masses'] = self.split_words('Masses',raw_lmp)
        sections['Atoms'] = self.split_words('Atoms',raw_lmp)
        sections['Bonds'] = self.split_words('Bonds',raw_lmp)
        sections['Angles'] = self.split_words('Angles',raw_lmp)
        sections['Dihedrals'] = self.split_words('Dihedrals',raw_lmp)
        sections['Impropers'] = self.split_words('Impropers',raw_lmp)
        sections['Bond Coeffs'] = self.strip_first_word('Bond Coeffs',raw_lmp)
        sections['Angle Coeffs'] = self.strip_first_word('Angle Coeffs',raw_lmp)
        sections['Dihedral Coeffs'] = self.strip_first_word('Dihedral Coeffs',raw_lmp)
        sections['Improper Coeffs'] = self.strip_first_word('Improper Coeffs',raw_lmp)
        return sections
        
    def split_words(self,keyword,raw_lmp):
        lines = raw_lmp.sections[keyword]
        new_lines = []
        for i in range(len(lines)):
            #words = str.split(lines[i])
            words = str.split( lines[i].strip() )
            new_lines.append(words)
        return new_lines

    def strip_first_word(self,keyword,raw_lmp):
        lines = raw_lmp.sections[keyword]
        new_lines = []
        for i in range(len(lines)):
            #words = str.split(lines[i])
            words = str.split( lines[i].strip() )
            words.pop(1)
            new_lines.append(words)
        return new_lines

    def extract_pair_coeffs(self,input_filename):
        pair_coeffs = []
        input_file = open(input_filename,'r')
        line = input_file.readline()
        while (line):
            words = str.split(line)
            if (len(words)>0):
                if (words[0]=='pair_coeff'):
                    #pair_coeffs.append(' '.join(words[2:]) + '\n')
                    pair_coeffs.append(' '.join(words[2:]))
            line = input_file.readline()
        input_file.close()
        return pair_coeffs


#dat = Fix_LAMMPS('../gromacs_example_3/complex_converted.input',
#                              '../gromacs_example_3/complex_converted.lmp',
#                              'testonly.dat')

#dat = Fix_LAMMPS('MD_GPU150_converted.input','MD_GPU150_converted.lmp','MD_GPU150.data')

##dat = dt.data('MD_GPU150_converted.lmp')
##
##lines = dat.sections['Bond Coeffs']
##new_lines = []
##for i in range(len(lines)):
##    words = str.split(lines[i])
##    words.pop(1)
##    new_line = " ".join(words) + '\n'
##    new_lines.append(new_line)
##dat.sections['Bond Coeffs'] = new_lines
##
##lines = dat.sections['Angle Coeffs']
##new_lines = []
##for i in range(len(lines)):
##    words = str.split(lines[i])
##    words.pop(1)
##    new_line = " ".join(words) + '\n'
##    new_lines.append(new_line)
##dat.sections['Angle Coeffs'] = new_lines
##
##lines = dat.sections['Dihedral Coeffs']
##new_lines = []
##for i in range(len(lines)):
##    words = str.split(lines[i])
##    words.pop(1)
##    new_line = " ".join(words) + '\n'
##    new_lines.append(new_line)
##dat.sections['Dihedral Coeffs'] = new_lines
##
##output = dt.data()
##output.title = dat.title
##output.headers = dat.headers
##output.sections = dat.sections
##output.write('MD_CPU150_postprocess.txt')
##    
##
