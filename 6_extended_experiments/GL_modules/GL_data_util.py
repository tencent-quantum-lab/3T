import numpy as np
import time

def _grab_bonded_atom(bonded_list, possible_list, bond_idx):
    expanded_list = dict() # this is the idx of all atoms bonded with atoms in bonded_list
    for bonded_idx in bonded_list:
        possible_idx_0 = np.where(bond_idx[:,0] == bonded_idx)[0]
        possible_idx_1 = np.where(bond_idx[:,1] == bonded_idx)[0]
        for i in np.concatenate([possible_idx_0,possible_idx_1]):
            expanded_list[i] = True
    valid_possible_list = []
    for possible_idx in possible_list:
        if possible_idx in expanded_list:
            valid_possible_list.append(possible_idx)
    if len(valid_possible_list) == 0:
        # no additional atom in the possible_list is bonded to any atom in bonded_list
        return bonded_list
    else:
        temp = dict()
        new_possible_list = possible_list.copy()
        for i in bonded_list:
            temp[i] = True
        for i in valid_possible_list:
            temp[i] = True
            new_possible_list.remove(i)
        new_bonded_list = list(temp.keys())
        new_bonded_list.sort()
        return _grab_bonded_atom(new_bonded_list, new_possible_list, bond_idx)
    

def segmentize_nearby_protein_atoms(gl_data, ligand_idx, dist_cutoff):
    # sanity check
    temp = dict()
    for i in ligand_idx:
        if i in temp: raise Exception('Atom index '+str(i)+' appear in ligand_idx multiple times')
        temp[i] = True
    bond_idx = gl_data.bond_idx
    all_pos = gl_data.atom_pos
    atom_molid = gl_data.atom_molid
    na = all_pos.shape[0]
    protein_idx = []
    for i in range(na):
        if not i in ligand_idx:
            protein_idx.append(i)
    n_p, n_l = len(protein_idx), len(ligand_idx)
    ligand_pos = all_pos[ ligand_idx, : ]
    protein_pos = all_pos[ protein_idx, : ]
    p_l_dist = np.linalg.norm( np.expand_dims(protein_pos,axis=1).repeat(n_l, axis=1) - \
                                             np.expand_dims(ligand_pos,axis=0).repeat(n_p, axis=0),
                                             axis=2 )
    p_l_dist = np.min(p_l_dist, axis=1) # min distance from a protein atom to any ligand atom
    p_idx = np.where(p_l_dist < dist_cutoff)[0]
    close_protein_idx = np.array(protein_idx)[ p_idx ]
    close_molid_atoms = dict()  # molid -> atom_idx, but only for the atoms within cutoff
    for i in close_protein_idx:
        temp = atom_molid[i]
        if temp in close_molid_atoms:
            close_molid_atoms[ temp ].append(i)
        else:    
            close_molid_atoms[ temp ] = [i]
    
    all_molid_atoms = dict()
    for i in close_molid_atoms:
        all_molid_atoms[i] = np.where(atom_molid == i)[0].tolist()
        all_molid_atoms[i].sort()
    
##    segments_out = []
##    for i in close_molid_atoms:
##        close_atom_list = _grab_bonded_atom(close_molid_atoms[i], all_molid_atoms[i], bond_idx)
##        segments_out.append( close_atom_list )

    segments_out = []
    for i in all_molid_atoms:
        segments_out.append( all_molid_atoms[i] )

    return segments_out

def extract_ligand_idx(gl_data):
    atom_molid = gl_data.atom_molid
    ligand_molid = np.max(atom_molid)
    ligand_idx = np.where(atom_molid == ligand_molid)[0]
    return ligand_idx

def segmentize_complex_ligand(gl_complex_data, gl_ligand_data, gl_rotbond_file):
    start = time.time()
    rotbonds, allbonds = dict(), dict()
    with open(gl_rotbond_file,'r') as f:
        line = f.readline()
        line = f.readline()
        while line:
            bond_idx = [int(i)-1 for i in line.strip().split(',')[1:3]]
            if bond_idx[0] in rotbonds: rotbonds[ bond_idx[0] ].append( bond_idx[1] )
            else: rotbonds[ bond_idx[0] ] = [ bond_idx[1] ]
            if bond_idx[1] in rotbonds: rotbonds[ bond_idx[1] ].append( bond_idx[0] )
            else: rotbonds[ bond_idx[1] ] = [ bond_idx[0] ]
            line = f.readline()
    lig_bond_idx = gl_ligand_data.bond_idx
    for bond in lig_bond_idx:
        if bond[0] in allbonds:   allbonds[bond[0]] += [bond[1]]
        else:                              allbonds[bond[0]] = [bond[1]]
        if bond[1] in allbonds:   allbonds[bond[1]] += [bond[0]]
        else:                              allbonds[bond[1]] = [bond[0]]
    #print('Rotbonds :',rotbonds)
    #print('Allbonds :',allbonds)
    lig_atoms = [i for i in range(gl_ligand_data.atom_molid.shape[0])]
    segments, current_segment = [], []
    while True:
        now = time.time()
        if now - start > 300:
            raise Exception('Infinite loop in ligand segmentation algorithm, check this edge case')
        if len(current_segment) == 0:
            center = lig_atoms.pop(0)
            current_segment.append(center)
        else:
            temp = current_segment.copy()
            restricted = []
            for atom in current_segment:
                if atom in rotbonds: restricted += rotbonds[atom] 
            for atom in current_segment:
                segment_nbrs = [i for i in allbonds[atom] if not (i in restricted)]
                temp += segment_nbrs
            current_segment = np.unique(np.array(temp)).tolist()
            remove_count = 0
            for atom in current_segment:
                if atom in lig_atoms:
                    lig_atoms.remove(atom)
                    remove_count += 1
            if remove_count == 0:
                segments.append( current_segment.copy() )
                current_segment = []
        #print('All segments :',segments)
        #print('Current segment :',current_segment)        
        if len(lig_atoms) == 0:
            if len(current_segment) > 0:
                segments.append( current_segment.copy() )
                current_segment = []
            break
    for segment in segments:
        segment.sort()
    #print('Segments :',segments)
    n_atoms = gl_ligand_data.atom_molid.shape[0]
    check = []
    for segment in segments: check += segment
    check = np.unique(np.array(check))
    if len(check) != n_atoms:
        raise Exception('Mistake in ligand segmentation algorithm, check this edge case')

    # Now that we have segmentize the individual ligand, we have to transfer this segmentation to
    # the ligand in the protein-ligand complex
    pl_ligand_idx = extract_ligand_idx(gl_complex_data)
    pl_ligand_idx.sort()
    lig_segments = []
    for segment in segments:
        lig_segment = []
        for idx in segment:
            lig_segment.append( pl_ligand_idx[idx] )
        lig_segments.append(lig_segment)
    #print('Complex ligand segments :',lig_segments)
    #raise Exception('Stop')
    return lig_segments
    
def separate_protein_backbone_sidechain(gl_complex_data, complex_gro):
    # This replacement/separation algorithm only works correctly if the complex.gro
    # and LAMMPS file atoms are sequenced based on amino acid / ligand ordering
    ref = ['N','C','O','CA', 'HN','H1','H2', 'H3','OT1','OT2', 'HA', 'HA1', 'HA2']
    new_molid = []
    current_molid = 0
    molid_dict = dict()
    current_molname = None
    lines = [x for x in open(complex_gro,'r')]
    lines = lines[1:]
    lines = [x.strip().split() for x in lines]
    lines = [x for x in lines if len(x) == 6]
    for words in lines:
        if len(new_molid) == 0:
            current_molname = words[0]
        if words[0] != current_molname:
            current_molname = words[0]
            current_molid = max([molid_dict[x] for x in molid_dict]) + 1
            molid_dict = dict()
        if 'LIG' in words[0]:
            new_molid.append(current_molid)
        else:
            if len(molid_dict) == 0:
                if words[1] in ref:
                    molid_dict['backbone'] = current_molid
                else:
                    molid_dict['sidechain'] = current_molid
            elif len(molid_dict) == 1:
                if (words[1] in ref) and (not 'backbone' in molid_dict):
                    molid_dict['backbone'] = current_molid + 1
                elif (not words[1] in ref) and (not 'sidechain' in molid_dict):
                    molid_dict['sidechain'] = current_molid + 1
            if words[1] in ref:
                new_molid.append( molid_dict['backbone'] )
            else:
                new_molid.append( molid_dict['sidechain'] )
    assert len(new_molid) == gl_complex_data.atom_molid.shape[0]

    gl_complex_data.atom_molid = np.array(new_molid)
    return
    
def segmentize_protein_atoms_from_center(gl_data, ligand_idx, center_pos, dist_cutoff):
    # sanity check
    temp = dict()
    for i in ligand_idx:
        if i in temp: raise Exception('Atom index '+str(i)+' appear in ligand_idx multiple times')
        temp[i] = True
    bond_idx = gl_data.bond_idx
    all_pos = gl_data.atom_pos
    atom_molid = gl_data.atom_molid
    na = all_pos.shape[0]
    protein_idx = []
    for i in range(na):
        if not i in ligand_idx:
            protein_idx.append(i)
    n_p = len(protein_idx)
    protein_pos = all_pos[ protein_idx, : ]
    center_pos = np.array(center_pos)
    p_c_dist = np.linalg.norm( protein_pos-center_pos, axis=1 )
    p_idx = np.where(p_c_dist < dist_cutoff)[0]
    close_protein_idx = np.array(protein_idx)[ p_idx ]
    close_molid_atoms = dict()  # molid -> atom_idx, but only for the atoms within cutoff
    for i in close_protein_idx:
        temp = atom_molid[i]
        if temp in close_molid_atoms:
            close_molid_atoms[ temp ].append(i)
        else:    
            close_molid_atoms[ temp ] = [i]
    
    all_molid_atoms = dict()
    for i in close_molid_atoms:
        all_molid_atoms[i] = np.where(atom_molid == i)[0].tolist()
        all_molid_atoms[i].sort()
    
##    segments_out = []
##    for i in close_molid_atoms:
##        close_atom_list = _grab_bonded_atom(close_molid_atoms[i], all_molid_atoms[i], bond_idx)
##        segments_out.append( close_atom_list )

    segments_out = []
    for i in all_molid_atoms:
        segments_out.append( all_molid_atoms[i] )

    return segments_out

def generate_special_rotation_centers(gl_data, movable_idx_list):
    special_rotation = dict()
    bond_idx = gl_data.bond_idx
    all_movables = []
    for movable_idx in movable_idx_list:
        all_movables += movable_idx
    for group_id, movable_idx in enumerate(movable_idx_list):
        # Find groups which are bonded to at most one other movable group
        group_bonds = []
        for atom_idx in movable_idx:
            pair1 = bond_idx[np.where(bond_idx[:,0]==atom_idx)[0], 1]
            pair2 = bond_idx[np.where(bond_idx[:,1]==atom_idx)[0], 0]
            group_bonds += pair1.tolist() + pair2.tolist()
        group_bonds = [i for i in group_bonds if (not i in movable_idx)]
        group_bonds = list(set(group_bonds))    # make unique
        # We generate special rotation center if and only if there is only 1 anchor atom for this group 
        if len(group_bonds) == 1:
            special_rotation[group_id] = group_bonds[0]
    # If there is no special rotation, we should return None
    if len(special_rotation)==0:
        return None
    else:
        return special_rotation

def generate_special_rotation_axes(gl_data, movable_idx_list, complex_gro=None):
    if complex_gro != None:
        lines = [x for x in open(complex_gro,'r')]
        lines = lines[1:]
        lines = [x.strip().split() for x in lines]
        lines = [x for x in lines if len(x) == 6]
        atom_group_name = [x[0][-3:] for x in lines]
        deeptrap_group_name = ['HIS', 'PHE', 'TRP', 'TYR']
    
    special_rotation = dict()
    bond_idx = gl_data.bond_idx
    all_movables = []
    for movable_idx in movable_idx_list:
        all_movables += movable_idx
    for group_id, movable_idx in enumerate(movable_idx_list):
        # Find groups which are bonded to at most one other movable group
        group_bonds = []
        for atom_idx in movable_idx:
            pair1 = bond_idx[np.where(bond_idx[:,0]==atom_idx)[0], 1]
            pair2 = bond_idx[np.where(bond_idx[:,1]==atom_idx)[0], 0]
            group_bonds += pair1.tolist() + pair2.tolist()
        group_bonds = [i for i in group_bonds if (not i in movable_idx)]
        group_bonds = list(set(group_bonds))    # make unique
        # We generate special rotation center if and only if there is only 1 anchor atom for this group 
        if len(group_bonds) == 1:
            out_center = group_bonds[0]
            pair1 = bond_idx[np.where(bond_idx[:,0]==out_center)[0], 1]
            pair2 = bond_idx[np.where(bond_idx[:,1]==out_center)[0], 0]
            out_center_bonds = pair1.tolist() + pair2.tolist()
            out_center_bonds = [i for i in out_center_bonds if (not i == out_center)]
            out_center_bonds = list(set(out_center_bonds))  # make unique
            in_bonds = [i for i in out_center_bonds if (i in movable_idx)]
            if len(in_bonds) != 1:
                raise Exception('Special rotation axes error, center = '+str(out_center)+', group bonds = '+str(in_bonds)+', group idx = '+str(movable_idx))
            in_center = in_bonds[0]
            if complex_gro == None:
                special_rotation[group_id] = [out_center, in_center, 0]
            elif atom_group_name[in_center] in deeptrap_group_name:
                group_pos = gl_data.atom_pos[movable_idx,:]
                A = gl_data.atom_pos[out_center,:]
                B = gl_data.atom_pos[in_center,:]
                u = (B-A)/np.linalg.norm(B-A)
                R = group_pos-A
                Z = np.sum(R*u, axis=1).reshape(-1,1)*u
                rot_pos = A + Z - (R-Z)
                other_idx = [j for j in range(gl_data.atom_pos.shape[0]) if (not j in movable_idx)]
                other_pos = gl_data.atom_pos[other_idx,:]
                all_dist = np.linalg.norm( np.expand_dims(other_pos,axis=1).repeat(rot_pos.shape[0],axis=1) -\
                                                        np.expand_dims(rot_pos,axis=0).repeat(other_pos.shape[0],axis=0),
                                                        axis=2)
                min_dist = np.min(all_dist)
                if min_dist < 1.0:
                    special_rotation[group_id] = [out_center, in_center, 0]
                else:
                    special_rotation[group_id] = [out_center, in_center, 1]
            else:
                special_rotation[group_id] = [out_center, in_center, 0]
    # If there is no special rotation, we should return None
    if len(special_rotation)==0:
        return None
    else:
        return special_rotation

def generate_macro_mode(gl_data, movable_idx_list, complex_gro, complex_secondary):
    # name will be int(1) from '1MET', or 'LIG' from '1LIG'
    name_order_dict = dict()
    with open(complex_secondary,'r') as f:
        line = f.readline()
        line = f.readline()
        order = 0
        while line:
            g_type, start, end = line.strip().split(',')
            if g_type == 'L':
                for i in range(int(start), int(end)+1):
                    name_order_dict[i] = order
                order += 1
            line = f.readline()
    name_order_dict[ 'LIG' ] = order
                
    atom_name_dict = dict()
    lines = [x for x in open(complex_gro,'r')]
    lines = lines[1:]
    lines = [x.strip().split() for x in lines]
    lines = [x for x in lines if len(x) == 6]
    props = [(x[0][:-3],x[0][-3:]) for x in lines]
    for atom_idx, prop in enumerate(props):
        if prop[1] == 'LIG':
            atom_name_dict[ atom_idx ] = 'LIG'
        else:
            atom_name_dict[ atom_idx ] = int(prop[0])

    order_idx_dict = dict()
    for group_idx, movable_group in enumerate(movable_idx_list):
        name = atom_name_dict[ movable_group[0] ]
        for atom_idx in movable_group:
            assert name == atom_name_dict[ atom_idx ]
        if name in name_order_dict:
            order = name_order_dict[ name ]
            if order in order_idx_dict:
                order_idx_dict[ order ].append( group_idx )
            else:
                order_idx_dict[ order ] = [ group_idx ]

    macro_mode = []
    keys = [ order for order in order_idx_dict if type(order)==int ]
    keys.sort()
    if 'LIG' in order_idx_dict:
        keys.append( 'LIG' )
    for key in keys:
        group_list = order_idx_dict[key]
        group_list.sort()
        macro_mode.append(group_list)

    if len(macro_mode) == 0:
        return None
    else:
        return macro_mode
            


##from GL_data import data
##
##import os
##
##convert_folder = '/media/sf_ShareVM/Python_Code/Test_Idea_RandomReverse/AutoKick_Pocket_v3/Converted_Structures/1fin_ENS_dH/cocrystal/1fq1_LIG_docking_1_re'
##
##gro_lammps_in = os.path.join(convert_folder,'complex_converted.input')
##gro_lammps_data = os.path.join(convert_folder,'complex_converted.lmp')
##data_struct = data(gro_lammps_in, gro_lammps_data)
####print('Before sidechain segmentation:')
####print([(i+1,data_struct.atom_molid[i]) for i in range(2367,2381)])
##
#### We separate protein amino acid backbone and sidechain (independent degrees of freedom)
##complex_gro = os.path.join(convert_folder,'complex.gro')
##separate_protein_backbone_sidechain(data_struct, complex_gro)
####print('After sidechain segmentation:')
####print([(i+1,data_struct.atom_molid[i]) for i in range(2367,2381)])
##
##ligand_idx = extract_ligand_idx(data_struct)
##
##gl_lig_in = os.path.join(convert_folder,'LIG_converted.input')
##gl_lig_data = os.path.join(convert_folder,'LIG_converted.lmp')
##gl_lig_rotbond = os.path.join(convert_folder,'LIG_converted.rotbond')
##lig_struct = data(gl_lig_in, gl_lig_data)
##seg_ligand_idx = segmentize_complex_ligand(data_struct, lig_struct, gl_lig_rotbond)
##
##dist_cutoff = 15.0
##center_pos = np.array([-13.094694, 205.03195, 114.42654])
##movable_protein_idx = segmentize_protein_atoms_from_center(data_struct, ligand_idx, center_pos, dist_cutoff)
##movable_index_list = movable_protein_idx.copy()
##movable_index_list += seg_ligand_idx
##
#### Special rotation
###special_rotation = generate_special_rotation_centers(data_struct, movable_index_list)
##special_rotation = generate_special_rotation_axes(data_struct, movable_index_list, complex_gro=complex_gro)
##
#### Macro mode for degrees of freedom
##complex_secondary = os.path.join(convert_folder,'complex_secondary_fragment.txt')
##macro_mode = generate_macro_mode(data_struct, movable_index_list, complex_gro, complex_secondary)
##
##from GL_potential_model import PotentialModel
##inverse_model = PotentialModel(data_struct, movable_index_list,
##                               special_rotation=special_rotation,
##                               macro_mode=macro_mode)
##for i in range(len(inverse_model.macro_mode_idx)):
##    print(inverse_model.macro_mode_idx[i])
###inverse_model()

