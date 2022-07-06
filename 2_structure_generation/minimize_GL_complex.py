import numpy as np
import torch
import torch.optim as optim
import sys, os
sys.path.append(os.path.join('.','GL_modules',''))
from GL_modules.GL_potential_model import PotentialModel
from GL_modules.GL_data import data
from GL_modules.GL_data_util import segmentize_nearby_protein_atoms, extract_ligand_idx
from GL_modules.GL_data_util import separate_protein_backbone_sidechain, segmentize_complex_ligand
from GL_modules.GL_data_util import segmentize_protein_atoms_from_center
from GL_modules.GL_data_util import generate_special_rotation_axes
from GL_modules.GL_data_util import generate_macro_mode
from torch.nn.parameter import Parameter
import ase
import ase.io as io

def square_loss(outp, outp_target):
    return torch.sum( (outp-outp_target)**2 )

def ase_parse_xyz(xyz_file, temp_folder):
    with open(xyz_file,'r') as f:
        last_frame = []
        line = f.readline()
        while line:
            words = line.strip().split()
            if len(words) == 1:
                last_frame = [line]
            else:
                last_frame.append(line)
            line = f.readline()
    temp_xyz = os.path.join(temp_folder, 'temp.xyz')
    with open(temp_xyz,'w') as f:
        for line in last_frame:
            f.write(line)
    ase_out = io.read(temp_xyz, format='xyz', index=-1)
    os.remove(temp_xyz)
    return ase_out

def out_file_list(out_tag):
    out_xyz = out_tag+'.xyz'
    out_outE = out_tag+'_outE.txt'
    return [out_xyz, out_outE]

def append_out_E_files(all_out_E, out_E_files):
    with open(all_out_E,'w') as f1:
        for out_E_file in out_E_files:
            with open(out_E_file,'r') as f2:
                f1.write( f2.read() )
    return    

def print_log(message):
    global log_file
    with open(log_file,'a') as f:
        f.write( str(message)+'\n')
    return

def run_model(inverse_model, optimizers, E_target, n_epoch, out_tag, schedulers=None, print_freq=10):
    
    # determine atom elements for printing convenience later on
    mass_elem_dict = {1:'H', 7:'Li', 9:'Be', 11:'B', 12:'C', 14:'N', 16:'O', 19:'F',
                                   23:'Na', 24:'Mg', 27:'Al', 28:'Si', 31:'P', 32:'S', 35:'Cl',
                                   39:'K', 40:'Ca', 70:'Ga', 73:'Ge', 75:'As', 79:'Se', 80:'Br',
                                   85:'Rb', 88:'Sr', 115:'In', 119:'Sn', 122:'Sb', 128:'Te', 127:'I'} # this is rounded mass to elem format
    atom_type = inverse_model.atom_type.cpu().detach().numpy().astype(int) # this is already in 0 to n_type-1 format
    temp = inverse_model.atom_mass.detach().cpu().numpy().astype(float)
    type_elem_dict = {}
    for i in range(temp.shape[0]):
        type_elem_dict[ i ] = mass_elem_dict[ round(temp[i]) ]
    del temp
    atom_elem = [ type_elem_dict[i] for i in atom_type ]

    out_xyz, out_outE = out_file_list(out_tag)

    loss_hist = np.zeros(n_epoch)
    out_hist = np.zeros(n_epoch)
    na = inverse_model.atom_pos.shape[0]
    xyz_hist = np.zeros([n_epoch+1,na,3])
    xyz_hist[0,:,:] = inverse_model.atom_pos.detach().cpu().numpy()

    #inverse_model.bonded_only = True
    inverse_model.bonded_only = False
    for epoch in range(n_epoch):
        outp_E = inverse_model()
        #loss = square_loss(outp_E, E_target)
        loss = outp_E
    
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(inverse_model.parameters(), 1e11)
        for optimizer in optimizers:
            optimizer.step()
        if schedulers:
            for scheduler in schedulers:
                scheduler.step()
        
        loss_hist[epoch] = loss.detach().cpu().numpy()
        out_hist[epoch] = outp_E.detach().cpu().numpy()
        xyz_hist[epoch+1,:,:] = inverse_model.atom_pos.detach().cpu().numpy()

        if epoch % print_freq == 0:
            print_log('Step:'+str(epoch))
            print('Step:'+str(epoch))
            
    # Clear the gradient after finishing the minimization
    for optimizer in optimizers:
        optimizer.zero_grad()

    with open(out_outE,'w') as f:
        for i in range(out_hist.shape[0]):
            f.write(str(out_hist[i])+'\n')

    # We only add printout of the last xyz
    atoms = ase.Atoms(symbols=atom_elem, positions=xyz_hist[n_epoch])
    io.write(out_xyz, atoms, format='xyz', append=False)

    # Use this section if we want to print out the entire minimization trajectory instead
##    for i in range(xyz_hist.shape[0]):
##        atoms = ase.Atoms(symbols=atom_elem, positions=xyz_hist[i])
##        if i == 0: io.write(out_xyz, atoms, format='xyz', append=False)
##        else: io.write(out_xyz, atoms, format='xyz', append=True)

    return

def main(pid, convert_folder, epochs, kick_prop, print_freq):
    ## Data structure construction
    gro_lammps_in = os.path.join(convert_folder,'complex_converted.input')
    gro_lammps_data = os.path.join(convert_folder,'complex_converted.lmp')
    data_struct = data(gro_lammps_in, gro_lammps_data)

    ## We separate protein amino acid backbone and sidechain (independent degrees of freedom)
    complex_gro = os.path.join(convert_folder,'complex.gro')
    separate_protein_backbone_sidechain(data_struct, complex_gro)
    complex_secondary = os.path.join(convert_folder,'complex_secondary_fragment.txt')

    ## Determine which are ligand and protein atoms
    ligand_idx = extract_ligand_idx(data_struct)

    ## New functionality, segmentize the ligand based on rotatable bond
    ## Requires that standalone ligand atom order be identical to complex ligand atoms
    gl_lig_in = os.path.join(convert_folder,'LIG_converted.input')
    gl_lig_data = os.path.join(convert_folder,'LIG_converted.lmp')
    gl_lig_rotbond = os.path.join(convert_folder,'LIG_converted.rotbond')
    lig_struct = data(gl_lig_in, gl_lig_data)
    seg_ligand_idx = segmentize_complex_ligand(data_struct, lig_struct, gl_lig_rotbond)

    ## Generate torch device
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    ## Model construction
    movable_index_list = seg_ligand_idx  # these are the segmentized atom indices for the ligand in the complex
    inverse_model = PotentialModel(data_struct, movable_index_list)
    inverse_model = inverse_model.to(device)
    inverse_model.train()

    ## Output file recording
    out_folder_path = ['Minimized_Structures'] + convert_folder.split('/')[-3:]
    for i in range(len(out_folder_path)):
        out_folder = '/'.join(out_folder_path[0:i+1])
        if not os.path.isdir(out_folder): os.mkdir(out_folder)
    out_E_files = []
    temp_folder = 'temp_'+str(pid)
    global log_file
    log_file = os.path.join('minimize_log_' + str(pid) + '.txt')
    if not os.path.isfile(log_file):
        with open(log_file,'w') as f:
            pass
    complex_tag = '/'.join(convert_folder.split('/')[-3:])
    print_log( complex_tag )

    ## First freeze protein, relax ligand structure
    print_log( 'Minimizing the ligand structure alone within the complex pocket' )
    movable_index_list = seg_ligand_idx  # these are the segmentized atom indices for the ligand in the complex
    ## Special rotation
    special_rotation = generate_special_rotation_axes(data_struct, movable_index_list)
    macro_mode = generate_macro_mode(data_struct, movable_index_list, complex_gro, complex_secondary)
    E_target = 0
    n_epoch = epochs[0]
    out_tag = os.path.join(out_folder, '_'.join(['step1'] + [str(i) for i in epochs[0:1]]) )
    out_xyz = out_tag+'.xyz'
    if os.path.isfile(out_xyz):
        # Grab last snapshot and update the atom positions in the model
        # Because of string-float conversion, it'll be slightly different
        # than the else branch below
        ase_atoms = ase_parse_xyz(out_xyz, temp_folder)
        print_log( 'Reloaded ' + out_xyz )
        xyz = torch.Tensor(ase_atoms.positions)
        inverse_model.attach_init_inputs(xyz, movable_index_list, special_rotation=special_rotation, macro_mode=macro_mode)
    else:
        # Update the atom positions in the model to start fresh
        xyz = torch.Tensor(inverse_model.atom_pos.detach().cpu().numpy())
        inverse_model.attach_init_inputs(xyz, movable_index_list, special_rotation=special_rotation, macro_mode=macro_mode)
        optim_params = [param for param in inverse_model.movable_pos_list]
        ligand_translation = inverse_model.translation_list
        ligand_rotation = inverse_model.rotation_list
        optim_params += [ligand_translation, ligand_rotation]
        if special_rotation != None:
            ligand_special_rotation = inverse_model.special_rotation_list
            optim_params += [ligand_special_rotation]
        if macro_mode != None:
            ligand_macro_translation = inverse_model.macro_mode_translation_list
            ligand_macro_rotation = inverse_model.macro_mode_rotation_list
            optim_params += [ligand_macro_translation, ligand_macro_rotation]
        optimizer = optim.Adam( optim_params , 1e-3,
                               weight_decay=0)
        optimizers = [ optimizer ]
        run_model( inverse_model, optimizers, E_target, n_epoch, out_tag, print_freq=print_freq)
    out_E_files.append( out_file_list(out_tag)[1] )

    ## Finally we will randomly jolt nearby protein segments, and allow the segments and ligands to re-minimize
    print_log( 'Minimizing both ligand and nearby protein pocket structure after energetic kick' )
    center_pos, dist_cutoff, strength, seed = kick_prop
    center_pos = np.array(center_pos)
    movable_protein_idx = segmentize_protein_atoms_from_center(data_struct, ligand_idx, center_pos, dist_cutoff)
    movable_index_list = movable_protein_idx.copy()
    movable_index_list += seg_ligand_idx
    ## Special rotation
    special_rotation = generate_special_rotation_axes(data_struct, movable_index_list, complex_gro=complex_gro)
    macro_mode = generate_macro_mode(data_struct, movable_index_list, complex_gro, complex_secondary)
    E_target = 0
    n_epoch = epochs[1]
    out_tag = os.path.join(out_folder, '_'.join(['step2'] + [str(i) for i in epochs[0:2]] + [str(strength),str(seed)]) )
    out_xyz = out_tag+'.xyz'
    if os.path.isfile(out_xyz):
        # Grab last snapshot and update the atom positions in the model
        # Because of string-float conversion, it'll be slightly different
        # than the else branch below
        ase_atoms = ase_parse_xyz(out_xyz, temp_folder)
        print_log( 'Reloaded ' + out_xyz )
        xyz = torch.Tensor(ase_atoms.positions)
        inverse_model.attach_init_inputs(xyz, movable_index_list, special_rotation=special_rotation, macro_mode=macro_mode)
    else:
        # Update the atom positions in the model to start fresh
        xyz = torch.Tensor(inverse_model.atom_pos.detach().cpu().numpy())
        inverse_model.attach_init_inputs(xyz, movable_index_list, special_rotation=special_rotation, macro_mode=macro_mode)
        # We now randomly jolt the movable atoms within the pocket
        trans = 0.3 * strength
        rot = 0.03 * strength
        inverse_model.jolt_movable_atoms(seed = seed, max_translation = trans, max_rotation = rot,
                                         ignore_last = False)
        optim_params = [param for param in inverse_model.movable_pos_list]
        pocket_translation = inverse_model.translation_list
        pocket_rotation = inverse_model.rotation_list
        optim_params += [pocket_translation, pocket_rotation]
        if special_rotation != None:
            pocket_special_rotation = inverse_model.special_rotation_list
            optim_params += [pocket_special_rotation]
        optimizer = optim.Adam( optim_params , 1e-1,
                               weight_decay=0)
        optimizers = [ optimizer ]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.7)
        schedulers = [ scheduler ]
        if macro_mode != None:
            pocket_macro_translation = inverse_model.macro_mode_translation_list
            pocket_macro_rotation = inverse_model.macro_mode_rotation_list
            optim_params = [pocket_macro_translation, pocket_macro_rotation]
            optimizer = optim.Adam( optim_params , 1e-2,
                                    weight_decay=0)
            optimizers.append( optimizer )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.7)
            schedulers.append( scheduler )
        run_model( inverse_model, optimizers, E_target, n_epoch, out_tag, schedulers=schedulers, print_freq=print_freq)
    out_E_files.append( out_file_list(out_tag)[1] )

    ## Combine output energy files for convenience purposes
    all_out_E = os.path.join(out_folder, '_'.join(['stepAll'] + [str(i) for i in epochs[0:2] + [str(strength),str(seed)] + ['outE.txt']]) )
    append_out_E_files(all_out_E, out_E_files)

    ## Notify job creator of complex minimization completion
    print_log('Finished minimizing complex | '+complex_tag+' | '+str(kick_prop))
    return

if __name__ == '__main__':
    n_args = 12
    if len(sys.argv) != n_args+1:
        sys.exit("\n" +
                 " *** IMPORTANT ***\n" +
                 "This script takes twelve arguments:\n" +
                 "1) Parallel process ID\n" +
                 "2) Folder of Gromacs-LAMMPS converted structure\n" +
                 "3) Epoch count for protein minimization\n" +
                 "4) Epoch count for ligand minimization\n" +
                 "5) Epoch count for complex pocket minimization\n" +
                 "6-8) Complex pocket kick center [x,y,z] position\n" +
                 "9) Complex pocket kick radius\n" +
                 "10) Complex pocket kick strength\n" +
                 "11) Complex pocket kick random seed\n" +
                 "12) Print frequency\n")
    assert len(sys.argv) == n_args+1, "Wrong number of arguments given"
    pid = int(sys.argv[1])
    convert_folder = str(sys.argv[2])
    epoch_pro = int(sys.argv[3])
    epoch_lig = int(sys.argv[4])
    epoch_pkt = int(sys.argv[5])
    kick_center = [float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8])]
    kick_radius = float(sys.argv[9])
    kick_strength = float(sys.argv[10])
    kick_seed = int(sys.argv[11])
    print_freq = int(sys.argv[12])

    epochs = [epoch_pro, epoch_lig, epoch_pkt]
    kick_prop = [kick_center, kick_radius, kick_strength, kick_seed]
    main(pid, convert_folder, epochs, kick_prop, print_freq)

#### Usage example
##pid = 0
##convert_folder = '/apdcephfs/private_jpmailoa/drug/complex_pl/AutoKick_Pocket/Converted_Structures/1fin_ENS_dH/cocrystal/1fq1_LIG_docking_2_re'
##epochs = [10,10]
####epochs = [100,250]
##kick_prop = [[-13.094694, 205.03195, 114.42654], 20.0, 1.0, 23456]
##print_freq = 10
##main(pid, convert_folder, epochs, kick_prop, print_freq)
 
