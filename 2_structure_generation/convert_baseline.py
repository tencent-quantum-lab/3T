import os, sys
from convert_complex import temp_dir_cleanup, pdb2gmx, convert_gromacs_lammps_complex
import torch
sys.path.append('./GL_modules/')
from GL_modules.GL_potential_model import PotentialModel
from GL_modules.GL_data import data

def calculate_baseline(pdb_tag):
    pro_folder = os.path.join( os.getcwd(), 'Input_Structures', pdb_tag )
    pdb_files = [pdb for pdb in os.listdir(pro_folder) if pdb.endswith('.pdb')]
    assert len(pdb_files)==1, 'Input folder '+pro_folder+' has more than one *.pdb file'

    for pdb_file in pdb_files:
        # Placeholder pid
        pid = 0
        
        # First, create tempporary working space folder if it does not exist yet
        ff_dir = 'charmm36-feb2021.ff'
        lig_ff_folder = 'LIG_ff'
        temp_dir = os.path.join('.', 'temp_'+str(pid))   # temp directory for this subprocess
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)
            os.system(' '.join(['cp -r', ff_dir, os.path.join(temp_dir, '')]))
        os.chdir(temp_dir)
        temp_dir_cleanup(lig_ff_folder)
        
        # Convert protein pdb to gromacs
        clean_pro = 'clean_pro.pdb'
        proc_pro = 'processed_pro.gro'
        reply_txt = 'reply.txt'
        os.system(' '.join(['cp', os.path.join(pro_folder,pdb_file), clean_pro]))
        pdb2gmx(clean_pro, proc_pro, reply_txt)

        # Convert complex Gromacs files to LAMMPS format files
        top_file = 'topol.top'
        convert_gromacs_lammps_complex(proc_pro, top_file)

        # Calculate baseline protein energy
        ## Data structure construction
        gro_lammps_in = 'processed_pro_converted.input'
        gro_lammps_data = 'processed_pro_converted.lmp'
        data_struct = data(gro_lammps_in, gro_lammps_data)

        ## Generate torch device
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        ## Model construction
        movable_index = [i for i in range(data_struct.atom_pos.shape[0])] # these are atom indices for all protein atoms
        movable_index_list = [movable_index]
        inverse_model = PotentialModel(data_struct, movable_index_list)
        inverse_model = inverse_model.to(device)

        ## Single-snapshot energy calculation
        inverse_model.bonded_only = False
        outE = str(inverse_model().detach().cpu().numpy())
        out_E_file = 'PRO_outE.txt'
        with open(out_E_file,'w') as f:
            f.write(outE)

        ## Write output
        out_folder = pro_folder
        if not os.path.isdir(out_folder): os.mkdir(out_folder)
        os.system(' '.join(['cp', out_E_file, out_folder+'/']))

        # Empty temp folder from unnecessary temporary files
        temp_dir_cleanup(lig_ff_folder)
        os.chdir('..')

#### Usage example:
## calculate_baseline('1fin_ENS_dH')
## calculate_baseline('1uyg_ENS')
## calculate_baseline('1ezq_ENS_dH')
