import os, time, sys
sys.path.append(os.path.join('.','GL_modules',''))
from GL_modules.GL_data import data
import numpy as np

def temp_dir_cleanup(lig_ff_folder):
    os.system('rm *')
    if os.path.isdir(lig_ff_folder):
        os.system(' '.join(['rm -r',lig_ff_folder]))
    return

def pdb2gmx(clean_pro, proc_pro, reply_txt):
    with open(reply_txt,'w') as f:
        f.write('1\n1\n')
    os.system(' '.join(['gmx pdb2gmx -f',clean_pro,'-o',proc_pro,'<',reply_txt]))
    return

def extract_cgenff(lig_mol2, dl_txt, lig_ori, lig_zip, reply_txt):
    os.system(' '.join(['cp',lig_mol2,lig_ori]))
    while True:
        out = os.system(' '.join(['python ../swiss.py -c',lig_mol2,'>',dl_txt,'2>',reply_txt]))
        if out == 0: break
        time.sleep(2)
    os.system(' '.join(["sed -i 's/index.html/"+lig_zip+"/'",dl_txt]))
    time.sleep(5)
    start_time = time.time()
    while True:
        out = os.system(' '.join(['cat',dl_txt,'| wget -i- --no-check-certificate','2>',reply_txt]))
        if out == 0: break
        time.sleep(5)
        current_time = time.time()
        if current_time - start_time > 600:
            # 10 minutes without server success, we should terminate this molecule
            raise Exception('SwissParam server mol2 conversion fail')
    with open(reply_txt,'w') as f:
        f.write('y\n')
    os.system(' '.join(['unzip',lig_zip,'<',reply_txt]))
    return

def charmm2gmx(lig_itp, lig_par, lig_prm, lig_ff_folder, lig_bonded):
    os.system(' '.join(['python ../py3charmm2gromacs-pvm.py',lig_itp,lig_par,lig_ff_folder]))
    out_lines = []
    with open(os.path.join(lig_ff_folder,'forcefield.itp'),'r') as f:
        line = f.readline()
        while line:
            words = line.strip().split()
            if len(words)==0:
                out_lines.append(line)
            elif not words[0]=='#include':
                out_lines.append(line)
            line = f.readline()
        out_lines.append('#include '+lig_bonded+'\n')
    with open(lig_prm,'w') as f:
        for line in out_lines:
            f.write(line)
    os.system(' '.join(['mv', os.path.join(lig_ff_folder,'ffbonded.itp'), lig_bonded]))
    return

def convert_gromacs_lammps_ligand(lig_itp, lig_prm, lig_gro, lig_top):
    with open(lig_itp,'r') as f:
        content = f.read()
    with open(lig_top,'w') as f:
        f.write('; Include forcefield parameters' + '\n' +\
                   '#include "./charmm36-feb2021.ff/forcefield.itp"' + '\n\n' +\
                   '; Include ligand parameters' + '\n' +\
                   '#include "' + lig_prm + '"\n\n')
        f.write(content)
        f.write('\n' +\
                   '[ molecules ]' +'\n' +\
                   '; Compound        #mols' + '\n' +\
                   'LIG                       1' + '\n')
    cwd = os.getcwd()
    lig_gro_full = os.path.join(cwd, lig_gro)
    lig_top_full = os.path.join(cwd, lig_top)
    os.chdir('../Convert_Gromacs_LAMMPS')
    os.system(' '.join(['python Convert_Gromacs_LAMMPS.py',
                        lig_gro_full, lig_top_full, cwd]))
    os.chdir(cwd)
    return    

def build_new_rotbond(lig_ori, lig_rotbond, converted_ligand_input, converted_ligand_data, converted_ligand_rotbond):
    lig_data = data(converted_ligand_input, converted_ligand_data)
    new_lig_pos = lig_data.atom_pos
    old_lig_pos = []
    with open(lig_ori, 'r') as f:
        line = f.readline()
        while line:
            words = line.strip().split()
            if len(words)==9:
                old_lig_pos.append([float(i) for i in words[2:5]])
            line = f.readline()
        old_lig_pos = np.array(old_lig_pos)
        if not old_lig_pos.shape[0] == new_lig_pos.shape[0]:
            raise Exception('Unmatched ligand atom count for rotatable bond rearrangement')
    n_atoms = old_lig_pos.shape[0]
    all_dist = np.linalg.norm(np.repeat(old_lig_pos[np.newaxis,:,:], n_atoms, axis=0) -\
                              np.repeat(new_lig_pos[:,np.newaxis,:], n_atoms, axis=1), axis=2)
    old_to_new = np.argmin(all_dist, axis=0)
    content = []
    with open(lig_rotbond,'r') as f:
        content.append(f.readline())
        line = f.readline()
        while line:
            words = line.split(',')
            words[1] = str(old_to_new[ int(words[1])-1 ] + 1)
            words[2] = str(old_to_new[ int(words[2])-1 ] + 1)
            content.append( ','.join(words) )
            line = f.readline()
    with open(converted_ligand_rotbond,'w') as f:
        for line in content:
            f.write(line)
    return

def build_complex(lig_pdb, lig_gro, proc_pro, complex_gro):
    os.system(' '.join(['gmx editconf -f',lig_pdb,'-o',lig_gro]))
    pro_lines = []
    with open(proc_pro,'r') as f:
        line = f.readline()
        while line:
            pro_lines.append(line)
            line = f.readline()
    lig_lines = []
    with open(lig_gro,'r') as f:
        line = f.readline()
        while line:
            lig_lines.append(line)
            line = f.readline()
    out_lines = pro_lines[:-1] + lig_lines[2:-1] + pro_lines[-1:]
    out_lines[1] = ' '+str(int(pro_lines[1].strip()) + int(lig_lines[1].strip()))+'\n'
    with open(complex_gro,'w') as f:
        for line in out_lines:
            f.write(line)
    return

def fix_topology(top_file, lig_itp, lig_prm):
    temp_top_file = 'temp_'+top_file
    os.system(' '.join(['cp',top_file,temp_top_file]))
    with open(temp_top_file,'r') as f:
        content = f.read()
    content = content.replace('; Include water topology',
                                              '; Include ligand topology\n#include "' + lig_itp +'"\n\n' +\
                                              '; Include water topology')
    content = content.replace('[ moleculetype ]',
                                             '; Include ligand parameters\n#include "' + lig_prm +'"\n\n' +\
                                             '[ moleculetype ]')
    lines = content.split('\n')
    lines = lines[:-1] + ['LIG                       1'] + lines[-1:]
    with open(top_file,'w') as f:
        content = '\n'.join(lines)
        f.write(content)
    return

def convert_gromacs_lammps_complex(complex_gro, top_file):
    cwd = os.getcwd()
    complex_gro_full = os.path.join(cwd, complex_gro)
    top_file_full = os.path.join(cwd, top_file)
    os.chdir('../Convert_Gromacs_LAMMPS')
    os.system(' '.join(['python Convert_Gromacs_LAMMPS.py',
                        complex_gro_full, top_file_full, cwd]))
    os.chdir(cwd)
    return    

def move_output_files(out_folder, output_files):
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
    for output_file in output_files:
        os.system(' '.join(['mv',output_file,out_folder]))
    time.sleep(2)
    return        


def main(pid, pro_pdb_path, lig_mol2_path):
    # First, create tempporary working space folder if it does not exist yet
    ff_dir = 'charmm36-feb2021.ff'
    lig_ff_folder = 'LIG_ff'
    temp_dir = os.path.join('.', 'temp_'+str(pid))   # temp directory for this subprocess
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
        os.system(' '.join(['cp -r', ff_dir, os.path.join(temp_dir, '')]))
    os.chdir(temp_dir)
    temp_dir_cleanup(lig_ff_folder)

    # Next, check if the conversion has already been done for this particular protein-ligand complex
    pro_code = pro_pdb_path.split('/')[-2]
    lig_code_1 = lig_mol2_path.split('/')[-2]
    lig_code_2 = lig_mol2_path.split('/')[-1][:-5]
    out_dir = os.path.join('..','Converted_Structures')
    converted_complex_input = 'complex_converted.input'
    converted_complex_data = 'complex_converted.lmp'
    complex_secondary_fragment = 'complex_secondary_fragment.txt'
    converted_ligand_input = 'LIG_converted.input'
    converted_ligand_data = 'LIG_converted.lmp'
    converted_ligand_rotbond = 'LIG_converted.rotbond'
    complex_gro = 'complex.gro'
    output_files = [converted_complex_input, converted_complex_data, complex_secondary_fragment,
                            converted_ligand_input, converted_ligand_data, converted_ligand_rotbond,
                            complex_gro]
    complete = True
    for output_file in output_files:
        output_file = os.path.join(out_dir, pro_code, lig_code_1, lig_code_2, output_file)
        if not os.path.isfile(output_file):
            complete = False
    if complete:
        os.chdir('..')
        return

    # Copy input files and prepare output folder
    clean_pro = 'clean_pro.pdb'
    sec_frag = 'secondary_fragment.txt'
    sec_frag = os.path.join(os.path.split(pro_pdb_path)[0], sec_frag)
    lig_mol2 = 'LIG.mol2'
    lig_rotbond_path = lig_mol2_path[:-5] + '.rotbond'
    lig_rotbond = 'LIG.rotbond'
    os.system(' '.join(['cp', pro_pdb_path, clean_pro]))
    os.system(' '.join(['cp', sec_frag, complex_secondary_fragment]))
    os.system(' '.join(['cp', lig_mol2_path, lig_mol2]))
    os.system(' '.join(['cp', lig_rotbond_path, lig_rotbond]))
    out_dir = os.path.join(out_dir, pro_code)
    if not os.path.isdir(out_dir): os.mkdir(out_dir)
    out_dir = os.path.join(out_dir, lig_code_1)
    if not os.path.isdir(out_dir): os.mkdir(out_dir)
    out_dir = os.path.join(out_dir, lig_code_2)
    if not os.path.isdir(out_dir): os.mkdir(out_dir)
        
    # Convert protein pdb to gromacs
    proc_pro = 'processed_pro.gro'
    reply_txt = 'reply.txt'
    pdb2gmx(clean_pro, proc_pro, reply_txt)

    # Get CGenFF params from SwissParam
    dl_txt = 'download.txt'
    lig_ori = 'LIG_ori.mol2'
    lig_zip = 'LIG.zip'
    extract_cgenff(lig_mol2, dl_txt, lig_ori, lig_zip, reply_txt)

    # Convert SwissParam CHARMM param file to Gromacs param file
    lig_itp = 'LIG.itp'
    lig_par = 'LIG.par'
    lig_prm = 'LIG.prm'
    lig_bonded = 'LIG_bonded.itp'
    charmm2gmx(lig_itp, lig_par, lig_prm, lig_ff_folder, lig_bonded)

    # Build the protein-ligand complex Gromacs structure file
    lig_pdb = 'LIG.pdb'
    lig_gro = 'LIG.gro'
    complex_gro = 'complex.gro'
    build_complex(lig_pdb, lig_gro, proc_pro, complex_gro)

    # Edit Gromacs file to include the new ligand param files
    top_file = 'topol.top'
    fix_topology(top_file, lig_itp, lig_prm)
    
    # Convert ligand Gromacs files to LAMMPS format files
    lig_gro = 'LIG.gro'
    lig_top = 'LIG.top'
    convert_gromacs_lammps_ligand(lig_itp, lig_prm, lig_gro, lig_top)

    # Build new rotatable bond file for ligand LAMMPS file (atom order has changed)
    build_new_rotbond(lig_ori, lig_rotbond, converted_ligand_input, converted_ligand_data, converted_ligand_rotbond)

    # Convert complex Gromacs files to LAMMPS format files
    convert_gromacs_lammps_complex(complex_gro, top_file)

    # Move converted LAMMPS files to output folder
    converted_complex_input = 'complex_converted.input'
    converted_complex_data = 'complex_converted.lmp'
    complex_secondary_fragment = 'complex_secondary_fragment.txt'
    converted_ligand_input = 'LIG_converted.input'
    converted_ligand_data = 'LIG_converted.lmp'
    converted_ligand_rotbond = 'LIG_converted.rotbond'
    output_files = [converted_complex_input, converted_complex_data, complex_secondary_fragment,
                            converted_ligand_input, converted_ligand_data, converted_ligand_rotbond,
                            complex_gro]
    move_output_files(out_dir, output_files)
    
    # Empty temp folder from unnecessary temporary files
    temp_dir_cleanup(lig_ff_folder)
    os.chdir('..')
    
    return

if __name__ == "__main__":
    n_args = 3
    if len(sys.argv) != n_args+1:
        sys.exit("\n" +
                 " *** IMPORTANT ***\n" +
                 "This script takes three arguments:\n" +
                 "1) Parallel process ID\n" +
                 "2) Full protein .pdb file path\n" +
                 "3) Full ligand .mol2 file path\n")
    assert len(sys.argv) == n_args+1, "Wrong number of arguments given"
    pid = int(sys.argv[1])
    pro_pdb_path = str(sys.argv[2])
    lig_mol2_path = str(sys.argv[3])
    assert pid>=0
    assert pro_pdb_path.endswith('.pdb')
    assert pro_pdb_path.startswith(os.getcwd())
    assert lig_mol2_path.endswith('.mol2')
    assert lig_mol2_path.startswith(os.getcwd())
    main(pid, pro_pdb_path, lig_mol2_path)


#### Usage example
##pid = 0
##pro_pdb_path = '/media/sf_ShareVM/Python_Code/Test_Idea_RandomReverse/AutoKick_Pocket_v3/Input_Structures/1aq1_ENS_dH/1aq1_ENS_dH.pdb'
##lig_mol2_path = '/media/sf_ShareVM/Python_Code/Test_Idea_RandomReverse/AutoKick_Pocket_v3/Input_Structures/1aq1_ENS_dH/Ligands/CDK2_cross_dock/dude_decoy_600_dock_1AQ1_1.mol2'
##main(pid, pro_pdb_path, lig_mol2_path)
