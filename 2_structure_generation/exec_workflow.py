import os, time, random
import psutil
import numpy as np
import multiprocessing as mp
import torch
from convert_complex import main as main_convert_complex
from minimize_GL_ligand import main as main_minimize_ligand
from minimize_GL_complex import main as main_minimize_complex

def print_log(log_file, message):
    if os.path.isfile(log_file): mode = 'a'
    else: mode = 'w'
    with open(log_file,mode) as f:
        f.write(message + '\n')
    return

def exec_job(job_args):
    # Prevent each CPU from trying to parallelize across CPU's and fighting for resources
    if (not torch.cuda.is_available()) and (cpu_count > 1):
        torch.set_num_threads(1)

    # Extract input parameters
    pid, ligand_epoch, complex_epochs, kick_center, kick_radius, kick_strength, kick_seeds, print_freq, job_list = job_args

    # Pin this process to a specific CPU if we are not using GPU
    if (not torch.cuda.is_available()) and (cpu_count > 1):
        proc = psutil.Process()
        proc.cpu_affinity([pid])
    
    cwd = os.getcwd()
    finish_count = 0
    total_count = len(job_list) * len(kick_seeds)
    log_file = 'minimize_log_'+str(pid)+'.txt'

    for job in job_list:
        pro_pdb_path, lig_mol2_path = job
        
        pro = pro_pdb_path.split('/')[-2]
        lig = '/'.join(lig_mol2_path.split('/')[-2:])[:-5]
        convert_folder = os.path.join(cwd,'Converted_Structures',pro,lig)

        # First we convert the protein ligand complex and the standalone ligand
        for n_retry in range(3):
            # We put the multiple retry and random time delay to prevent overloading SwissParam force field webserver
            # during large-scale ligand batch conversion
            time.sleep(random.randint(0,5)) #300))
            os.chdir(cwd)
            try:
                main_convert_complex(pid, pro_pdb_path, lig_mol2_path)
                break
                # If conversion is successful, immediately get out of for loop
            except:
                print_log(os.path.join(cwd,log_file), 'Exception : convert_complex '+os.path.join(pro,lig))

        # Then we minimize the standalone ligand
        os.chdir(cwd)
        try:
            main_minimize_ligand(pid, convert_folder, ligand_epoch, print_freq)
        except:
            print_log(log_file, 'Exception : minimize_GL_ligand '+os.path.join(pro,lig))

        # Then we minimize the protein ligand complex several times (multiple conformers)
        for kick_seed in kick_seeds:
            kick_prop = [kick_center, kick_radius, kick_strength, kick_seed]
            
            os.chdir(cwd)
            try:
                main_minimize_complex(pid, convert_folder, complex_epochs, kick_prop, print_freq)
            except:
                print_log(log_file, 'Exception : minimize_GL_complex '+os.path.join(pro,lig)+' '+str(kick_seed))

            finish_count += 1
            print_log(log_file, 'Finish '+str(finish_count)+'/'+str(total_count)+' jobs')

    os.chdir(cwd)
            
    return pid

if __name__ == '__main__':
    # Define job list and hyperparameters
    job_list = []
    ligand_epoch = 200
    complex_epochs = [200,2000]
    kick_center = [-13.094694, 205.03195, 114.42654]
    kick_radius = 20.0
    kick_strength = 5.0
    print_freq = 10
    conformers = 10
    init_random_seed = 12345
    np.random.seed(init_random_seed)
    kick_seeds = np.random.randint(100000, size=conformers)

    proteins = os.listdir('Input_Structures')
    cwd = os.getcwd()
    for protein in proteins:
        pro_folder = os.path.join('Input_Structures',protein)
        pro_pdb = [name for name in os.listdir(pro_folder) if name.endswith('.pdb')]
        assert len(pro_pdb) == 1, 'There are more than 1 protein structure in '+pro_folder
        pro_pdb = pro_pdb[0]
        pro_pdb_path = os.path.join(cwd,pro_folder,pro_pdb)
        ligand_folder = os.path.join(pro_folder,'Ligands')
        ligand_sets = os.listdir(ligand_folder)

        for ligand_set in ligand_sets:
            ligand_subfolder = os.path.join(ligand_folder,ligand_set)
            if os.path.isfile(ligand_subfolder):
                continue
            ligands = [name for name in os.listdir(ligand_subfolder) if name.endswith('.mol2')]

            for ligand in ligands:
                lig_mol2_path = os.path.join(cwd,ligand_subfolder,ligand)
                
                job_list.append( tuple((pro_pdb_path, lig_mol2_path)) )

    # Limit job count for testing
    #job_list = job_list[0:1]

    pool_job_args_list = []
    cpu_count = mp.cpu_count()
    if torch.cuda.is_available():
        cpu_count = 1    # if we are using GPU, don't split the jobs across CPU cores
    #cpu_count = 1
    job_count = len(job_list)

    for pid in range(cpu_count):
        start = int(np.floor( job_count / cpu_count * pid ))
        end = int(np.floor( job_count / cpu_count * (pid+1) ))
        pid_job_list = job_list[start:end]

        job_args = [pid,
                    ligand_epoch,
                    complex_epochs,
                    kick_center, kick_radius, kick_strength, kick_seeds,
                    print_freq,
                    pid_job_list]
        pool_job_args_list.append(job_args)

    # Launch the process-based jobs
    procs = []
    if cpu_count == 1:
        exec_job(pool_job_args_list[0])
    else:
        for pid in range(cpu_count):
            proc = mp.Process( target=exec_job, args=(pool_job_args_list[pid], ) )
            procs.append(proc)
            proc.start()
        
        # Complete the processes
        for proc in procs:
            proc.join()

