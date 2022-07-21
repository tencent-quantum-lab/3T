import pickle

def downsample(in_pkl, out_pkl):
    factor = 100
    data = pickle.load(open(in_pkl,'rb'))
    energy = data['E_binding']
    energy = energy[:,:, [i for i in range(0,energy.shape[2],factor)]]
    data['E_binding'] = energy
    pickle.dump(data, open(out_pkl,'wb'))
    return

downsample('CDK2_run_outputs_20A_2000_3T_energy.pkl', 'CDK2_3T_energy.pkl')
downsample('CDK2_run_outputs_25A_2000_3T_energy.pkl', 'CDK2_25A_3T_energy.pkl')
downsample('HSP90_run_outputs_20A_2000_3T_energy.pkl', 'HSP90_3T_energy.pkl')
downsample('HSP90_rigid_run_outputs_20A_2000_3T_energy.pkl', 'HSP90_rigid_3T_energy.pkl')
downsample('FXA_run_outputs_20A_2000_3T_energy.pkl', 'FXA_3T_energy.pkl')
