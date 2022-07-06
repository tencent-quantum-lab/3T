import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
# we will suppress warnings caused by some nan entries due to failed force field parametrization
warnings.filterwarnings("ignore")   

def load_tsv(tsv_in):
    data = pd.read_csv(tsv_in, sep = '\t', low_memory=False)
    return data

def rearrange_array(in_arr):
    out_arr = np.copy(in_arr)
    idx_dict = {'complex_0':0,
                'complex_1':1,
                'complex_2':2}
    n = 12
    for i in range(3):
        start_idx = i*n
        j = idx_dict[ in_arr[start_idx,0] ]
        out_arr[j*n : (j+1)*n, :] = in_arr[i*n : (i+1)*n, :]
    return out_arr

def extract_RMSD_pairs(pro_code):
    n_top = 3
    tag = os.path.join('Input_Data',pro_code)
    all_RMSD = load_tsv(tag+'_RMSD_features.tsv')
    all_vina = load_tsv(tag+'_vina_features.tsv')
    # Indices on rows are not ordered. But they are ordered on columns.
    RMSD_data = rearrange_array(all_RMSD.to_numpy())
    vina_data = rearrange_array(all_vina.to_numpy())
    for i in range(RMSD_data.shape[0]):
        assert RMSD_data[i,0] == vina_data[i,0]
    RMSD_data = RMSD_data[:,1:]
    vina_data = vina_data[:,1:]
    RMSD_data = RMSD_data.reshape([3,-1,RMSD_data.shape[-1]])
    vina_data = vina_data.reshape([3,-1,vina_data.shape[-1]])
    vina_idx = np.argsort(vina_data[:,2:,:], axis=1) + 2
    RMSD_3Ttop3 = np.take_along_axis(RMSD_data, vina_idx[:,:n_top,:], axis=1)
    RMSD_3Tbest = np.min(RMSD_3Ttop3, axis=1)
    RMSD_ref = RMSD_data[:,0,:]
    return RMSD_3Tbest, RMSD_ref

def plot_fig3_code(pro):
    print(pro+' structure generation RMSD evaluation:')
    RMSD_3Tbest, RMSD_ref = extract_RMSD_pairs(pro)
    dr = 0.25
    rmin = -2
    rmax = 5
    bins = [dr*i for i in range(round(rmin/dr),round(rmax/dr)+1)]
    colors = ['teal','orangered','darkgoldenrod']
    markers = ['o','s','^']
    RMSD_diff = RMSD_ref-RMSD_3Tbest
    for i in range(3):
        plt.figure(0)
        idx = ~np.isnan(RMSD_diff[i,:].astype(float))
        plt.hist(RMSD_diff[i,idx], bins=bins, histtype='step', color=colors[i])
        plt.figure(1)
        plt.scatter(RMSD_ref[i,idx], RMSD_diff[i,idx], c=colors[i], s=5, marker=markers[i])
        print('Dock pose',i,' improved:',np.sum(RMSD_diff[i,:]>=0), '/', RMSD_diff.shape[-1], ', d_RMSD:', round(np.mean(RMSD_diff[i,idx]),2), '+/-', round(np.std(RMSD_diff[i,idx]),2))
        temp1 = RMSD_ref[i,idx]
        temp2 = RMSD_diff[i,idx]
        neg_only = [temp1[j] for j in range(len(temp2)) if temp2[j]<=0]
        pos_only = [temp1[j] for j in range(len(temp2)) if temp2[j]>0]
        print('Data mean:',round(np.mean(temp1),2),'+/-',round(np.std(temp1),2),'Neg mean:',round(np.mean(neg_only),2),'+/-',round(np.std(neg_only),2),'Pos mean:',round(np.mean(pos_only),2),'+/-',round(np.std(pos_only),2))
    print('')
    
    font_legend = 16
    font_label = 18
    font_tick = 16

    if pro == 'CDK2' or pro == 'CDK2_25A':
        #CDK2
        plt.figure(0)
        ymax = 70
        if pro == 'CDK2_25A': ymax = 90
        plt.legend(['Initial Pose 1', 'Initial Pose 2', 'Initial Pose 3'], fontsize=font_legend)
        plt.axis([rmin,rmax,0,ymax])
        plt.plot([0,0],[0,ymax],'--k')
        plt.xlabel(r'$\Delta$$RMSD$ ($\AA$)',fontsize=font_label)
        plt.ylabel('Count',fontsize=font_label)
        plt.xticks(fontsize=font_tick)
        plt.yticks(fontsize=font_tick)
        plt.tight_layout()
        plt.figure(1)
        plt.legend(['Initial Pose 1', 'Initial Pose 2', 'Initial Pose 3'], fontsize=font_legend)
        plt.axis([0,30,-2,4])
        plt.plot([0,30],[0,0],'--k')
        plt.plot([0,4],[0,4],'--k')
        plt.xlabel(r'$RMSD_{init}$ ($\AA$)',fontsize=font_label)
        plt.ylabel(r'$\Delta$$RMSD$ ($\AA$)',fontsize=font_label)
        plt.xticks(fontsize=font_tick)
        plt.yticks(fontsize=font_tick)
        plt.tight_layout()
    elif pro == 'HSP90' or pro == 'HSP90_rigid':
        #HSP90
        plt.figure(0)
        ymax = 70
        if pro == 'HSP90_rigid': ymax = 90
        plt.legend(['Initial Pose 1', 'Initial Pose 2', 'Initial Pose 3'], fontsize=font_legend)
        plt.axis([rmin,rmax,0,ymax])
        plt.plot([0,0],[0,ymax],'--k')
        plt.xlabel(r'$\Delta$$RMSD$ ($\AA$)',fontsize=font_label)
        plt.ylabel('Count',fontsize=font_label)
        plt.xticks(fontsize=font_tick)
        plt.yticks(fontsize=font_tick)
        plt.tight_layout()
        plt.figure(1)
        plt.legend(['Initial Pose 1', 'Initial Pose 2', 'Initial Pose 3'], fontsize=font_legend)
        plt.axis([0,25,-1.5,3])
        plt.plot([0,25],[0,0],'--k')
        plt.plot([0,3],[0,3],'--k')
        plt.xlabel(r'$RMSD_{init}$ ($\AA$)',fontsize=font_label)
        plt.ylabel(r'$\Delta$$RMSD$ ($\AA$)',fontsize=font_label)
        plt.xticks(fontsize=font_tick)
        plt.yticks(fontsize=font_tick)
        plt.tight_layout()
    elif pro == 'FXA':
        #FXa
        plt.figure(0)
        ymax = 30
        plt.legend(['Initial Pose 1', 'Initial Pose 2', 'Initial Pose 3'], fontsize=font_legend)
        plt.axis([rmin,rmax,0,ymax])
        plt.plot([0,0],[0,ymax],'--k')
        plt.xlabel(r'$\Delta$$RMSD$ ($\AA$)',fontsize=font_label)
        plt.ylabel('Count',fontsize=font_label)
        plt.xticks(fontsize=font_tick)
        plt.yticks(fontsize=font_tick)
        plt.tight_layout()
        plt.figure(1)
        plt.legend(['Initial Pose 1', 'Initial Pose 2', 'Initial Pose 3'], fontsize=font_legend)
        plt.axis([0,25,-1.5,3])
        plt.plot([0,25],[0,0],'--k')
        plt.plot([0,3],[0,3],'--k')
        plt.xlabel(r'$RMSD_{init}$ ($\AA$)',fontsize=font_label)
        plt.ylabel(r'$\Delta$$RMSD$ ($\AA$)',fontsize=font_label)
        plt.xticks(fontsize=font_tick)
        plt.yticks(fontsize=font_tick)
        plt.tight_layout()

    tag = os.path.join('Figures','')
    if pro=='CDK2':
        plt.figure(0)
        plt.savefig(tag+'Fig_3c.svg', format='svg', dpi=1200, transparent=True)
        plt.figure(1)
        plt.savefig(tag+'Fig_3d.svg', format='svg', dpi=1200, transparent=True)
    if pro=='CDK2_25A':
        plt.figure(0)
        plt.savefig(tag+'Fig_S2a.svg', format='svg', dpi=1200, transparent=True)
        plt.figure(1)
        plt.savefig(tag+'Fig_S2e.svg', format='svg', dpi=1200, transparent=True)
    elif pro=='HSP90':
        plt.figure(0)
        plt.savefig(tag+'Fig_S2b.svg', format='svg', dpi=1200, transparent=True)
        plt.figure(1)
        plt.savefig(tag+'Fig_S2f.svg', format='svg', dpi=1200, transparent=True)
    elif pro=='HSP90_rigid':
        plt.figure(0)
        plt.savefig(tag+'Fig_S2c.svg', format='svg', dpi=1200, transparent=True)
        plt.figure(1)
        plt.savefig(tag+'Fig_S2g.svg', format='svg', dpi=1200, transparent=True)
    elif pro=='FXA':
        plt.figure(0)
        plt.savefig(tag+'Fig_S2d.svg', format='svg', dpi=1200, transparent=True)
        plt.figure(1)
        plt.savefig(tag+'Fig_S2h.svg', format='svg', dpi=1200, transparent=True)
    return

plot_fig3_code('CDK2')
plt.show()
plot_fig3_code('CDK2_25A')
plt.show()
plot_fig3_code('HSP90')
plt.show()
plot_fig3_code('HSP90_rigid')
plt.show()
plot_fig3_code('FXA')
plt.show()

