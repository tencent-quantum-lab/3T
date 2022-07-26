import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def extract_ML_prop(csv_in):
    in_df = pd.read_csv(csv_in, sep=',', low_memory=False)
    nef_indices = []
    roc_indices = []
    for i in range(len(in_df)):
        row_type = in_df[ in_df.columns[0] ].values[i]
        if row_type == 'nef_Ra': nef_indices.append(i)
        elif row_type == 'roc_auc': roc_indices.append(i)

    nef_df = in_df.iloc[ nef_indices ]
    roc_df = in_df.iloc[ roc_indices ]

    def _process_df(df):
        columns = df.columns
        columns = [columns[i] for i in range(len(columns))]
        out_dict = dict()
        for column in columns:
            if column.startswith('sample_') or column=='GBT':
                out_dict[column] = df[column].values
        return out_dict

    return _process_df(nef_df), _process_df(roc_df)


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, ylabel, xlabels, yrange):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(xlabels) + 1), labels=xlabels, fontsize=14)
    ax.set_xlim(0.25, len(xlabels) + 0.75)
    ax.set_xlabel(r'$n_{conf}$', fontsize=18)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_ylim(yrange[0], yrange[1])
    return

def create_violinplot(fig_num, data, ylabel, xlabels, yrange):
    fig = plt.figure(fig_num, figsize=(5,5))
    plt.grid(visible=True, which='major', axis='y', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(visible=True, which='minor', axis='y', color='#999999', linestyle='-', alpha=0.2)
    violplt = plt.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    idx_3T = xlabels.index('$\\bf{10_{3T}}$')
    for i, pc in enumerate(violplt['bodies']):
        if i == idx_3T: pc.set_facecolor('#F8CBAD')
        else: pc.set_facecolor('#DEEBF7')
        #pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    plt.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    plt.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    plt.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    # set style for the axes
    for ax in fig.axes:
        set_axis_style(ax, ylabel, xlabels, yrange)
        ax.set_axisbelow(True)

    plt.xticks(fontsize=14) #, rotation=45)
    plt.yticks(fontsize=14)
    plt.tick_params(axis='x', which='minor', bottom=False, top=False)
    plt.tight_layout()

    return

def plot_fig4_code(tag):
    in_tag = os.path.join('Input_Data',tag)
    nef, roc = extract_ML_prop(in_tag + '_Ricci_ML.csv')
    nef3T, roc3T = extract_ML_prop(in_tag + '_scrambled_3T_ML.csv')
    nef['3T'] = nef3T['GBT']
    roc['3T'] = roc3T['GBT']
    print(tag+' classification metrics (30x4-cv, 120 data points):')
    print('AUC-ROC:', np.mean(roc['3T']), '+/-', np.std(roc['3T']))
    print('NEF_Ra:', np.mean(nef['3T']), '+/-', np.std(nef['3T']))
    print('')

    if tag == 'CDK2':
        labels = ['sample_10','sample_20','3T','sample_50','sample_100','sample_200','sample_402']
        xlabels = ['10', '20', '$\\bf{10_{3T}}$', '50', '100', '200', '402']
        nef_yrange = [0.1, 1.0]
        roc_yrange = [0.5, 1.0]
    elif tag == 'CDK2_25A':
        labels = ['sample_10','sample_20','sample_50','sample_100','sample_200','sample_402','3T']
        xlabels = ['10', '20', '50', '100', '200', '402', '$\\bf{10_{3T}}$']
        nef_yrange = [0.1, 1.0]
        roc_yrange = [0.5, 1.0]
    elif tag == 'HSP90':
        labels = ['sample_10','sample_20','sample_30','sample_50','sample_64','3T']
        xlabels = ['10', '20', '30', '50', '64', '$\\bf{10_{3T}}$']
        nef_yrange = [0.1, 1.0]
        roc_yrange = [0.5, 1.0]
    elif tag == 'HSP90_rigid':
        labels = ['sample_10','sample_20','sample_30','sample_50','sample_64','3T']
        xlabels = ['10', '20', '30', '50', '64', '$\\bf{10_{3T}}$']
        nef_yrange = [0.1, 1.0]
        roc_yrange = [0.5, 1.0]
    elif tag == 'FXA':
        labels = ['sample_10','sample_20','sample_30','sample_60','sample_136','3T']
        xlabels = ['10', '20', '30', '60', '136', '$\\bf{10_{3T}}$']
        nef_yrange = [0.1, 1.0]
        roc_yrange = [0.5, 1.0]
        
    fig_num = 0
    data = [nef[label] for label in labels]
    ylabel = r'NEF$_{(Ra)}$'
    create_violinplot(fig_num, data, ylabel, xlabels, nef_yrange)
    fig_num = 1
    data = [roc[label] for label in labels]
    ylabel = 'AUC-ROC'
    create_violinplot(fig_num, data, ylabel, xlabels, roc_yrange)

    out_tag = os.path.join('Figures','')
    if tag=='CDK2':
        plt.figure(0)
        plt.savefig(out_tag+'Fig_S3a.svg', format='svg', dpi=1200, transparent=True)
        plt.figure(1)
        plt.savefig(out_tag+'Fig_S3d.svg', format='svg', dpi=1200, transparent=True)
    elif tag=='CDK2_25A':
        plt.figure(0)
        plt.savefig(out_tag+'Fig_4b.svg', format='svg', dpi=1200, transparent=True)
        plt.figure(1)
        plt.savefig(out_tag+'Fig_4d.svg', format='svg', dpi=1200, transparent=True)
    elif tag=='HSP90':
        plt.figure(0)
        plt.savefig(out_tag+'Fig_4c.svg', format='svg', dpi=1200, transparent=True)
        plt.figure(1)
        plt.savefig(out_tag+'Fig_4e.svg', format='svg', dpi=1200, transparent=True)
    elif tag=='HSP90_rigid':
        plt.figure(0)
        plt.savefig(out_tag+'Fig_S3b.svg', format='svg', dpi=1200, transparent=True)
        plt.figure(1)
        plt.savefig(out_tag+'Fig_S3e.svg', format='svg', dpi=1200, transparent=True)
    elif tag=='FXA':
        plt.figure(0)
        plt.savefig(out_tag+'Fig_S3c.svg', format='svg', dpi=1200, transparent=True)
        plt.figure(1)
        plt.savefig(out_tag+'Fig_S3f.svg', format='svg', dpi=1200, transparent=True)
    return
