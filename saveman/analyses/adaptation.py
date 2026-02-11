"""Adaptation / eccentricity analysis utilities."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr
import cmasher as cmr
from surfplot import Plot
from saveman.config import Config
from saveman.utils import parse_roi_names, get_surfaces, test_regions, get_files
from saveman.analyses import plotting

def eccentricity_analysis(data, method='anova', factor='epoch'):
    """Determine if regions show significant changes in eccentricity across
    task epochs

    Basic mass-univariate approach that performs an F-test across each region,
    followed by follow-up paired t-tests on significant regions

    Parameters
    ----------
    data : pandas.DataFrame
        Subject gradient data with distance column

    Returns
    -------
    pd.DataFrame, pd.DataFrame
        ANOVA and post-hoc stats tables, respectively
    """
    res = test_regions(data, method, factor)

    # post hoc analysis
    if method != 'ttest':
        if factor != 'epoch':
            sources = np.unique(res['Source'].values)
            posthoc = {}
            for source in sources:
                source_res = res.query('Source == @source')
                sig_regions = source_res.loc[source_res['sig_corrected'].astype(bool), 'roi'].tolist()
                if sig_regions:
                    post_data = data[data['roi'].isin(sig_regions)]
                    posthoc[source] = test_regions(post_data, 'ttest', factor=source)  
            posthoc = pd.concat([posthoc[i].fillna("-") for i in posthoc], ignore_index=True) \
                    .drop('index', axis=1) \
                    .sort_values(by='roi_ix').reset_index(drop=True)
            posthoc.insert(3, 'epoch', posthoc.pop('epoch'))
        else:   
            sig_regions = res.loc[res['sig_corrected'].astype(bool), 'roi'].tolist()
            if sig_regions:
                post_data = data[data['roi'].isin(sig_regions)]
                posthoc = test_regions(post_data, 'ttest', factor)  
            else:
                # no significant anova results
                posthoc = None
        return res, posthoc

    else: return res
 
def anova_stat_map(anova, out_dir, name='anova', vmax='auto',
                   vmin='auto', thresholded=True, outline=True):
    """Plot thresholded or unthresholded mass-univariate ANOVA results 

    Threshold set as q < .05, where q = FDR-corrected two-tailed p values

    Parameters
    ----------
    anova : pandas.DataFrame
        ANOVA results

    Returns
    -------
    pandas.DataFrame
        ANOVA results
    """
    df = anova.query("sig_corrected == 1") if thresholded else anova
    if len(df) == 0:
        return None
    fvals = df['F'].values
    if vmax == 'auto':
        vmax = int(np.nanmax(fvals))
    if vmin == 'auto':
        vmin = np.nanmin(fvals)

    # get orange (positive) portion. Max reduced because white tends to wash 
    # out on brain surfaces
    cmap = cmr.get_sub_cmap(plotting.stat_cmap(), .5, 1)
    # get cmap that spans from stat threshold to max rather than whole range, 
    # which matches scaling of t-test maps
    cmap_min = vmin / vmax
    cmap = cmr.get_sub_cmap(cmap, cmap_min, 1)

    plotting.plot_cbar(cmap, vmin, vmax, 'horizontal', size=(1, .3), 
                         n_ticks=2)
    prefix = os.path.join(out_dir, name)
    plt.savefig(prefix + '_cbar')

    surfaces = get_surfaces()
    sulc = plotting.get_sulc()
    x = plotting.weights_to_vertices(fvals, Config().atlas, 
                                       df['roi_ix'].values)
    sulc_params = dict(data=sulc, cmap='gray', cbar=False)
    layer_params = dict(cmap=cmap, cbar=False, color_range=(vmin, vmax))
    outline_params = dict(data=(np.abs(x) > 0).astype(bool), cmap='binary', 
                          cbar=False, as_outline=True)

    # 2x2 grid
    p = Plot(surfaces['lh'], surfaces['rh'])
    p.add_layer(**sulc_params)
    p.add_layer(x, **layer_params)
    if outline:
        p.add_layer(**outline_params)

    cbar_kws = dict(n_ticks=2, aspect=8, shrink=.15, draw_border=False)
    fig = p.build()#(cbar_kws=cbar_kws)
    fig.savefig(prefix)

    # dorsal views
    p = Plot(surfaces['lh'], surfaces['rh'], views='dorsal', size=(150, 200), 
             zoom=3.3)
    p.add_layer(**sulc_params)
    p.add_layer(x, **layer_params)
    if outline:
        p.add_layer(**outline_params)
    fig = p.build(colorbar=False)
    fig.savefig(prefix + '_dorsal')

    # posterior views
    p = Plot(surfaces['lh'], surfaces['rh'], views='posterior', 
             size=(150, 200), zoom=3.3)
    p.add_layer(**sulc_params)
    p.add_layer(x, **layer_params)
    if outline:
        p.add_layer(**outline_params)
    fig = p.build(colorbar=False)
    fig.savefig(prefix + '_posterior')
    try:
        plt.close(fig)
    except Exception:
        pass
    return x

def find_optimal_k(data, k_min=2, k_max=10, out_dir=None, fname="ensemble_optimal_k"):
    inertias, silhouettes = [], []
    K_range = range(k_min, k_max+1)
    
    # distance from each point to the line p1–p2
    def _dist_to_line(xy, p1, p2):
        # cross product magnitude / line length
        return np.abs(np.cross(p2-p1, p1-xy)) / np.linalg.norm(p2-p1)

    for k in K_range:
        km = KMeans(n_clusters=k, n_init='auto', random_state=1234).fit(data)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(data, km.labels_))
    # Plot
    fig, ax = plt.subplots(1,2, figsize=(5,2))
    ax[0].plot(K_range, inertias, '-o', color='k')
    ax[0].set_xlabel('k'); ax[0].set_ylabel('Inertia'); ax[0].set_title('Elbow Method')
    ax[1].plot(K_range, silhouettes, '-o', color='k')
    ax[1].set_xlabel('k'); ax[1].set_ylabel('Silhouette Score'); ax[1].set_title('Silhouette Analysis')
    ax[0].set_xticks(K_range); ax[1].set_xticks(K_range)
    plt.tight_layout()
    sns.despine()
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, fname))
    # Return the two heuristic picks
    p1 = np.array([K_range[0], inertias[0]])
    p2 = np.array([K_range[-1], inertias[-1]])
    dists = np.array([_dist_to_line(np.array([k, i]), p1, p2)
                      for k, i in zip(K_range, inertias)])
    # elbow is the k with maximum distance
    elbow_idx  = np.argmax(dists)
    best_elbow = K_range[elbow_idx]
    best_silh = K_range[silhouettes.index(max(silhouettes))]
    return best_elbow, best_silh

def _ensemble_cmap(n, as_cmap=False):
    yeo_cmap = plotting.yeo_cmap(networks=7)
    if n == 3:
        colors = [yeo_cmap['Default'], yeo_cmap['Vis'], yeo_cmap['SomMot']]
    elif n == 4:
        colors = [yeo_cmap['Default'], yeo_cmap['Vis'], yeo_cmap['SalVentAttn'], yeo_cmap['SomMot']]
    if as_cmap:
        return LinearSegmentedColormap.from_list('cmap', colors, N=n)
    else:
        return dict(zip(range(1, n + 1), colors))

def ensemble_analysis(gradients, anova, out_dir, k=3, base='base',
                      prefix='', n_clusters=3, use_optimal_k=False):
    """Cluster significant regions into functional ensembles

    Parameters
    ----------
    gradients : pandas.DataFrame
        Subject gradient data
    anova : pandas.DataFrame
        Pre-computed ANOVA data
    out_dir : str
        Figure save/output directory
    k : int, optional
        Number of gradients/dimensions to include, by default 3
    base : one of the eixisting epochs in gradient dataframe
           base epoch to perform clustering on
    prefix : 
             prefix for saving the figure
    reorder : True or False
              if it is true the epochs are in chronological order
    area : 'lh', 'rh', 'whole'
            the area of brain to perform clustering on

    Returns
    -------
    pandas.DataFrame
        Region ensemble assignments
    """
    cols = [f'g{i}' for i in np.arange(k) + 1]
    base_loadings = gradients.query("epoch == 'base'")   
    sig_rois = anova.query("sig_corrected == 1")['roi'].tolist()
    # clusster based on the mean loadings or base loadings
    if base == 'mean':
        df = gradients.query("roi in @sig_rois") \
                          .groupby(['roi'], sort=False)[cols] \
                          .mean() \
                          .reset_index()
    elif base == 'base':
        df = base_loadings.query("roi in @sig_rois") \
                          .groupby(['epoch', 'roi'], sort=False)[cols] \
                          .mean() \
                          .reset_index()
    if use_optimal_k:
        # Heuristic: use the silhouette-optimal k (second return value)
        _, n_clusters = find_optimal_k(df[cols], out_dir=out_dir)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=1234)
    # fit
    kmeans.fit(df[cols])
    raw_labels = kmeans.labels_             
    
    centroids = kmeans.cluster_centers_     
    order = np.argsort(centroids[:, 0])        
    label_map = { old: new + 1 for new, old in enumerate(order) }
    
    # apply it
    df['ensemble'] = [label_map[l] for l in raw_labels]
    res = base_loadings.merge(df[['roi', 'ensemble']], on='roi', how='left')

    # brain plot
    x = plotting.weights_to_vertices(res['ensemble'].values, Config().atlas)
    x = np.nan_to_num(x)
    surfaces = get_surfaces()
    sulc = plotting.get_sulc()
    cmap = _ensemble_cmap(n_clusters, True)

    p = Plot(surfaces['lh'], surfaces['rh'])
    p.add_layer(data=sulc, cmap='gray', cbar=False)
    p.add_layer(x, cbar=None, cmap=cmap)
    p.add_layer((np.abs(x) > 0).astype(bool), as_outline=True, 
                cbar=None, cmap='binary')
    fig = p.build()
    fig.savefig(os.path.join(out_dir, f'anova_{prefix}_{n_clusters}_{base}_ensembles'))
    
    p = Plot(surfaces['lh'], surfaces['rh'], views='dorsal', size=(150, 200), 
             zoom=3.3)
    p.add_layer(data=sulc, cmap='gray', cbar=False)
    p.add_layer(x, cbar=None, cmap=cmap)
    p.add_layer((np.abs(x) > 0).astype(bool), as_outline=True, 
                cbar=None, cmap='binary')
    fig = p.build(colorbar=False)
    fig.savefig(os.path.join(out_dir, f'anova_{prefix}_ensembles_{n_clusters}_{base}_dorsal'))

    # posterior views
    p = Plot(surfaces['lh'], surfaces['rh'], views='posterior', 
             size=(150, 200), zoom=3.3)
    p.add_layer(data=sulc, cmap='gray', cbar=False)
    p.add_layer(x, cbar=None, cmap=cmap)
    p.add_layer((np.abs(x) > 0).astype(bool), as_outline=True, 
                cbar=None, cmap='binary')
    fig = p.build(colorbar=False)
    fig.savefig(os.path.join(out_dir, f'anova_{prefix}_ensembles_{n_clusters}_{base}_posterior'))

    data = gradients.merge(df[['roi', 'ensemble']], on='roi', how='left')
    data = data.groupby(['sub', 'epoch', 'ses', 'ensemble'])['distance'] \
               .mean() \
               .reset_index()
    colors = list(_ensemble_cmap(n_clusters).values())
    ensemble_sizes = df['ensemble'].value_counts().to_dict()
    g = sns.FacetGrid(data=data, row='ensemble', col='ses', hue='ensemble',
                      palette=colors, height=2.2, sharey=True, aspect=0.8)
    g.map_dataframe(sns.lineplot, x='epoch', y='distance', errorbar=None, 
                    marker='o', ms=5, lw=1.4, mfc='k', mec='k', color='k')
    
    g.map_dataframe(sns.stripplot, x='epoch', y='distance', jitter=.1, 
                    zorder=-1, s=4, alpha=.6)
    g.set_axis_labels('', "Eccentricity")
    g.set_xticklabels(['Baseline', 'Early', 'Late', 'Washout-early', 'Washout-late'], rotation=90)
    g.set_titles('')
    for i, row_val in enumerate(g.row_names):
        ax = g.axes[i, 0]  # leftmost axis in each row
        n = ensemble_sizes.get(row_val, 0)
        ax.text(1, 1.1, f'n_rois = {n}', transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontweight='bold')
    g.savefig(os.path.join(out_dir, f'ensemble_{n_clusters}_{base}_{prefix}_ecc_plot'))
        
    return df[['roi', 'ensemble']]

def plot_displacements(data, anova, k=3, ax=None, hue='network'):
    """Plot low-dimensional displacements of regions that show significant 
    ANOVA results (i.e. changes in eccentricity)

    Parameters
    ----------
    data : pandas.DataFrame
        Subject gradient data with distance column
    anova : pandas.DataFrame
        ANOVA results
    k : int, optional
        Number of gradients to include, by default 3
    ax : matplotlib.axes._axes.Axes, optional
        Preexisting matplotlib axis, by default None

    Returns
    -------
    matplotlib.figure.Figure and/or matplotlib.axes._axes.Axes
        Displacement scatterplot figure
    """
    if isinstance(k, int):
        k = [f'g{i}' for i in np.arange(k) + 1]


    mean_loadings = data.groupby(['epoch', 'ses', 'roi', 'roi_ix'])[k].mean().reset_index()
    mean_loadings = parse_roi_names(mean_loadings)
    

    base = mean_loadings.query("epoch == 'base' & ses == 'ses-01'")
    sig_regions = anova.loc[anova['sig_corrected'].astype(bool), 'roi']
    sig_base = base[base['roi'].isin(sig_regions)]
    shifts = mean_loadings[mean_loadings['roi'].isin(sig_regions)]

    if hue == 'network':
        cmap = plotting.yeo_cmap(networks=9)
    elif hue == 'ensemble':
        if 'ensemble' not in data.columns:
            raise KeyError("hue='ensemble' requested but 'ensemble' column is missing. "
                           "Run ensemble_analysis() and merge labels into `data` first.")
        ensb = data[['roi', 'ensemble']].groupby('roi', sort=False).first()
        mean_loadings = mean_loadings.merge(ensb, on='roi', how='left')
        n = int(np.max(ensb['ensemble']))
        cmap = _ensemble_cmap(n)
    if len(k) == 2:
        x, y = k
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 3))
        
        # all regions  
        sns.scatterplot(x=x, y=y, data=base, color='k', alpha=.3, s=5, 
                        linewidths=0, legend=False, ax=ax)

        # plot shifts/lines of significant regions
        for roi in shifts['roi'].unique():
            roi_df = shifts.query("roi == @roi")
            xx = roi_df[x].values
            yy = roi_df[y].values
            val = roi_df[hue].iloc[0]
            ax.plot(xx, yy, lw=1, c=cmap[val])
            
            arrowprops = dict(lw=.1, width=.1, headwidth=4, headlength=3, 
                              color=cmap[val])
            ax.annotate(text='', xy=(xx[-1], yy[-1]), xytext=(xx[-2], yy[-2]), 
                        arrowprops=arrowprops)
        
        # plot color-coded markers of significant regions
        sns.scatterplot(x=x, y=y, data=sig_base, hue=hue, s=16, 
                        edgecolor='k', palette=cmap, linewidths=1, ax=ax, 
                        legend=False, zorder=20)
        sns.despine()
        return ax
    
    elif len(k) == 3:
        x, y, z = k
        sns.set(style='whitegrid')
        fig = plt.figure(figsize=(8, 4))
        gs = fig.add_gridspec(nrows=10, ncols=10)
        ax1 = fig.add_subplot(gs[:, :6], projection='3d')

        # remove sig regions so that their points don't obstruct their 
        # colour-coded points plotted below
        base_nonsig = base[~base['roi'].isin(sig_regions)]
        ax1 = plotting.plot_3d(base_nonsig[x], base_nonsig[y], base_nonsig[z],
                                color='gray', alpha=.3, s=1, ax=ax1, 
                                view_3d=(30, -120))
        ax1.set(xticks=range(-4, 6))

        # plot shifts/lines of significant regions
        for roi in shifts['roi'].unique():
            roi_df = shifts.query("roi == @roi")
            xx = roi_df[x].values
            yy = roi_df[y].values
            zz = roi_df[z].values
            val = roi_df[hue].iloc[0]
            ax1.plot(xs=xx, ys=yy, zs=zz, lw=1, c=cmap[val])
        
        # color-coded significant regions
        sig_base['c'] = sig_base[hue].apply(lambda x: cmap[x])
        ax1 = plotting.plot_3d(sig_base[x], sig_base[y], sig_base[z], 
                                color=sig_base['c'], alpha=1, s=20,
                                ax=ax1, zorder=20, edgecolors='k', 
                                linewidths=.5)
        ax1.set(ylim=(-2, 3), xticks=np.arange(-2, 4, 1))
        sns.set(style='darkgrid')
        ax2 = fig.add_subplot(gs[:5, 6:9])
        ax2 = plot_displacements(data, anova, ['g1', 'g2'], ax=ax2)
        ax2.set(ylim=(-3, 3), xlim=(-3, 4), xticklabels=[], 
                xlabel='')
        ax2.set_ylabel('PC2', fontsize=12, fontweight='bold')
        ax3 = fig.add_subplot(gs[5:, 6:9])
        ax3 = plot_displacements(data, anova, ['g1', 'g3'], ax=ax3)
        ax3.set(ylim=(-3, 3), xlim=(-3, 4), xticks=np.arange(-3, 4, 1))
        ax3.set_xlabel('PC1', fontsize=12, fontweight='bold')
        ax3.set_ylabel('PC3', fontsize=12, fontweight='bold')

        fig.tight_layout()
        plt.close()
        return fig, ax
    else:
        return None, None

def plot_sig_region_eccentricity(ttest_stats, gradients, fig_dir):
    '''
    plot eccentricity pattern for significant region for each contrast during all epochs for 
    all subjects
    
    Parameters
    ----------
    data : Pandas DataFrame 
        with eccentrity column for all rois and all subjects 
        and all the existing epochs in the experiment.
    seed : str
        roi name for the desired seed region.

    Returns
    -------
    plot of eccentricity pattern and mean eccentricity for input seed.

    '''
    ttest_stats = parse_roi_names(ttest_stats)
    effect = 'epoch'
    effect_sig_regions = ttest_stats.query('Contrast == @effect & sig_corrected == 1') \
                                .groupby(['A', 'B'])['roi']
    for name, g in effect_sig_regions:
        for hemi in ['LH', 'RH']:
            data = gradients.query('roi in @g.values & hemi == @hemi').copy()
            data = data.groupby(['sub', 'epoch']).mean(numeric_only=True).reset_index()
            fig = plt.figure(figsize=(3, 3))
            sns.lineplot(data=data, x='epoch', y='distance', errorbar=None, 
                         marker='o', ms=6, lw=1.2, mfc='k', mec='k', color='k')
            sns.stripplot(data=data, x='epoch', y='distance', jitter=.1, 
                      zorder=-1, s=5, alpha=.5, hue='epoch')
            plt.xlabel('', fontsize=12, fontweight='bold')
            plt.ylabel('Eccentricity', fontsize=12, fontweight='bold')
            ax = plt.gca()
            # Set the tick labels with rotation and fontsize
            # labels = ['Base', 'Early', 'R',
            #           'Right Learning: Late', 'Left Transfer: Early', 'Left Transfer: Late']
            # ax.set_xticklabels(labels, rotation=90, fontsize=12)
            ax.set_yticks(np.arange(1, 4).astype(int))
            sns.despine()
            plt.show()
            fig_name = f'{effect}_{name[1]}_vs_{name[0]}_{hemi}_sig-regions_ecc'
            fig.savefig(os.path.join(fig_dir, fig_name))
    effect = 'epoch * ses'
    effect_sig_regions = ttest_stats.query('Contrast == @effect & sig_corrected == 1')
    rois = effect_sig_regions['roi']
    data = gradients.query('roi in @rois').copy()
    data = data.groupby(['sub', 'epoch']).mean(numeric_only=True).reset_index()

    fig = plt.figure(figsize=(3, 3))
    sns.lineplot(data=data, x='epoch', y='distance', errorbar=None,
                 marker='o', ms=6, lw=1.2, mfc='k', mec='k', color='k', 
                 alpha=1, legend=False)
    sns.stripplot(data=data, x='epoch', y='distance', jitter=.1, 
                  zorder=-1, s=5, alpha=.7, hue='epoch')
    plt.xlabel('', fontsize=12, fontweight='bold')
    plt.ylabel('Eccentricity', fontsize=12, fontweight='bold')
    ax = plt.gca()
    ax.set_yticks(np.arange(1, 4).astype(int))
    sns.despine()
    fig_name = f'{effect}_sig-regions_ecc'
    fig.savefig(os.path.join(fig_dir, fig_name))
    plt.close()

def count_effects_sig_regions(anova_stats, fig_dir):
    """
    Count the number of significant regions in each network for each source and 
    plot the results as a bar chart.

    Parameters:
    anova_stats (pandas DataFrame): DataFrame containing ANOVA statistics
    fig_dir (str): Directory to save the output figures

    Returns:
    None

    This function first parses the ROI names in the ANOVA statistics, then 
    groups the data by source and network, and counts the number of significant 
    regions in each network for each source. The results are then plotted as a 
    bar chart, with the x-axis showing the networks, the y-axis showing the 
    proportion of significant regions, and the color of the bars indicating the 
    network. The figures are saved to the specified directory.
    """
    anova_stats = parse_roi_names(anova_stats)
    cmap = plotting.yeo_cmap(networks=9)
    network_counts = anova_stats.groupby(['Source', 'network'])['sig_corrected'] \
                                .sum().reset_index()
    for s in network_counts['Source'].unique():
        effect_count = network_counts.query('Source == @s')
        effect_count = effect_count.assign(sig_corrected_ratio = lambda x: x['sig_corrected'] / x['sig_corrected'].sum())
        effect_count.sort_values('sig_corrected_ratio', ascending=False, inplace=True)
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(data=effect_count, x='network', y='sig_corrected_ratio', hue='network',
                    palette=cmap, ax=ax, saturation=.9)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), ha='center', rotation=90, fontsize=12, fontweight='bold')
        # ax.set_yticks(np.arange(0.00, 0.40, 0.05))
        # ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, fontweight='bold')
        sns.despine()
        fig.savefig(os.path.join(fig_dir, f'{s}_sig_regions_ratio'))
    
def plot_sig_region_bold(anova, input_data, fig_dir):
    from saveman.connectivity import (_split_by_learning, _split_by_rotation,
                                            _window_length,
                                            join_sub_cerebellum, _drop_nontask_samps, 
                                            _baseline_window_length)
    from scipy.stats import zscore
    split = 'learning'
    cmap = ['b', 'orange', 'green', 'red', 'purple']
    timeseries = get_files(input_data + '/*.tsv')
    sub_bold = []
    for ts in timeseries:
        fname = os.path.split(ts)[1]
        data = pd.read_table(ts)
        data = join_sub_cerebellum(data, fname)
        tseries = data.values
        regions = data.columns
        # handle different scan types
        if 'rotation' in fname:
            tseries = _drop_nontask_samps(tseries)
            
            if split == 'learning':
                dataset = _split_by_learning(tseries)
            elif split == 'rotation':
                dataset = _split_by_rotation(tseries)
            else:
                dataset = [tseries]
    
        elif 'washout' in fname:
            dataset = [_drop_nontask_samps(tseries)]
            if _window_length() < _baseline_window_length() // 2:
                dataset = _split_by_learning(tseries, include_base=False)
            else:
                dataset = [tseries[:_window_length()]]
        else:
            dataset = [tseries]
        n_matrices = len(dataset)
        if n_matrices == 2 and split == 'learning':
            epochs = ['washout-early', 'washout-late']
        elif n_matrices == 3 and split == 'learning':
            epochs = ['base', 'early', 'late']
        elif n_matrices == 2 and split=='rotation':
            epochs = ['base', 'learning']
        else:
            epochs = ['']
        sub = fname.split('_')[0]
        ses = fname.split('_')[1]
        for epoch, cmat in zip(epochs, dataset):
            df = pd.DataFrame(np.mean(cmat.T, axis=1), 
                              index=regions, columns=['tmean']).reset_index() \
                .rename(columns={'index': 'roi'})
            df[['sub', 'epoch', 'ses']] = [sub, epoch, ses]
            sub_bold.append(parse_roi_names(df))
    sub_bold = pd.concat(sub_bold)
    sub_bold_test = sub_bold.groupby(['roi', 'epoch', 'ses'])['tmean'].apply(zscore).reset_index()
    sub_bold['tmean'] = sub_bold_test['tmean'].values

    anova = parse_roi_names(anova)
    for effect in ['epoch']:
        sig_regions = anova.query('Source == @effect & sig_corrected == 1')['roi']
        data = sub_bold.query('roi in @sig_regions') \
                        .groupby(['sub', 'epoch', 'ses']).mean(numeric_only=True).reset_index()
        g = sns.FacetGrid(data=data, col_wrap=2, col='ses', hue='ses', palette=['darkcyan', '#FFA066'],
                          height=2.5, sharey=True, aspect=.75)
        g.map_dataframe(sns.lineplot, x='epoch', y='tmean', errorbar=None, 
                        marker='o', ms=6, lw=1.3, mfc='k', mec='k', color='k')
        g.map_dataframe(sns.stripplot, x='epoch', y='tmean', jitter=.1, 
                        zorder=-1, s=4, alpha=.7)
        g.set_axis_labels('', "Z-score")
        g.set_xticklabels(['Baseline', 'Early', 'Late', 'Washout-early', 'Washout-late'], rotation=90)
        # Set y-ticks for each axis in the FacetGrid
        for ax in g.axes.flat:
            ax.set_yticks(np.arange(-0.3, 0.4, 0.1))

        for ax, title in zip(g.axes.flat, g.col_names):
            ax.set_title(title.replace('ses = ', ''))
        g.savefig(os.path.join(fig_dir, f'{effect}_sig-regions_bold'))

def plot_anova_corr(df_original, df_clean, source="epoch", max_roi=400):
    """
    Compare voxel/parcel-wise ANOVA F-statistics before and after
    ventral visual regression.

    Parameters
    ----------
    df_original : pd.DataFrame
        ANOVA results without the visual-bleeding correction.
    df_clean : pd.DataFrame
        ANOVA results after the visual-bleeding correction.
    source : str, optional
        Level of the ANOVA factor to keep (default: 'epoch').
    max_roi : int, optional
        Upper bound on roi index to include (default: 400).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    r, p : float
        Pearson correlation and p-value between F-statistics.
    """

    # Select the relevant rows and rename F columns
    df_original = (
        df_original
        .loc[(df_original["Source"] == source) & (df_original["roi_ix"] <= max_roi)]
        .rename(columns={"F": "F_original"})
    )

    df_clean = (
        df_clean
        .loc[(df_clean["Source"] == source) & (df_clean["roi_ix"] <= max_roi)]
        .rename(columns={"F": "F_corrected"})
    )

    # Merge on ROI label
    df_all = pd.merge(
        df_original[["roi", "F_original"]],
        df_clean[["roi", "F_corrected"]],
        on="roi",
        how="inner",
    ).dropna()

    # Correlation
    r, p = pearsonr(df_all["F_original"].values, df_all["F_corrected"].values)
    # Plot
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.regplot(
        data=df_all,
        x="F_original",
        y="F_corrected",
        # ax=ax,
        scatter_kws={'color':'k', 'alpha': 1.0},
        line_kws={"linewidth": 1.5, 'color': 'k'},
    )

    # 1:1 line for reference
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, linestyle="--", color="gray", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("F-statistic (original ANOVA)", fontsize=11, fontweight="bold")
    ax.set_ylabel("F-statistic (visual-corrected ANOVA)", fontsize=11, fontweight="bold")
    sns.despine()

    # Annotate r and p on the plot
    ax.text(
        0.05,
        0.95,
        f"r = {r:.2f}\np = {p:.3g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.8),
    )

    fig.tight_layout()
    plt.close()
    return fig

def main():
    config = Config()
    plotting.set_plotting()

    gradients = pd.read_table(
        os.path.join(config.results, 'subject_gradients.tsv')
    )

    fig_dir = os.path.join(Config().figures, 'adaptation')
    os.makedirs(fig_dir, exist_ok=True)
       
    # two way anova stats + ttest posthoc    
    anova_stats, posthoc_ttest = eccentricity_analysis(gradients, 
                                                                 method='anova',
                                                                 factor=['epoch', 'ses'])
    #droping subcortical and cerbellar regions
    anova_stats_cor = anova_stats.query('roi_ix <= 400')
    posthoc_ttest_cor = posthoc_ttest.query('roi_ix <= 400')
    plot_sig_region_bold(anova_stats, config.tseries, fig_dir)
    count_effects_sig_regions(anova_stats, fig_dir)
    
    anova_stats.to_csv(os.path.join(config.results, 'ecc_anova_stats.tsv'), 
                       sep='\t', index=False)
    posthoc_ttest.to_csv(os.path.join(config.results, 'ecc_posthoc_ttest.tsv'), 
                       sep='\t', index=False)
    
    vmax_sig = int(np.nanmax(anova_stats_cor.query('sig_corrected == 1')['F']))
    vmin_sig = np.nanmin(anova_stats_cor.query('sig_corrected == 1')['F'])

    plt.close('all')
    vmax_sig = np.nanmax(-posthoc_ttest.query('sig_corrected == 1')['T'])
    vmin_sig = np.nanmin(np.abs(-posthoc_ttest.query('sig_corrected == 1')['T']))
    for f in posthoc_ttest_cor['Contrast'].unique():
        temp = posthoc_ttest_cor[posthoc_ttest_cor['Contrast'] == f]
        if f in ['epoch', 'ses']:
            plotting.pairwise_stat_maps(data=None, data_posthoc=temp,
                                    prefix=os.path.join(fig_dir, f'{f}_ecc_ttests_'),
                                    vmax=vmax_sig, vmin=vmin_sig)
        elif f in ['epoch * ses']:
            for h in temp.iloc[:, 3].unique():
                plotting.pairwise_stat_maps(data=None, data_posthoc=temp[temp.iloc[:, 3]==h],
                                    prefix=os.path.join(fig_dir, f'{f}_{h}_ecc_ttests_'),
                                    vmax=vmax_sig, vmin=vmin_sig)
                                
    epoch_effect = anova_stats.query('Source == "epoch"')
    ensb_4 = ensemble_analysis(gradients, epoch_effect, fig_dir, config.k, prefix='epoch', base='base', use_optimal_k=True)
    ensb_4.columns = ['roi', 'ensemble_4']
    anova_stats_ensemble = pd.merge(anova_stats, ensb_4, on='roi', how='left')
    anova_stats_ensemble.to_csv(os.path.join(config.results, 'ecc_anova_stats_ensemble.tsv'), 
                       sep='\t', index=False)
    anova_stats = anova_stats.query('Source == "epoch"')
    if config.k == 3:
        fig, _ = plot_displacements(gradients, anova_stats, config.k)
        fig.savefig(os.path.join(fig_dir, 'displacements_epoch'))

if __name__ == '__main__':
    main()
