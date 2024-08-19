import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as Patch
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, SparsePCA

class AutoPCA():
    """
    creates an pca object and provides a funciton for plottig a biplot

    Inputs:
        - ``data``: ``DataFrame`` of data supplies. Should be numeric and contain the sample name in one column
        - ``target``: ``str`` of the column containing the sample name
        - ``sparse``: ``bool`` of default ``False`` to use sprase PCA
        - ``alpha`` : ``float`` for alpha value of sparse PCA.

    Methodes:
        - ``biplot``: creates a biplot on supplies axes. If no axis supplied, a new one is created

    Notes:
        - When using a sparsePCA for the analysis the explained varience of each component becoms meaning less as
        the components are not orthogonal anymore. Owing to this, the explaind variance would overlapp. Thus, 
        only the total explained variance is worth reporting, suggesting how good the overall model is with 
        two components (as is used when generating a PCA() or sparsePCA() object). (trace(P @ T.T @ T @ P.T) is
        variance of model). Additinaly the loadings are not the same as the rotation, as loadings are not 
        orthogonal as such the Rotations R = P @ (P.T @ P)^-1

    """
    def __init__(self, data: pd.DataFrame, target: str, scale = True, sparse = False, alpha = 1):

        self.data = data
        self.target = target
        self.sparse = sparse
        self.scale = scale

        self.X = self.data.copy() \
            .drop(self.target, axis=1)
        self.y = self.data.copy() \
            [self.target]
        
        self.X = self.X.dropna(axis=1)
        
        if self.scale == True:
            standard_scaler = StandardScaler()
            X_scaled = standard_scaler.fit_transform(self.X)
        else:
            X_scaled = self.X
        
        if sparse == False:
            pca = PCA(n_components=2).fit(X_scaled)
        else:
            pca = SparsePCA(n_components=2, random_state=112, alpha=alpha).fit(X_scaled)

        self.pca = pca
        self.X_reduced = pca.transform(X_scaled)
        self.scores = self.X_reduced[:, :2]
        self.loadings = pca.components_[:2].T

        total_variance = np.trace(X_scaled.T @ X_scaled)
        explained_variance = np.trace(self.X_reduced @ self.loadings.T @ self.loadings @ self.X_reduced.T)
        self.explained_variance_ratio = explained_variance/total_variance

        if sparse == False:
            self.pvars = pca.explained_variance_ratio_[:2] * 100

    def return_loadings(self):
        loadings = pd.DataFrame(self.loadings.copy(), columns = ['Comp1', 'Comp2'])
        loadings['variable'] = self.X.columns
        return loadings
        
    def biplot(self, axs = None):
        arrow = self.loadings * np.abs(self.scores).max(axis=0)

        if axs is None:
            fig, axs = plt.subplots(figsize=(8,6))

        plt.sca(axs)

        # cols = self.y.iloc[:,0].drop_duplicates().to_list()
        # color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        # colors = {color : color_list[i] for i, color in enumerate(cols)}
        
        # shps  = self.y.iloc[:,1].drop_duplicates().to_list()
        # shape_list = ['o', '^', 's', 'p', 'D', 'v', '<', '>']
        # shapes = {shape : shape_list[i] for i, shape in enumerate(shps)}

        # for name in self.y.iloc[:,0].drop_duplicates().to_list():
        #     for shape in self.y.iloc[:,1].drop_duplicates().to_list():
        #         axs.scatter(*zip(*self.scores[(self.y.iloc[:,0] == name) & (self.y.iloc[:,1] == shape)]), label=None, marker = shapes[shape], color = colors[name])

        # for name in self.y.iloc[:,0].drop_duplicates().to_list():
        #     axs.scatter(*zip(*self.scores[(self.y.iloc[:,0] == name)]), label=None, color = colors[name])

        axs.scatter(*zip(*self.scores), c = np.array(self.y), cmap="plasma")


        #color_handles = [Line2D([0], [0], marker = 's', markersize=8, linestyle='None', label = cols[i], color=color) for i, color in enumerate(colors.values())]
        #color_legend = axs.legend(handles = color_handles, loc = 'lower center', ncols = len(cols), bbox_to_anchor=(0.5, -0.13),  frameon=False)

        # axs.legend(labels=shapes.keys(), loc = 'lower center', ncols = len(shps), bbox_to_anchor=(0.5, -0.2), frameon=False)
        #axs.add_artist(color_legend)

        width = -0.003 * np.min([np.subtract(*plt.xlim()), np.subtract(*plt.ylim())])
        for i, arrow in enumerate(arrow):
            axs.arrow(0, 0, *arrow, color='k', alpha=0.5, width = width, ec='none',
                    length_includes_head=True)
            axs.text(*(arrow * 1.05), self.X.columns[i],
                    ha='center', va='center', fontsize = 7)
            
        for i, axis in enumerate('xy'):
            getattr(plt, f'{axis}ticks')([])
            if self.sparse == False:
                getattr(plt, f'{axis}label')(f'PC{i + 1} ({self.pvars[i]:.2f}%)')
            else:
                getattr(plt, f'{axis}label')(f'PC{i + 1}')

        return axs
    
    def scatter_plot(self, axs = None):
       
        if axs is None:
            fig, axs = plt.subplots(figsize=(8,6))

        plt.sca(axs)

        cols = self.y.iloc[:,0].drop_duplicates().to_list()
        color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        colors = {color : color_list[i] for i, color in enumerate(cols)}
        
        shps  = self.y.iloc[:,1].drop_duplicates().to_list()
        shape_list = ['o', '^', 's', 'p', 'D', 'v', '<', '>']
        shapes = {shape : shape_list[i] for i, shape in enumerate(shps)}

        for name in self.y.iloc[:,0].drop_duplicates().to_list():
            for shape in self.y.iloc[:,1].drop_duplicates().to_list():
                axs.scatter(*zip(*self.scores[(self.y.iloc[:,0] == name) & (self.y.iloc[:,1] == shape)]), label=None, marker = shapes[shape], color = colors[name])

        color_handles = [Line2D([0], [0], marker = 's', markersize=8, linestyle='None', label = cols[i], color=color) for i, color in enumerate(colors.values())]
        color_legend = axs.legend(handles = color_handles, loc = 'lower center', ncols = len(cols), bbox_to_anchor=(0.5, -0.13),  frameon=False)

        axs.legend(labels=shapes.keys(), loc = 'lower center', ncols = len(shps), bbox_to_anchor=(0.5, -0.2), frameon=False)
        axs.add_artist(color_legend)
            
        for i, axis in enumerate('xy'):
            getattr(plt, f'{axis}ticks')([])
            if self.sparse == False:
                getattr(plt, f'{axis}label')(f'PC{i + 1} ({self.pvars[i]:.2f}%)')
            else:
                getattr(plt, f'{axis}label')(f'PC{i + 1}')

        return axs