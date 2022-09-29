import numpy as np
from minepy import MINE
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import grangercausalitytests as gct


class CausalTree():
    def __init__(self,
                 num_features,
                 name_features,
                 corr_thresold=0.5,
                 mic_thresold=0.5,
                 flag=[1, 0, 0],
                 depth=2):
        self.num_features = num_features
        self.name_features = name_features
        self.corr_thresold = corr_thresold
        self.mic_thresold = mic_thresold
        self.flag = flag
        self.tree = {}
        self.depth = depth

    def __call__(self, inputs):
        print('[CLSTM] get adjacency matrix')
        self.adjacency_matrix = self.get_adjacency_matrix(inputs)
        print('[CLSTM] turn adjacency matrix to tree causality')
        self.adjacency_to_tree()
        print('[CLSTM] get input tree structure of Causal LSTM')
        child_input_idx = self.get_input_idx()
        children = self.get_num_nodes()
        child_state_idx = self.get_state_idx()
        return self.tree, \
            children, child_input_idx, child_state_idx, self.adjacency_matrix

    def get_adjacency_matrix(self, inputs):
        # init adjacency matrix
        self.adj_matrix = np.zeros((self.num_features, self.num_features))

        # perform correlation/mic/granger test based on flags
        if self.flag[0] == 1:
            corr, sign_corr_matrix = self.linear_correlation_test(inputs)
        if self.flag[1] == 1:
            _, sign_mic_matrix = self.mic_test(inputs)
        if self.flag[2] == 1:
            _, sign_granger_matrix = self.granger_test(inputs)

        # combine the results of different tests
        if self.flag[1] == 1:  # combine corr/mic test
            sign_corr_matrix = sign_corr_matrix + sign_mic_matrix
            sign_corr_matrix[sign_corr_matrix == 2] = 1

        if self.flag[2] == 0:
            self.adj_matrix = sign_corr_matrix
        else:
            self.adj_matrix = sign_corr_matrix+sign_granger_matrix
            self.adj_matrix[self.adj_matrix == -1] = 0

        # remove causality of same features in adj matrix
        for i in range(self.num_features):
            self.adj_matrix[i, i] = 0

        # NOTE: if root node have no children, we used the most significant
        #       correlation nodes as children nodes and construst corresponding
        #       adjacency matrix.
        child_root = self.adj_matrix[:, -1]
        child_corr_root = corr[:-1, -1]
        if np.sum(child_root) == 0:
            i = np.argmax(np.abs(child_corr_root))
            self.adj_matrix[i, self.num_features-1] = 1
            self.adj_matrix[self.num_features-1, i] = 1

        return self.adj_matrix

    def linear_correlation_test(self, inputs):
        """linear correlation test."""
        # init
        corrcoef_matrix = np.full(
            (self.num_features, self.num_features), np.nan)
        sign_matrix = np.zeros(
            (self.num_features, self.num_features))

        # corr & sign matrix
        for i in range(self.num_features):
            for j in range(self.num_features):
                corr, p = pearsonr(inputs[:, i], inputs[:, j])
                corrcoef_matrix[i, j] = corr
                if corr > self.corr_thresold and p < 0.05:
                    sign_matrix[i, j] = 1
        return corrcoef_matrix, sign_matrix

    def mic_test(self, inputs):
        """max Information-based Nonparametric Exploration."""
        # init
        mic_matrix = np.full(
            (self.num_features, self.num_features), np.nan)
        sign_matrix = np.zeros(
            (self.num_features, self.num_features))

        # mic & sign matrix
        for i in range(self.num_features):
            for j in range(self.num_features):
                if i != j:
                    mine = MINE(alpha=0.6, c=15)
                    mine.compute_score(inputs[:, i], inputs[:, j])
                    mic = mine.mic()

                    mic_matrix[i, j] = mic
                    if mic > self.mic_thresold:
                        sign_matrix[i, j] = 1
        return mic_matrix, sign_matrix

    def granger_test(self, inputs, maxlag=2):
        """Linear/Non-linear granger causality test."""
        # init
        granger_matrix = np.full(
            (self.num_features, self.num_features), np.nan)
        sign_matrix = np.zeros(
            (self.num_features, self.num_features))

        for i in range(self.num_features):
            for j in range(self.num_features):
                sig = self.vanilla_gc_test(inputs[:, i], inputs[:, j], maxlag)
                sign_matrix[i, j] = sig

        return granger_matrix, sign_matrix

    def get_causal_driver(self, nums):
        return [i for i in np.where(self.adj_matrix[:, nums] == 1)[0]]

    def adjacency_to_tree(self):
        """Change adjacency matrix to tree causality.

        Returns:
            tree: {LV1: [[-1]],
                   LV2: [[2, 3, 4]],
                   LV3: [[3, 4, 5], [1,2], [3,4]]}
        """
        # NOTE: enforce precipitation into level 2. see Section 4.f
        # self.adj_matrix[10,-1] = 1

        # soil moisture as root, level 1
        self.tree['1'] = [[self.num_features-1]]

        # generate tree, level 2 - level depth
        for level in np.arange(1, self.depth+1):
            print(level)
            self.tree[str(level+1)] = []
            for node_group in self.tree[str(level)]:
                for node in node_group:
                    m = self.get_causal_driver(node)
                    self.tree[str(level+1)].append(m)

    def get_num_nodes(self):
        """Get num of child nodes for each nodes"""
        children = []
        for node_group in self.tree[str(self.depth+1)]:
            for node in node_group:
                children.append(0)
        for level in np.arange(self.depth, 0, -1):
            for node_group in self.tree[str(level+1)]:
                children.append(len(node_group))
        return children

    def get_input_idx(self):
        """Get input feat idx of each nodes"""
        child_input_idx = []

        for level in np.arange(self.depth+1, 0, -1):
            for node_group in self.tree[str(level)]:
                for node in node_group:
                    child_input_idx.append([int(node)])
        return child_input_idx

    def get_state_idx(self):
        """Get state idx of each nodes"""
        child_state_idx = []
        for node_group in self.tree[str(self.depth+1)]:
            for node in node_group:
                child_state_idx.append([])

        count = -1
        for level in np.arange(self.depth, 0, -1):
            for node_group in self.tree[str(level+1)]:
                _idx = []
                for node in node_group:
                    count += 1
                    _idx.append(count)
                child_state_idx.append(_idx)
        return child_state_idx

    def vanilla_gc_test(self, X, y, maxlags):
        """Linear/Nonlinear granger causality test.

        NOTE: The test is an vanilla GC test on one grids
              (X/y should be time series). We do not perform cross validation 
              to get the best both baseline and full models. Full edition
              of nonlinear gc test refer to GitHub:
              h-cel/ClimateVegetationDynamics_GrangerCausality
        """
        sig = 0
        x_lag = np.roll(y, maxlags)
        X = np.concatenate([X, x_lag], axis=1)

        # linear
        mdl = Ridge()
        mdl.fit(x_lag, y)
        y_pred_baseline = mdl.predict(x_lag)
        r2_baseline = r2_score(y, y_pred_baseline)
        mdl = Ridge()
        mdl.fit(X, y)
        y_pred_full = mdl.predict(X)
        r2_full = r2_score(y, y_pred_full)
        if r2_full < r2_baseline:
            sig = -1

        if sig == 0:
            # nonlinear
            mdl = RandomForestRegressor(n_estimators=100)
            mdl.fit(x_lag, y)
            y_pred_baseline = mdl.predict(x_lag)
            r2_baseline = r2_score(y, y_pred_baseline)
            mdl = RandomForestRegressor(n_estimators=100)
            mdl.fit(X, y)
            y_pred_full = mdl.predict(X)
            r2_full = r2_score(y, y_pred_full)
            if r2_full < r2_baseline:
                sig = -1
        return sig
