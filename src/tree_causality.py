from numpy.lib.function_base import corrcoef
from statsmodels.tsa.stattools import grangercausalitytests as gct
from minepy import MINE
from scipy.stats import pearsonr
import numpy as np


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
        print('get adjacency matrix')
        self.get_adjacency_matrix(inputs)
        print('...done...')
        print('turn adjacency matrix to tree causality')
        self.adjacency_to_tree()
        print('...done...')
        print('get input tree structure of Causal LSTM')
        child_input_idx = self.get_input_idx()
        children = self.get_num_nodes()
        child_state_idx = self.get_state_idx()
        print('...done...')
        print(self.adjacency_matrix)
        return children, child_input_idx, child_state_idx

    def _init_adjacency_matrix(self):
        self.adjacency_matrix = np.zeros(
            (self.num_features, self.num_features))

    def get_adjacency_matrix(self, inputs):
        self._init_adjacency_matrix()

        if self.flag[0] == 1:
            corrcoef_matrix, sign_corr_matrix = self.linear_correlation_test(
                inputs)
        if self.flag[1] == 1:
            mic_matrix, sign_mic_matrix = self.mic_test(
                inputs)
        if self.flag[2] == 1:
            granger_matrix, sign_granger_matrix = self.linear_granger_test(
                inputs)

        import seaborn as sns
        
        sns.heatmap(corrcoef_matrix, square=True, annot=True)
        plt.savefig('1.pdf')
       
        sns.heatmap(mic_matrix, square=True, annot=True)
        plt.savefig('2.pdf')

        sns.heatmap(granger_matrix, square=True, annot=True)
        plt.savefig('3.pdf')
               
        
        try:
            sign_corr_matrix = sign_corr_matrix + sign_mic_matrix
            sign_corr_matrix[sign_corr_matrix == 2] = 1
        except:
            print('This case do not use mic!')
            
            
        sns.heatmap(sig_granger_matrix, annot=True)


        if self.flag[2] == 0:
            self.adjacency_matrix = sign_corr_matrix
        else:
            self.adjacency_matrix = sign_corr_matrix+sign_granger_matrix
            self.adjacency_matrix[self.adjacency_matrix == -1] = 0

        for i in range(self.num_features):
            self.adjacency_matrix[i, i] = 0

        # if root node have no children.
        child_root = self.adjacency_matrix[:, -1]
        child_corr_root = corrcoef_matrix[:-1, -1]

        if np.sum(child_root) == 0:
            i = np.argmax(np.abs(child_corr_root))
            self.adjacency_matrix[i, self.num_features-1] = 1
            self.adjacency_matrix[self.num_features-1, i] = 1

        return self.adjacency_matrix

    def linear_correlation_test(self, inputs):
        """linear correlation test.

        Args:
            inputs ([type]): [description]

        Returns:
            [type]: [description]
        """
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
        """max Information-based Nonparametric Exploration.

        Args:
            inputs ([type]): [description]
        """
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

    def te_test(self, inputs):
        pass

    def linear_granger_test(self, inputs, maxlag=2):
        """Linear granger causality test.

        Args:
            inputs ([type]): [description]
        """
        # init
        granger_matrix = np.full(
            (self.num_features, self.num_features), np.nan)
        sign_matrix = np.zeros(
            (self.num_features, self.num_features))

        for i in range(self.num_features):
            for j in range(self.num_features):
                gc = gct(inputs[:, [i, j]], maxlag=maxlag)

                for t in range(maxlag):
                    F = gc[t+1][0]['ssr_ftest'][0]
                    p = gc[t+1][0]['ssr_ftest'][1]

                    if p > 0.01:
                        sign_matrix[i, j] = -1

        return granger_matrix, sign_matrix

    def nonlinear_granger_test(self, inputs):
        pass

    def get_causal_driver(self, nums):
    
        a = self.adjacency_matrix[:,nums]
        idx = np.where(a==1)[0]
    
        return [i for i in idx]
    
    def adjacency_to_tree(self):
        """Change adjacency matrix to tree causality.

        Args:

        Returns:
            tree: {LV1: [[-1]],
                   LV2: [[2, 3, 4]],
                   LV3: [[3, 4, 5], [1,2], [3,4]]}
        """
        #self.adjacency_matrix[5,-1] = 1 # turn precipitation to layer 2.
        #self.adjacency_matrix[10,-1] = 1 # turn precipitation to layer 2.

        self.tree['1'] = [[self.num_features-1]]
        

        for level in np.arange(1,self.depth+1):
            print(level)
            self.tree[str(level+1)] = []
            for node_group in self.tree[str(level)]:
                for node in node_group:
                    self.tree[str(level+1)].append(self.get_causal_driver(node))
        print(self.adjacency_matrix)
        print(self.tree)
        
        """
        self.tree['2'] = [list(np.where(self.adjacency_matrix[:, -1] == 1)[0])]
        self.tree['3'] = []

        for i in self.tree['2']:
            for j in i:
                self.tree['3'].append(
                    list(np.where(self.adjacency_matrix[:, j] == 1)[0]))
        """
        
    def get_num_nodes(self):
        """Get num of nodes in each 
        """
        children = []
        
        for node_group in self.tree[str(self.depth+1)]:
            for node in node_group:
                children.append(0)
                
        for level in np.arange(self.depth, 0, -1):
            print(level)
            for node_group in self.tree[str(level+1)]:
                children.append(len(node_group))
        print(children)
        """
        for i in self.tree['3']:
            for j in i:
                children.append(0)
        for i in self.tree['3']:
            children.append(len(i))

        children.append(len(self.tree['2'][0]))
        """
        return children

    def get_input_idx(self):
        child_input_idx = []
        
        for level in np.arange(self.depth+1,0,-1):
            for node_group in self.tree[str(level)]:
                for node in node_group:
                    child_input_idx.append([int(node)])   
        print(child_input_idx)
        """
        for i in self.tree['3']:
            for j in np.array(i):
                child_input_idx.append([j])

        for i in self.tree['2']:
            for j in np.array(i):
                child_input_idx.append([j])

        for i in self.tree['1']:
            child_input_idx.append(i)
        """
        return child_input_idx

    def get_state_idx(self):
        child_state_idx = []

        for node_group in self.tree[str(self.depth+1)]:
            for node in node_group:
                child_state_idx.append([])
                
        count = -1
        
        for level in np.arange(self.depth,0,-1):
            for node_group in self.tree[str(level+1)]:
                _idx = []
                for node in node_group:
                    count += 1
                    _idx.append(count)
                
                child_state_idx.append(_idx)    
        """     
        for node_group in self.tree[]
        for i in self.tree['3']:
            for j in i:
                child_state_idx.append([])

        count = -1
        for i in self.tree['3']:
            _idx = []
            for j in i:
                count += 1
                _idx.append(count)
            child_state_idx.append(_idx)

        for i in self.tree['2']:
            _idx = []
            for j in i:
                count += 1
                _idx.append(count)

        child_state_idx.append(_idx)
        """
        return child_state_idx
