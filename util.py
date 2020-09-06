import numpy as np
import pandas as pd
import sqlite3
import re
from matplotlib import pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.spatial.distance import cdist
import scipy.sparse as sp
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import torch

### Define SQL statements
def match_sql():
    return 'SELECT * FROM Match'

def player_sql():
    return """SELECT p.birthday, p.height, p.weight, a.*
             FROM Player p, Player_Attributes a
             WHERE p.player_api_id = a.player_api_id
                 AND p.player_fifa_api_id = a.player_fifa_api_id
             ORDER BY a.player_api_id, date"""

### Read in data
def ingest_match(con):
    return pd.read_sql(match_sql(), con=con)

def ingest_player(con):
    return pd.read_sql(player_sql(), con)

### Clean data
def clean_match(match):
    # Drop betting odds columns
    bet_cols = match.columns[list(match.columns).index('B365H'):]
    match = match.drop(bet_cols, axis=1)
    
    # Create outcome column
    match['outcome'] = np.where(match.home_team_goal > match.away_team_goal,
                                'home_win',
                                np.where(match.home_team_goal == match.away_team_goal,
                                         'draw',
                                         'home_loss'))
    
    match = pd.get_dummies(match, columns = ['outcome'])
    
    # Create sin & cos date vars
    match['date'] = pd.to_datetime(match.date)
    match['sin_date'] = np.sin(2*np.pi*match.date.dt.dayofyear/365)
    match['cos_date'] = np.cos(2*np.pi*match.date.dt.dayofyear/365)
    
    # Rename some player columns to work with pd.wide_to_long
    match.columns = [re.sub('^(.*)_player_(\\d{,2})$',
                            '\\1_player_id\\2',
                            i) for i in match.columns]
    
    # Drop rows with any null values for ids or X/Y coords
    player_cols = [i for i in match.columns if 'player' in i]
    match = match.loc[~match[player_cols].isnull().any(axis=1)]
    
    return match

def clean_player(player):
    # Create player age in years
    player['date'] = pd.to_datetime(player.date)
    player['birthday'] = pd.to_datetime(player.birthday)
    player['age'] = (player.date - player.birthday).dt.days/365
    
    # Clean FIFA work rates
    work_rate_keeps = ['high', 'medium', 'low']
    player['attacking_work_rate'] = np.where(~player.attacking_work_rate.isin(work_rate_keeps),
                                             'uncommon',
                                             player.attacking_work_rate)
    
    player['defensive_work_rate'] = np.where(~player.defensive_work_rate.isin(work_rate_keeps),
                                             'uncommon',
                                             player.defensive_work_rate)
    
    player = pd.get_dummies(player, drop_first=True)
    
    drop_cols = ['birthday',
                 'player_fifa_api_id',
                 'id']
    
    player = player.drop(drop_cols, axis=1).sort_values('date')
    
    return player

### Functions to transform X/Y coords of certain players
# Keeper X coords are too far left
def keeper_xfrm(df):
    df['away_player_X'] = np.where(df.player == 1, 5, df.away_player_X)
    df['home_player_X'] = np.where(df.player == 1, 5, df.home_player_X)
    
    return df

# Flip away player Y coords about a horizontal axis
def away_xfrm(df):
    df['away_player_Y'] = -1*(df.away_player_Y - 11) + 3
    
    return df
    
### Join match date to long match data
def add_match_vars(long_match, match_vars):
    long_match = long_match.merge(match_vars,
                                  on='match_api_id').sort_values('date')
    
    return long_match

### Join player attributes to long match data
def add_player_attrs(long_match, player):
    return pd.merge_asof(long_match,
                          player,
                          on='date',
                          by='player_api_id').dropna(subset=['crossing',
                                                             'height'])

# Transform match from wide to long format to set up graph
def transform_match(wide_match, player):
    # Perform first pd.wide_to_long
    player_cols = [i for i in wide_match.columns if 'player' in i]
    long_match = pd.wide_to_long(wide_match[['match_api_id'] + player_cols],
                               stubnames=['home_player_id',
                                          'home_player_X',
                                          'home_player_Y',
                                          'away_player_id',
                                          'away_player_X',
                                          'away_player_Y'],
                               i='match_api_id',
                               j='player').reset_index()
    
    # Transform keeper X & Y coords
    long_match = keeper_xfrm(long_match)
    
    # Transform away player X & Y coords
    long_match = away_xfrm(long_match)
    
    # Rename columns to work with pd.wide_to_long
    long_match.columns = [re.sub('(.*)_player_(.*)',
                            '\\2_\\1',
                            i) for i in long_match.columns]
    
    # Perform second pd.wide_to_long
    long_match = pd.wide_to_long(long_match,
                            stubnames=['X',
                                       'Y',
                                       'id'],
                            i=['match_api_id',
                               'player'],
                            j='home_away',
                            sep='_',
                            suffix='[a-z]*')
    
    long_match = long_match.reset_index().sort_values(['match_api_id',
                                             'player'])
    
    # Reformat player_api_id & home_away
    long_match = long_match.rename({'id': 'player_api_id',
                                    'home_away': 'home'},
                                   axis=1)
    
    long_match['player_api_id'] = long_match.player_api_id.astype(np.int64)
    long_match['home'] = np.where(long_match.home == 'home',
                                       1,
                                       0)
    
    long_match = add_match_vars(long_match, wide_match[['match_api_id',
                                                        'date',
                                                        'outcome_home_win',
                                                        'outcome_draw',
                                                        'outcome_home_loss']])
    
    match_player = add_player_attrs(long_match, player)
    
    # Ensure all matches have 22 players
    incomplete_matches = match_player.loc[match_player.groupby('match_api_id')['player_api_id'].transform('nunique') < 22].match_api_id.unique()
    
    match_player = match_player.loc[~(match_player.match_api_id.isin(incomplete_matches))].sort_values(['match_api_id',
                                                                                                        'home',
                                                                                                        'player'])
    
    # Fill nan with mean of column for that match
    for i in list(match_player.columns[match_player.isna().any()]):
        match_player[i] = match_player[i].fillna(match_player.groupby('match_api_id')[i].transform('mean'))
    
    return match_player

### Plot one match in long format
def plot_match(one_match):
    plt.figure(figsize=(16,8))
    sns.scatterplot(x='X',
                    y='Y',
                    hue='home_away',
                    data=one_match)
    plt.title(f'Match: {one_match.match_api_id.iloc[0]}')
    plt.show()

### Ingest data
def ingest_data(return_all=False):
    try:
        con = sqlite3.connect('database.sqlite')
    except Exception as e:
        print('Could not connect to database.sqlite')
        raise
    else:
        print('Connected to database.sqlite')
    
    try:
        match = clean_match(ingest_match(con))
    except Exception as e:
        print('Error loading and cleaning match data')
        raise
    else:
        print('Loaded and cleaned match data')
    
    try:
        player = clean_player(ingest_player(con))
    except Exception as e:
        print('Error loading and cleaning player data')
        raise
    else:
        print('Loaded and cleaned player data')
    
    con.close()
    try:
        match_player = transform_match(match, player)
    except Exception as e:
        print('Error transforming match data')
        raise
    
    if return_all:
        return match, player, match_player
    else:
        return match_player

### Get train/test splits of features & targets
def get_splits(match_player, test_size=.20, random_state=42):
    train_inds, test_inds = next(GroupShuffleSplit(test_size=test_size,
                                                   n_splits=2,
                                                   random_state=random_state).split(match_player,
                                                                                    groups=match_player['match_api_id']))
    
    train_match = match_player.iloc[train_inds]
    test_match = match_player.iloc[test_inds]
    
    target_cols = [i for i in match_player.columns if 'outcome_' in i] + ['match_api_id']
    drop_cols = target_cols + ['date', 'player', 'player_api_id']
    
    x_train = train_match.drop(drop_cols,
                               axis=1)
    y_train = train_match[target_cols].drop_duplicates().drop('match_api_id',
                                                               axis=1)
    train_ids = list(train_match.match_api_id.unique())
    
    x_test = test_match.drop(drop_cols,
                             axis=1)
    y_test = test_match[target_cols].drop_duplicates().drop('match_api_id',
                                                            axis=1)
    test_ids = list(test_match.match_api_id.unique())
    
    return x_train, y_train, train_ids, x_test, y_test, test_ids

### Create graph/node attributes
def create_blank_match_graph(one_match_player):
    return nx.Graph(match_api_id=one_match_player.match_api_id.iloc[0],
                    date=one_match_player.date.iloc[0])

def create_weighted_edges(one_match_player):
    df = one_match_player.copy()
    points = list(zip(df.X, df.Y))
    
    dist = cdist(points, points)
    dist = 1 - dist/dist.max()
    
    return list(zip(df.player_api_id.iloc[np.triu_indices(22)[0]],
                    df.player_api_id.iloc[np.triu_indices(22)[1]],
                    dist[np.triu_indices(22)]))

def create_node_attributes(one_match_player):
    node_drops = ['match_api_id',
                  'date']
    
    return one_match_player.drop(node_drops,
                        axis=1).set_index('player_api_id').to_dict('index')

### Row normalize adjacency matrices
def normalize(mx, sparse=False):
    """Row-normalize sparse matrix"""
    if sparse:
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
    else:
        row_sums = mx.sum(axis=1)
        mx = mx / row_sums
    return mx

### Create graph of one match
def create_graph(one_match_player, as_matrix, sparse):
    df = one_match_player.reset_index().copy()
    
    # Create blank graph
    g = create_blank_match_graph(df)
    
    # Add nodes from player ids
    g.add_nodes_from(df.player_api_id.to_list())
    
    # Add weighted edges
    weighted_edges = create_weighted_edges(df)
    
    g.add_weighted_edges_from(weighted_edges)
    
    node_attrs = create_node_attributes(df)
    
    nx.set_node_attributes(g, node_attrs)
    
    if as_matrix:
        g = nx.adjacency_matrix(g)
        
        g = normalize(g, sparse=sparse)
    
    return g

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

### Get block diagonal matrices from graph lists
def get_block_matrices(ls, as_tensor):
    if as_tensor:
        block = sparse_mx_to_torch_sparse_tensor(sp.block_diag(tuple(ls)))
    else:
        block = sp.block_diag(tuple(ls))
    return block

### Get graphs from train and test ids
def get_graphs(match_player, train_ids, test_ids, as_matrix, sparse):
    from tqdm import tqdm
    
    df = match_player.copy()
        
    df = df.set_index('match_api_id')
    
    ids = train_ids + test_ids
    
    g_ls = [create_graph(df.loc[i],
                         as_matrix=as_matrix,
                         sparse=sparse) for i in tqdm(ids, desc='Creating graphs')]
    
    return g_ls

### Scale features
def scale_data(x_train, x_test):
    scl = StandardScaler()
    scl.fit(x_train)
    
    x_train_scl = scl.transform(x_train)
    x_test_scl = scl.transform(x_test)
    
    return x_train_scl, x_test_scl

### Get block diagonal feature matrices
def get_feature_matrices(x_train_scl, x_test_scl, as_tensor):
    train_ls = [sp.csr_matrix(x_train_scl[i*22:(i*22 + 22)]) for i in range(int(x_train_scl.shape[0]/22))]
    test_ls = [sp.csr_matrix(x_test_scl[i*22:(i*22 + 22)]) for i in range(int(x_test_scl.shape[0]/22))]
    
    feat_ls = train_ls + test_ls
    
    return get_block_matrices(feat_ls, as_tensor=as_tensor)

def one_hot_to_target_tensor(y):
    return torch.argmax(torch.Tensor(y.to_numpy(dtype='float32')), dim=1)

### Get simple block diagonal pooling matrices
def get_pooling_matrices(train_len, test_len, as_tensor):
    pool = [sp.csr_matrix(np.ones((1,22))) for i in range(train_len + test_len)]
    
    return get_block_matrices(pool, as_tensor)

def main(as_tensor=True):
    from time import time
    from datetime import datetime
    
    t = time()
    print(f"Ingesting data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    match_player = ingest_data()
    print(f'Finished ingesting data in {np.round((time() - t)/60, 2)} minutes')
    
    t = time()
    print(f"Getting train/test splits: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    x_train, y_train, train_ids, x_test, y_test, test_ids = get_splits(match_player)
    
    x_train_scl, x_test_scl = scale_data(x_train, x_test)
    print(f'Finished ingesting data in {np.round((time() - t)/60, 2)} minutes')
    
    t = time()
    print(f"Generating graphs: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        g_ls = get_graphs(match_player,
                          train_ids,
                          test_ids,
                          as_matrix=True,
                          sparse=True)
    except Exception as e:
        print('Error creating graphs')
        raise
    else:
        print(f'Finished graph generation in {np.round((time() - t)/60, 2)} minutes')
    
    t = time()
    print(f"Creating block diagonal adjacency matrices: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        block_adj = get_block_matrices(g_ls, as_tensor=as_tensor)
    except Exception as e:
        print('Error creating adjacency matrices')
        raise
    else:
        print(f'Finished adjacency matrices in {np.round((time() - t)/60, 2)} minutes')
        
    t = time()
    print(f"Creating block diagonal feature matrices: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:        
        block_feat = get_feature_matrices(x_train_scl,
                                          x_test_scl,
                                          as_tensor=as_tensor)
    except Exception as e:
        print('Error creating feature matrices')
        raise
    else:
        print(f'Finished feature matrices in {np.round((time() - t)/60, 2)} minutes')
        
    t = time()
    print(f"Creating block diagonal pooling matrices: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        block_pool = get_pooling_matrices(len(y_train),
                                          len(y_test),
                                          as_tensor=as_tensor)
    except Exception as e:
        print('Error creating pooling matrices')
        raise
    else:
        print(f'Finished pooling matrices in {np.round((time() - t)/60, 2)} minutes')

    if as_tensor:
        y_train = one_hot_to_target_tensor(y_train)
        y_test = one_hot_to_target_tensor(y_test)
        
        y = torch.cat((y_train, y_test), 0)
    else:
        y = pd.concat([y_train, y_test], axis=0)
        
    train_len = len(y_train)
    
    total_len = train_len + len(y_test)
    
    return block_adj, block_feat, block_pool, y, train_len, total_len

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    
    return correct / len(labels)

if __name__ == '__main__':
    import pickle
    
    matrix_ls = main()   
    
    pkl = "data.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(matrix_ls, f)