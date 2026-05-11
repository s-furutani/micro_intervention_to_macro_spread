import os
import matplotlib.pyplot as plt
import csv
import json
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import time
import seaborn as sns
import scipy.stats as stats
from tqdm import tqdm
from collections import Counter

def Nikolov_susceptibility_graph():
    """Nikolovの実ネットワークデータを読み込み（高速化対応）"""
    pickle_file = './data/nikolov/nikolov_graph.pkl'
    
    # 加工済みpickleファイルが存在する場合は高速読み込み
    if os.path.exists(pickle_file):
        # print("加工済みpickleファイルを読み込み中...")
        start_time = time.time()
        with open(pickle_file, 'rb') as f:
            G = pickle.load(f)
        load_time = time.time() - start_time
        # print(f"pickle読み込み完了: {load_time:.2f}秒")
        # print(f"num_nodes: {G.number_of_nodes()}, num_edges: {G.number_of_edges()}")
        return G
    
    # 生データから構築
    print("生データからグラフを構築中...")
    start_time = time.time()
    
    path = './data/nikolov/'
    
    measures_file = path+"measures.tab"
    friends_file = path+"anonymized-friends.json"
    
    # ファイルの存在確認
    if not os.path.exists(measures_file):
        raise FileNotFoundError(f"Measures file not found: {measures_file}")
    if not os.path.exists(friends_file):
        raise FileNotFoundError(f"Friends file not found: {friends_file}")
    
    df = pd.read_csv(measures_file, sep="\t")
    G = nx.DiGraph() 
    
    print("ノードを追加中...")
    for _, row in df.iterrows():
        node_id = int(row["ID"])  # IDを整数に変換
        partisanship = row["Partisanship"]
        susceptibility = row["Misinformation"]
        G.add_node(node_id, partisanship=partisanship, suscep=susceptibility)
    
    print("エッジを追加中...")
    with open(friends_file, "r") as f:
        friend_data = json.load(f)
    
    for node, friends in friend_data.items():
        node = int(node)  # ノードIDを整数に変換
        friends = [int(f) for f in friends]  # 友達リストも整数に変換
    
        for friend in friends:
            G.add_edge(node, friend)  # 有向エッジを追加
    
    print("最大連結成分を抽出中...")
    valid_nodes = [node for node in G.nodes if 'suscep' in G.nodes[node]]
    subG = G.subgraph(valid_nodes).copy()
    weakly_connected_components = list(nx.weakly_connected_components(subG))
    lcc = max(weakly_connected_components, key=len)
    subG = subG.subgraph(lcc).copy()
    
    build_time = time.time() - start_time
    print(f"グラフ構築完了: {build_time:.2f}秒")
    print(f"Nikolov graph: n={subG.number_of_nodes()}, m={subG.number_of_edges()}")
    
    # pickleファイルとして保存
    print("pickleファイルとして保存中...")
    os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
    with open(pickle_file, 'wb') as f:
        pickle.dump(subG, f)
    print(f"pickleファイルを保存しました: {pickle_file}")
    
    return subG

def save_nikolov_graph_pickle():
    """Nikolovグラフをpickle形式で保存（手動実行用）"""
    print("Nikolovグラフをpickle形式で保存します...")
    G = Nikolov_susceptibility_graph()
    print("保存完了！")
    return G

def randomized_nikolov_graph():
    G = Nikolov_susceptibility_graph()
    G = nx.DiGraph(G)
    # susceptibilityをランダムにシャッフル（シードは42）
    suscep_values = [G.nodes[node]['suscep'] for node in G.nodes]
    np.random.seed(42)
    shuffled_suscep = np.random.permutation(suscep_values)
    for i, node in enumerate(G.nodes):
        G.nodes[node]['suscep'] = shuffled_suscep[i]
    return G


def uniform_nikolov_graph():
    """Nikolov と同一トポロジーで、susceptibility を全ノードで元グラフの平均値にそろえる。"""
    G = Nikolov_susceptibility_graph()
    G = nx.DiGraph(G)
    mean_s = float(np.mean([G.nodes[node]['suscep'] for node in G.nodes]))
    for node in G.nodes:
        G.nodes[node]['suscep'] = mean_s
    G.graph['graph_name'] = 'uniform_nikolov'
    return G

# G = Nikolov_susceptibility_graph()
# print(len(G.nodes()), len(G.edges()))