import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
import dgl
from dgl.data import DGLDataset
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import random
from datetime import datetime

# 设置 matplotlib 支持中文显示，解决中文乱码问题
plt.rcParams["font.family"] = ["SimHei"]  
plt.rcParams["axes.unicode_minus"] = False  

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 下载VADER情感分析模型（首次运行需要）
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

class CombinedGraphConstructor:
    def __init__(self, enron_data_dir, facebook_data_dir, output_dir='combined_graph_data'):
        """初始化融合图构建器
        
        Args:
            enron_data_dir (str): Enron预处理数据目录
            facebook_data_dir (str): Facebook预处理数据目录
            output_dir (str): 图数据保存目录
        """
        self.enron_data_dir = enron_data_dir
        self.facebook_data_dir = facebook_data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载两个数据集的预处理数据
        self.enron_users = pd.read_csv(os.path.join(enron_data_dir, 'users_with_features.csv'))
        self.enron_edges = pd.read_csv(os.path.join(enron_data_dir, 'edges_with_features.csv'))
        self.enron_email_tfidf = pd.read_csv(os.path.join(enron_data_dir, 'email_tfidf_features.csv'))
        
        self.facebook_users = pd.read_csv(os.path.join(facebook_data_dir, 'users_with_features.csv'))
        self.facebook_edges = pd.read_csv(os.path.join(facebook_data_dir, 'edges_with_features.csv'))
        
        # 图数据存储
        self.nx_graphs = {}  # NetworkX图：{'friend': 无向图, 'cooperation': 无向图, 'follow': 有向图}
        self.pyg_data = None  # PyTorch Geometric数据对象
        self.dgl_graphs = {}  # DGL图
        
        # 合并用户数据并创建统一ID映射（确保无冲突）
        self._merge_users()
        # 合并边数据
        self._merge_edges()
        
        print(f"数据集合并完成: 总用户数={self.num_nodes}, 总边数={len(self.combined_edges)}")
        
    def _merge_users(self):
        """合并Facebook和Enron用户数据，确保ID唯一"""
        # 为Facebook用户添加数据集标识
        self.facebook_users['dataset'] = 'facebook'
        # 为Enron用户添加数据集标识
        self.enron_users['dataset'] = 'enron'
        
        # 确保用户ID不冲突（如果有冲突则重新映射）
        fb_ids = set(self.facebook_users['user_id'].unique())
        enron_ids = set(self.enron_users['user_id'].unique())
        conflicting_ids = fb_ids & enron_ids
        
        if conflicting_ids:
            print(f"发现{len(conflicting_ids)}个冲突用户ID，正在重新映射...")
            # 重新映射Enron用户ID
            max_fb_id = max(fb_ids) if fb_ids else 0
            enron_id_map = {uid: max_fb_id + 1 + i for i, uid in enumerate(enron_ids)}
            self.enron_users['user_id'] = self.enron_users['user_id'].map(enron_id_map)
        
        # 统一用户特征列（取并集）
        common_cols = list(set(self.facebook_users.columns) & set(self.enron_users.columns))
        fb_only_cols = list(set(self.facebook_users.columns) - set(common_cols))
        enron_only_cols = list(set(self.enron_users.columns) - set(common_cols))
        
        # 对齐特征列
        self.facebook_users = self.facebook_users.reindex(columns=common_cols + fb_only_cols).fillna(0)
        self.enron_users = self.enron_users.reindex(columns=common_cols + enron_only_cols).fillna(0)
        
        # 合并用户数据
        self.combined_users = pd.concat([self.facebook_users, self.enron_users], ignore_index=True)
        
        # 创建连续ID映射
        self.user_id_map = {user_id: i for i, user_id in enumerate(self.combined_users['user_id'].unique())}
        self.num_nodes = len(self.user_id_map)
        
    def _merge_edges(self):
        """合并Facebook和Enron边数据"""
        # 为Facebook边添加类型和数据集标识
        self.facebook_edges['edge_type'] = 'friend'  # Facebook主要是社交好友关系
        self.facebook_edges['dataset'] = 'facebook'
        
        # 为Enron边添加类型和数据集标识
        self.enron_edges['edge_type'] = 'cooperation'  # Enron主要是企业合作关系
        self.enron_edges['dataset'] = 'enron'
        
        # 确保边的源和目标ID与合并后的用户ID一致
        fb_id_conflict = set(self.facebook_edges['source']) & set(self.enron_users['user_id'])
        if fb_id_conflict:
            # 处理可能的边ID冲突
            max_fb_id = max(self.facebook_users['user_id']) if not self.facebook_users.empty else 0
            enron_id_map = {uid: max_fb_id + 1 + i for i, uid in enumerate(self.enron_users['user_id'].unique())}
            self.enron_edges['source'] = self.enron_edges['source'].map(lambda x: enron_id_map.get(x, x))
            self.enron_edges['target'] = self.enron_edges['target'].map(lambda x: enron_id_map.get(x, x))
        
        # 合并边数据
        common_edge_cols = list(set(self.facebook_edges.columns) & set(self.enron_edges.columns))
        combined_edges_temp = pd.concat(
            [self.facebook_edges[common_edge_cols], self.enron_edges[common_edge_cols]], 
            ignore_index=True
        )
        
        # 关键修复：过滤掉用户数据中不存在的节点ID
        valid_user_ids = set(self.combined_users['user_id'].unique())

        # 保留源和目标都有效的边
        combined_edges_temp = combined_edges_temp[
            combined_edges_temp['source'].isin(valid_user_ids) & 
            combined_edges_temp['target'].isin(valid_user_ids)
        ]
        
        self.combined_edges = combined_edges_temp
        
        # 生成跨数据集的关注关系（有向边）
        self._generate_cross_edges()
        
    def _generate_cross_edges(self):
        """生成跨数据集的关注关系，增加两个网络的连接"""
        fb_users = self.combined_users[self.combined_users['dataset'] == 'facebook']['user_id'].unique()
        enron_users = self.combined_users[self.combined_users['dataset'] == 'enron']['user_id'].unique()
        
        if len(fb_users) > 0 and len(enron_users) > 0:
            cross_count = int(0.05 * len(self.combined_edges))  # 生成总边数5%的跨数据集边
            cross_edges = []
            
            # Facebook用户关注Enron用户
            for _ in range(cross_count // 2):
                source = random.choice(fb_users)
                target = random.choice(enron_users)
                if source != target:
                    cross_edges.append({
                        'source': source,
                        'target': target,
                        'interaction_freq': random.randint(1, 30),
                        'last_interaction': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'edge_type': 'follow',
                        'dataset': 'cross'
                    })
            
            # Enron用户关注Facebook用户
            for _ in range(cross_count - cross_count // 2):
                source = random.choice(enron_users)
                target = random.choice(fb_users)
                if source != target:
                    cross_edges.append({
                        'source': source,
                        'target': target,
                        'interaction_freq': random.randint(1, 30),
                        'last_interaction': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'edge_type': 'follow',
                        'dataset': 'cross'
                    })
            
            self.combined_edges = pd.concat([self.combined_edges, pd.DataFrame(cross_edges)], ignore_index=True)
            print(f"生成跨数据集关注关系: {cross_count}条边")
    
    def define_graph_structure(self):
        """定义融合图结构：节点、边类型及特征"""
        print("开始定义融合图结构...")
        
        # 1. 节点特征处理（合并结构化特征与TF-IDF文本特征）
        # 处理Enron用户的TF-IDF特征
        if 'email' in self.enron_users.columns and not self.enron_email_tfidf.empty:
            email_user_map = pd.read_csv(os.path.join(self.enron_data_dir, 'cleaned_emails.csv'))[['email_id', 'from_addr']]
            user_email_df = pd.merge(
                self.enron_users[['user_id', 'email']],
                email_user_map,
                left_on='email',
                right_on='from_addr',
                how='left'
            )
            
            # 聚合每个用户的TF-IDF特征
            user_tfidf = pd.merge(
                user_email_df[['user_id', 'email_id']],
                self.enron_email_tfidf,
                on='email_id',
                how='left'
            ).groupby('user_id').mean().reset_index()
            
            # 合并到总用户数据
            self.combined_users = pd.merge(
                self.combined_users,
                user_tfidf,
                on='user_id',
                how='left'
            ).fillna(0)
        
        # 2. 计算边权重：0.6×交互频率 + 0.4×情感倾向
        # 为Facebook边生成情感倾向（模拟）
        fb_mask = self.combined_edges['dataset'] == 'facebook'
        self.combined_edges.loc[fb_mask, 'sentiment'] = [random.uniform(-1, 1) for _ in range(sum(fb_mask))]
        
        # 为Enron边生成情感倾向（可基于邮件内容，这里简化处理）
        enron_mask = self.combined_edges['dataset'] == 'enron'
        self.combined_edges.loc[enron_mask, 'sentiment'] = [random.uniform(-1, 1) for _ in range(sum(enron_mask))]
        
        # 为跨数据集边生成情感倾向
        cross_mask = self.combined_edges['dataset'] == 'cross'
        self.combined_edges.loc[cross_mask, 'sentiment'] = [random.uniform(-1, 1) for _ in range(sum(cross_mask))]
        
        # 归一化交互频率到[0,1]
        freq_min, freq_max = self.combined_edges['interaction_freq'].min(), self.combined_edges['interaction_freq'].max()
        self.combined_edges['freq_norm'] = (self.combined_edges['interaction_freq'] - freq_min) / (freq_max - freq_min + 1e-8)
        
        # 计算边权重
        self.combined_edges['weight'] = 0.6 * self.combined_edges['freq_norm'] + 0.4 * ((self.combined_edges['sentiment'] + 1) / 2)
        
        # 3. 计算交互时长（天）
        self.combined_edges['last_interaction'] = pd.to_datetime(self.combined_edges['last_interaction'])
        earliest_date = self.combined_edges['last_interaction'].min()
        self.combined_edges['interaction_duration'] = (self.combined_edges['last_interaction'] - earliest_date).dt.days
        
        print(f"融合图结构定义完成: 节点数={self.num_nodes}, 边数={len(self.combined_edges)}")
        print(f"边类型分布: {self.combined_edges['edge_type'].value_counts().to_dict()}")
        print(f"数据集来源分布: {self.combined_edges['dataset'].value_counts().to_dict()}")
        return self
    
    def build_networkx_graphs(self, visualize=True):
        """使用NetworkX构建融合图并可视化"""
        print("开始构建NetworkX融合图...")
        
        # 1. 构建不同类型的图
        # 无向图：好友（主要来自Facebook）、合作（主要来自Enron）
        for edge_type in ['friend', 'cooperation']:
            sub_edges = self.combined_edges[self.combined_edges['edge_type'] == edge_type]
            G = nx.Graph()
            
            # 添加节点及特征
            for _, row in self.combined_users.iterrows():
                node_id = self.user_id_map[row['user_id']]
                # 提取节点特征（排除非数值列）
                features = {k: v for k, v in row.to_dict().items() 
                           if k not in ['user_id', 'email', 'username', 'dataset'] and not pd.isna(v)}
                G.add_node(node_id, **features, dataset=row['dataset'])
            
            # 添加边及特征
            for _, row in sub_edges.iterrows():
                u = self.user_id_map[row['source']]
                v = self.user_id_map[row['target']]
                edge_features = {
                    'weight': row['weight'],
                    'interaction_freq': row['interaction_freq'],
                    'interaction_duration': row['interaction_duration'],
                    'sentiment': row['sentiment'],
                    'dataset': row['dataset']
                }
                G.add_edge(u, v,** edge_features)
            
            self.nx_graphs[edge_type] = G
            print(f"构建{edge_type}无向图: 节点数={G.number_of_nodes()}, 边数={G.number_of_edges()}")
        
        # 2. 构建有向图：关注（包含跨数据集关系）
        edge_type = 'follow'
        sub_edges = self.combined_edges[self.combined_edges['edge_type'] == edge_type]
        G = nx.DiGraph()
        
        # 添加节点
        for _, row in self.combined_users.iterrows():
            node_id = self.user_id_map[row['user_id']]
            features = {k: v for k, v in row.to_dict().items() 
                       if k not in ['user_id', 'email', 'username', 'dataset'] and not pd.isna(v)}
            G.add_node(node_id, **features, dataset=row['dataset'])
        
        # 添加有向边
        for _, row in sub_edges.iterrows():
            u = self.user_id_map[row['source']]
            v = self.user_id_map[row['target']]
            edge_features = {
                'weight': row['weight'],
                'interaction_freq': row['interaction_freq'],
                'interaction_duration': row['interaction_duration'],
                'dataset': row['dataset']
            }
            G.add_edge(u, v,** edge_features)
        
        self.nx_graphs[edge_type] = G
        print(f"构建{edge_type}有向图: 节点数={G.number_of_nodes()}, 边数={G.number_of_edges()}")
        
        # 3. 可视化小规模图（区分不同数据集节点）
        if visualize:
            for name, G in self.nx_graphs.items():
                # 只可视化前100个节点的子图
                if G.number_of_nodes() > 100:
                    sample_nodes = random.sample(list(G.nodes()), 100)
                    subG = G.subgraph(sample_nodes)
                else:
                    subG = G
                
                plt.figure(figsize=(12, 8))
                pos = nx.spring_layout(subG, seed=42)
                
                # 按数据集区分节点颜色
                node_colors = ['lightgreen' if G.nodes[n]['dataset'] == 'facebook' else 'lightblue' 
                              for n in subG.nodes()]
                
                # 绘制节点
                nx.draw_networkx_nodes(subG, pos, node_size=100, node_color=node_colors, alpha=0.8)
                
                # 绘制边（按权重调整宽度）
                nx.draw_networkx_edges(subG, pos, 
                                      width=[d['weight']*2 for (u, v, d) in subG.edges(data=True)],
                                      alpha=0.5)
                
                # 添加图例
                plt.scatter([], [], c='lightgreen', label='Facebook用户')
                plt.scatter([], [], c='lightblue', label='Enron用户')
                plt.legend()
                
                plt.title(f'NetworkX {name} Graph (Combined Dataset)')
                plt.savefig(os.path.join(self.output_dir, f'nx_combined_{name}_graph.png'))
                plt.close()
                print(f"融合{name}图可视化完成，已保存到文件")
        
        # 保存NetworkX图
        with open(os.path.join(self.output_dir, 'networkx_combined_graphs.pkl'), 'wb') as f:
            pickle.dump(self.nx_graphs, f)
        
        return self
    
    def build_pyg_graph(self):
        """使用PyTorch Geometric构建适配GNN的融合图Data对象"""
        print("开始构建PyTorch Geometric融合图...")
        
        # 1. 预处理节点特征，处理字符串类型
        processed_users = self.combined_users.copy()
        
        # 识别并处理字符串类型的列（正确使用dtypes）
        string_cols = [col for col in processed_users.columns 
                      if processed_users[col].dtypes == 'object'  # 这里修正为dtypes
                      and col not in ['user_id', 'email', 'username', 'dataset']]
        
        # 对字符串类型列进行One-Hot编码
        for col in string_cols:
            # 创建独热编码
            one_hot = pd.get_dummies(processed_users[col], prefix=col, drop_first=True)
            # 合并到用户数据
            processed_users = pd.concat([processed_users, one_hot], axis=1)
            # 删除原始字符串列
            processed_users = processed_users.drop(col, axis=1)
            print(f"对字符串特征 '{col}' 进行One-Hot编码，生成 {len(one_hot.columns)} 个特征")
        
        # 2. 准备节点特征
        # 提取数值特征列
        feature_cols = [col for col in processed_users.columns 
                       if col not in ['user_id', 'email', 'username', 'dataset']]
        
        # 检查并处理可能残留的非数值特征
        for col in feature_cols:
            # 正确检查列的 dtype
            if processed_users[col].dtypes == 'object':
                print(f"警告：特征 '{col}' 仍为字符串类型，将其转换为类别编码")
                # 先填充缺失值，避免编码错误
                processed_users[col] = processed_users[col].fillna('unknown')
                # 转换为类别编码
                processed_users[col] = processed_users[col].astype('category').cat.codes
        
        # 确保所有特征都是数值型，并处理可能的缺失值
        processed_users[feature_cols] = processed_users[feature_cols].fillna(0)
        
        # 转换为数值矩阵
        x = processed_users.sort_values('user_id')[feature_cols].values.astype(np.float32)
        x = torch.tensor(x, dtype=torch.float)
        
        # 添加数据集标识作为节点特征
        dataset_id = processed_users.sort_values('user_id')['dataset'].apply(
            lambda x: 0 if x == 'facebook' else 1
        ).values.reshape(-1, 1)
        x = torch.cat([x, torch.tensor(dataset_id, dtype=torch.float)], dim=1)
        
        # 3. 准备边信息（区分边类型）
        edge_list = []
        edge_attrs = []
        edge_types = []  # 0: friend, 1: cooperation, 2: follow
        edge_sources = []  # 0: facebook, 1: enron, 2: cross
        
        # 好友边（无向）
        friend_edges = self.combined_edges[self.combined_edges['edge_type'] == 'friend']
        for _, row in friend_edges.iterrows():
            u = self.user_id_map[row['source']]
            v = self.user_id_map[row['target']]
            edge_list.append([u, v])
            edge_list.append([v, u])  # 无向边双向添加
            attr = [row['weight'], row['interaction_freq'], row['interaction_duration']]
            edge_attrs.extend([attr, attr])
            edge_types.extend([0, 0])
            src_code = 0 if row['dataset'] == 'facebook' else 1 if row['dataset'] == 'enron' else 2
            edge_sources.extend([src_code, src_code])
        
        # 合作边（无向）
        coop_edges = self.combined_edges[self.combined_edges['edge_type'] == 'cooperation']
        for _, row in coop_edges.iterrows():
            u = self.user_id_map[row['source']]
            v = self.user_id_map[row['target']]
            edge_list.append([u, v])
            edge_list.append([v, u])  # 无向边双向添加
            attr = [row['weight'], row['interaction_freq'], row['interaction_duration']]
            edge_attrs.extend([attr, attr])
            edge_types.extend([1, 1])
            src_code = 0 if row['dataset'] == 'facebook' else 1 if row['dataset'] == 'enron' else 2
            edge_sources.extend([src_code, src_code])
        
        # 关注边（有向）
        follow_edges = self.combined_edges[self.combined_edges['edge_type'] == 'follow']
        for _, row in follow_edges.iterrows():
            u = self.user_id_map[row['source']]
            v = self.user_id_map[row['target']]
            edge_list.append([u, v])
            attr = [row['weight'], row['interaction_freq'], row['interaction_duration']]
            edge_attrs.append(attr)
            edge_types.append(2)
            src_code = 0 if row['dataset'] == 'facebook' else 1 if row['dataset'] == 'enron' else 2
            edge_sources.append(src_code)
        
        # 转换为张量
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        edge_source = torch.tensor(edge_sources, dtype=torch.long)
        
        # 4. 创建PyG Data对象
        self.pyg_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_type=edge_type,
            edge_source=edge_source,
            num_nodes=self.num_nodes
        )
        
        # 保存PyG数据
        torch.save(self.pyg_data, os.path.join(self.output_dir, 'pyg_combined_graph.pt'))
        print(f"PyTorch Geometric融合图构建完成: 节点特征={x.shape}, 边数={edge_index.shape[1]}")
        return self
    
    def build_pyg_graph(self):
        """使用PyTorch Geometric构建适配GNN的融合图Data对象"""
        print("开始构建PyTorch Geometric融合图...")
        
        # 1. 预处理节点特征，处理字符串类型
        processed_users = self.combined_users.copy()
        
        # 获取所有列的数据类型（返回一个Series）
        col_dtypes = processed_users.dtypes
        
        # 识别并处理字符串类型的列（使用更严谨的方式）
        string_cols = []
        for col in processed_users.columns:
            if col in ['user_id', 'email', 'username', 'dataset']:
                continue
            # 确保获取的是单个 dtype 值
            dtype = col_dtypes[col]
            if isinstance(dtype, object) or str(dtype) == 'object':
                string_cols.append(col)
        
        # 对字符串类型列进行One-Hot编码
        for col in string_cols:
            # 创建独热编码
            one_hot = pd.get_dummies(processed_users[col], prefix=col, drop_first=True)
            # 合并到用户数据
            processed_users = pd.concat([processed_users, one_hot], axis=1)
            # 删除原始字符串列
            processed_users = processed_users.drop(col, axis=1)
            print(f"对字符串特征 '{col}' 进行One-Hot编码，生成 {len(one_hot.columns)} 个特征")
        
        # 2. 准备节点特征
        # 提取数值特征列
        feature_cols = [col for col in processed_users.columns 
                       if col not in ['user_id', 'email', 'username', 'dataset']]
        
        # 再次检查并处理可能残留的非数值特征
        col_dtypes_updated = processed_users.dtypes
        for col in feature_cols:
            dtype = col_dtypes_updated[col]
            # 安全检查数据类型
            if isinstance(dtype, object) or str(dtype) == 'object':
                print(f"警告：特征 '{col}' 仍为字符串类型，将其转换为类别编码")
                # 先填充缺失值，避免编码错误
                processed_users[col] = processed_users[col].fillna('unknown')
                # 转换为类别编码
                processed_users[col] = processed_users[col].astype('category').cat.codes
        
        # 确保所有特征都是数值型，并处理可能的缺失值
        processed_users[feature_cols] = processed_users[feature_cols].fillna(0)
        
        # 转换为数值矩阵
        try:
            x = processed_users.sort_values('user_id')[feature_cols].values.astype(np.float32)
            x = torch.tensor(x, dtype=torch.float)
        except ValueError as e:
            print(f"转换特征为张量时出错: {e}")
            # 打印有问题的列以帮助调试
            for col in feature_cols:
                try:
                    processed_users[col].astype(np.float32)
                except ValueError:
                    print(f"无法转换列 '{col}' 为浮点型")
            raise  # 重新抛出异常
        
        # 添加数据集标识作为节点特征
        dataset_id = processed_users.sort_values('user_id')['dataset'].apply(
            lambda x: 0 if x == 'facebook' else 1
        ).values.reshape(-1, 1)
        x = torch.cat([x, torch.tensor(dataset_id, dtype=torch.float)], dim=1)
        
        # 3. 准备边信息（区分边类型）
        edge_list = []
        edge_attrs = []
        edge_types = []  # 0: friend, 1: cooperation, 2: follow
        edge_sources = []  # 0: facebook, 1: enron, 2: cross
        
        # 好友边（无向）
        friend_edges = self.combined_edges[self.combined_edges['edge_type'] == 'friend']
        for _, row in friend_edges.iterrows():
            u = self.user_id_map[row['source']]
            v = self.user_id_map[row['target']]
            edge_list.append([u, v])
            edge_list.append([v, u])  # 无向边双向添加
            attr = [row['weight'], row['interaction_freq'], row['interaction_duration']]
            edge_attrs.extend([attr, attr])
            edge_types.extend([0, 0])
            src_code = 0 if row['dataset'] == 'facebook' else 1 if row['dataset'] == 'enron' else 2
            edge_sources.extend([src_code, src_code])
        
        # 合作边（无向）
        coop_edges = self.combined_edges[self.combined_edges['edge_type'] == 'cooperation']
        for _, row in coop_edges.iterrows():
            u = self.user_id_map[row['source']]
            v = self.user_id_map[row['target']]
            edge_list.append([u, v])
            edge_list.append([v, u])  # 无向边双向添加
            attr = [row['weight'], row['interaction_freq'], row['interaction_duration']]
            edge_attrs.extend([attr, attr])
            edge_types.extend([1, 1])
            src_code = 0 if row['dataset'] == 'facebook' else 1 if row['dataset'] == 'enron' else 2
            edge_sources.extend([src_code, src_code])
        
        # 关注边（有向）
        follow_edges = self.combined_edges[self.combined_edges['edge_type'] == 'follow']
        for _, row in follow_edges.iterrows():
            u = self.user_id_map[row['source']]
            v = self.user_id_map[row['target']]
            edge_list.append([u, v])
            attr = [row['weight'], row['interaction_freq'], row['interaction_duration']]
            edge_attrs.append(attr)
            edge_types.append(2)
            src_code = 0 if row['dataset'] == 'facebook' else 1 if row['dataset'] == 'enron' else 2
            edge_sources.append(src_code)
        
        # 转换为张量
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        edge_source = torch.tensor(edge_sources, dtype=torch.long)
        
        # 4. 创建PyG Data对象
        self.pyg_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_type=edge_type,
            edge_source=edge_source,
            num_nodes=self.num_nodes
        )
        
        # 保存PyG数据
        torch.save(self.pyg_data, os.path.join(self.output_dir, 'pyg_combined_graph.pt'))
        print(f"PyTorch Geometric融合图构建完成: 节点特征={x.shape}, 边数={edge_index.shape[1]}")
        return self
    
    def build_dgl_graphs(self):
        """使用DGL构建融合图，用于动态更新"""
        print("开始构建DGL融合图...")
        
        # 预处理节点特征，确保与PyG图使用相同的特征处理
        processed_users = self.combined_users.copy()
        
        # 获取所有列的数据类型
        col_dtypes = processed_users.dtypes
        
        # 识别并处理字符串类型的列
        string_cols = []
        for col in processed_users.columns:
            if col in ['user_id', 'email', 'username', 'dataset']:
                continue
            dtype = col_dtypes[col]
            if isinstance(dtype, object) or str(dtype) == 'object':
                string_cols.append(col)
        
        # 对字符串类型列进行One-Hot编码
        for col in string_cols:
            one_hot = pd.get_dummies(processed_users[col], prefix=col, drop_first=True)
            processed_users = pd.concat([processed_users, one_hot], axis=1)
            processed_users = processed_users.drop(col, axis=1)
        
        # 提取数值特征列
        feature_cols = [col for col in processed_users.columns 
                       if col not in ['user_id', 'email', 'username', 'dataset']]
        
        # 处理可能残留的非数值特征
        col_dtypes_updated = processed_users.dtypes
        for col in feature_cols:
            dtype = col_dtypes_updated[col]
            if isinstance(dtype, object) or str(dtype) == 'object':
                processed_users[col] = processed_users[col].fillna('unknown')
                processed_users[col] = processed_users[col].astype('category').cat.codes
        
        # 填充缺失值
        processed_users[feature_cols] = processed_users[feature_cols].fillna(0)
        
        # 为每种边类型创建DGL图
        for edge_type in ['friend', 'cooperation', 'follow']:
            sub_edges = self.combined_edges[self.combined_edges['edge_type'] == edge_type]
            if sub_edges.empty:
                continue
                
            # 转换为连续ID
            u = [self.user_id_map[uid] for uid in sub_edges['source']]
            v = [self.user_id_map[uid] for uid in sub_edges['target']]
            
            # 创建图（兼容非常旧版本的DGL）
            if edge_type in ['friend', 'cooperation']:
                # 无向图：通过添加双向边实现
                # 将(u, v)和(v, u)都添加为边
                u_undirected = u + v
                v_undirected = v + u
                g = dgl.DGLGraph((u_undirected, v_undirected))
            else:
                # 有向图
                g = dgl.DGLGraph((u, v))
            
            # 手动设置节点数量（旧版本DGL兼容方式）
            if hasattr(g, 'num_nodes') and g.num_nodes() < self.num_nodes:
                g.add_nodes(self.num_nodes - g.num_nodes())
            
            # 添加边特征
            # 对于无向图，需要复制特征以匹配双向边
            if edge_type in ['friend', 'cooperation']:
                # 复制特征以匹配双向边
                weights = sub_edges['weight'].values
                freq = sub_edges['interaction_freq'].values
                duration = sub_edges['interaction_duration'].values
                sentiment = sub_edges['sentiment'].values
                sources = [0 if d == 'facebook' else 1 if d == 'enron' else 2 
                          for d in sub_edges['dataset']]
                
                # 双向边特征
                g.edata['weight'] = torch.tensor(np.concatenate([weights, weights]), dtype=torch.float)
                g.edata['interaction_freq'] = torch.tensor(np.concatenate([freq, freq]), dtype=torch.int)
                g.edata['interaction_duration'] = torch.tensor(np.concatenate([duration, duration]), dtype=torch.int)
                g.edata['sentiment'] = torch.tensor(np.concatenate([sentiment, sentiment]), dtype=torch.float)
                g.edata['source'] = torch.tensor(sources + sources, dtype=torch.int)
            else:
                # 有向图直接使用特征
                g.edata['weight'] = torch.tensor(sub_edges['weight'].values, dtype=torch.float)
                g.edata['interaction_freq'] = torch.tensor(sub_edges['interaction_freq'].values, dtype=torch.int)
                g.edata['interaction_duration'] = torch.tensor(sub_edges['interaction_duration'].values, dtype=torch.int)
                g.edata['sentiment'] = torch.tensor(sub_edges['sentiment'].values, dtype=torch.float)
                g.edata['source'] = torch.tensor(
                    [0 if d == 'facebook' else 1 if d == 'enron' else 2 
                     for d in sub_edges['dataset']], 
                    dtype=torch.int
                )
            
            # 准备节点特征
            node_feats = processed_users.sort_values('user_id')[feature_cols].values.astype(np.float32)
            
            # 添加数据集标识作为节点特征
            dataset_id = processed_users.sort_values('user_id')['dataset'].apply(
                lambda x: 0 if x == 'facebook' else 1
            ).values.reshape(-1, 1)
            node_feats = np.hstack([node_feats, dataset_id])
            
            g.ndata['feat'] = torch.tensor(node_feats, dtype=torch.float)
            g.ndata['dataset'] = torch.tensor(
                [0 if d == 'facebook' else 1 for d in processed_users.sort_values('user_id')['dataset']],
                dtype=torch.int
            )
            
            self.dgl_graphs[edge_type] = g
            print(f"DGL {edge_type}融合图构建完成: 节点数={g.num_nodes()}, 边数={g.num_edges()}")
        
        # 保存DGL图
        for name, g in self.dgl_graphs.items():
            dgl.save_graphs(os.path.join(self.output_dir, f'dgl_combined_{name}_graph.bin'), [g])
        
        return self
    
    

    def dynamic_update_demo(self, edge_type='friend', num_updates=10):
        """演示DGL动态更新功能，支持跨数据集更新"""
        if edge_type not in self.dgl_graphs:
            print(f"没有找到{edge_type}类型的DGL图")
            return self
        
        print(f"开始演示{edge_type}融合图的动态更新...")
        g = self.dgl_graphs[edge_type]
        
        # 记录初始状态
        initial_freq = g.edata['interaction_freq'].sum().item()
        
        # 随机选择边进行更新
        for i in range(num_updates):
            # 随机选择一条边
            eid = random.randint(0, g.num_edges() - 1)
            
            # 更新交互频率（+1）
            g.edata['interaction_freq'][eid] += 1
            
            # 重新计算权重
            freq_min = g.edata['interaction_freq'].min().item()
            freq_max = g.edata['interaction_freq'].max().item()
            freq_norm = (g.edata['interaction_freq'][eid].item() - freq_min) / (freq_max - freq_min + 1e-8)
            
            # 更新权重（保持情感不变）
            g.edata['weight'][eid] = 0.6 * freq_norm + 0.4 * ((g.edata['sentiment'][eid].item() + 1) / 2)
            
            if (i + 1) % 5 == 0:
                src_type = "Facebook" if g.edata['source'][eid].item() == 0 else "Enron" if g.edata['source'][eid].item() == 1 else "Cross"
                print(f"完成{i+1}次更新，{src_type}边{eid}的交互频率变为{g.edata['interaction_freq'][eid].item()}")
        
        # 计算总变化
        final_freq = g.edata['interaction_freq'].sum().item()
        print(f"动态更新完成: 交互频率总计增加 {final_freq - initial_freq}")
        
        # 保存更新后的图
        dgl.save_graphs(os.path.join(self.output_dir, f'dgl_combined_{edge_type}_graph_updated.bin'), [g])
        self.dgl_graphs[edge_type] = g
        
        return self


if __name__ == "__main__":
    # 预处理数据目录
    ENRON_DATA_DIR = "D:/cc/graph1.0/enron_processed_data"
    FACEBOOK_DATA_DIR = "D:/cc/graph1.0/facebook_processed_data"  # Facebook预处理数据目录
    
    # 创建融合图构建器并执行完整流程
    (CombinedGraphConstructor(ENRON_DATA_DIR, FACEBOOK_DATA_DIR)
     .define_graph_structure()       # 定义融合图结构
     .build_networkx_graphs()        # 构建NetworkX融合图
     .build_pyg_graph()              # 构建PyTorch Geometric融合图
     .build_dgl_graphs()             # 构建DGL融合图
     .dynamic_update_demo())         # 演示动态更新
    
    print("\n所有融合图构建步骤完成！图数据已保存到 'combined_graph_data' 目录")
