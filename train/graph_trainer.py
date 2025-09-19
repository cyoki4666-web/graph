import os
import sys
import torch
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

# 设置 matplotlib 支持中文显示，解决中文乱码问题
plt.rcParams["font.family"] = ["SimHei"]  
plt.rcParams["axes.unicode_minus"] = False  

# -----------------------------------------------------------------------------
# 1. 基础配置与工具函数
# -----------------------------------------------------------------------------
def set_seed(seed=42):
    """设置随机种子确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def get_project_root():
    """获取项目根目录（适配train/build目录结构）"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # train文件夹的上级目录即根目录
    return project_root

# 动态添加项目根目录到Python路径（解决跨文件夹导入）
project_root = get_project_root()
sys.path.append(project_root)

# 导入图构造模块（需确保build目录下有combined_graph_construction.py）
from build.combined_graph_construction import CombinedGraphConstructor

# -----------------------------------------------------------------------------
# 2. 数据集标注与划分
# -----------------------------------------------------------------------------
def create_labeled_dataset(combined_edges, combined_users, user_id_map):
    """
    创建带标签的边数据集（解决索引越界）
    - 正样本：交互频率≥8次的关系对（1）
    - 负样本：随机生成的非关系对（0）
    - 严格过滤无效用户ID和索引
    """
    labeled_data = []
    all_user_ids = set(combined_users['user_id'].unique())
    existing_edges = set()
    num_nodes = len(user_id_map)  # 节点总数=映射表大小（确保索引范围[0, num_nodes-1]）

    # 1. 处理正样本（已有关系）
    for _, row in combined_edges.iterrows():
        u, v = row['source'], row['target']
        
        # 过滤无效用户ID（不在映射表中的用户）
        if u not in user_id_map or v not in user_id_map:
            continue
        
        # 转换为连续索引并验证范围
        u_idx = user_id_map[u]
        v_idx = user_id_map[v]
        if u_idx >= num_nodes or v_idx >= num_nodes:
            continue
        
        # 避免重复边（按u<v排序）
        if u > v:
            u, v = v, u
        if (u, v) not in existing_edges:
            existing_edges.add((u, v))
            # 按规则标注标签
            label = 1 if row['interaction_freq'] >= 8 else 0
            labeled_data.append({
                'source': u,
                'target': v,
                'edge_type': row['edge_type'],
                'interaction_freq': row['interaction_freq'],
                'label': label,
                'dataset': row['dataset'],
                'source_id': u_idx,
                'target_id': v_idx
            })

    # 2. 生成负样本（与正样本数量平衡，仅用有效用户）
    valid_user_list = list(all_user_ids)
    neg_count = len(labeled_data)  # 负样本数量=正样本数量（类别平衡）
    neg_samples = []
    
    while len(neg_samples) < neg_count and len(existing_edges) < len(valid_user_list)**2:
        u = random.choice(valid_user_list)
        v = random.choice(valid_user_list)
        
        # 过滤无效用户和自环边
        if u == v or u not in user_id_map or v not in user_id_map:
            continue
        
        # 验证索引范围
        u_idx = user_id_map[u]
        v_idx = user_id_map[v]
        if u_idx >= num_nodes or v_idx >= num_nodes:
            continue
        
        # 避免与已有边重复
        if u > v:
            u, v = v, u
        if (u, v) not in existing_edges:
            existing_edges.add((u, v))
            neg_samples.append({
                'source': u,
                'target': v,
                'edge_type': 'none',
                'interaction_freq': random.randint(0, 7),  # 负样本交互频率<8
                'label': 0,
                'dataset': random.choice(['facebook', 'enron']),
                'source_id': u_idx,
                'target_id': v_idx
            })

    # 合并正负样本并返回
    labeled_df = pd.DataFrame(labeled_data + neg_samples)
    print(f"[数据集标注] 总样本数：{len(labeled_df)} | 正样本：{sum(labeled_df['label'])} | 负样本：{len(labeled_df)-sum(labeled_df['label'])}")
    return labeled_df

def split_dataset(labeled_df):
    """按7:1:2划分训练/验证/测试集（分层抽样保证类别平衡）"""
    # 先分训练+验证集 与 测试集（20%）
    train_val_df, test_df = train_test_split(
        labeled_df,
        test_size=0.2,
        random_state=42,
        stratify=labeled_df['label']  # 按标签分层
    )
    
    # 再分训练集（70%）与 验证集（10%）
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.125,  # 0.125 * 0.8 = 0.1（总数据的10%）
        random_state=42,
        stratify=train_val_df['label']
    )
    
    print(f"[数据集划分] 训练集：{len(train_df)} | 验证集：{len(val_df)} | 测试集：{len(test_df)}")
    return train_df, val_df, test_df

def create_pyg_data(labeled_df, num_nodes):
    """创建PyTorch Geometric数据对象（显式指定num_nodes消除警告）"""
    data_list = []
    for _, row in labeled_df.iterrows():
        # 构造单条边的数据对象
        data = Data(
            edge_index=torch.tensor([[row['source_id']], [row['target_id']]], dtype=torch.long),
            y=torch.tensor([row['label']], dtype=torch.long),
            edge_type=row['edge_type'],
            num_nodes=num_nodes  # 显式指定节点总数
        )
        data_list.append(data)
    return data_list

# -----------------------------------------------------------------------------
# 3. 模型定义（GCN + GAT）
# -----------------------------------------------------------------------------
class GCN(torch.nn.Module):
    """基于GCNConv的图卷积网络（解决维度不匹配）"""
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)    # 第一层卷积：输入→隐藏
        self.conv2 = GCNConv(hidden_dim, hidden_dim//2)# 第二层卷积：隐藏→隐藏/2
        self.fc = torch.nn.Linear(hidden_dim//2, 2)    # 输出层：隐藏/2→2分类
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        """提取节点嵌入"""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        return x

    def predict(self, x, edge_index, source_indices, target_indices):
        """预测边标签（支持批次处理）"""
        node_emb = self.forward(x, edge_index)
        # 边嵌入=源节点嵌入与目标节点嵌入的绝对差
        edge_emb = torch.abs(node_emb[source_indices] - node_emb[target_indices])
        return self.fc(edge_emb)

class GAT(torch.nn.Module):
    """基于GATConv的图注意力网络（多注意力头）"""
    def __init__(self, input_dim, hidden_dim, heads=4, dropout=0.3):
        super().__init__()
        # 第一层：多注意力头（输出维度=hidden_dim*heads）
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        # 第二层：单注意力头（输出维度=hidden_dim）
        self.conv2 = GATConv(hidden_dim*heads, hidden_dim, heads=1, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_dim, 2)  # 输出层：hidden_dim→2分类
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        """提取节点嵌入"""
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # GAT常用ELU激活函数
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        return x

    def predict(self, x, edge_index, source_indices, target_indices):
        """预测边标签（支持批次处理）"""
        node_emb = self.forward(x, edge_index)
        edge_emb = torch.abs(node_emb[source_indices] - node_emb[target_indices])
        return self.fc(edge_emb)

# -----------------------------------------------------------------------------
# 4. 训练工具（早停机制）
# -----------------------------------------------------------------------------
class EarlyStopping:
    """早停机制（解决np.Inf兼容问题）"""
    def __init__(self, patience=5, verbose=False, path='best_model.pt'):
        self.patience = patience  # 容忍验证损失上升的轮次
        self.verbose = verbose    # 是否打印日志
        self.counter = 0          # 计数器
        self.best_score = None    # 最佳分数（-验证损失）
        self.early_stop = False   # 是否早停
        self.val_loss_min = np.inf# 最小验证损失（NumPy 2.0用np.inf）
        self.path = path          # 最佳模型保存路径

    def __call__(self, val_loss, model):
        # 计算分数（负损失，分数越高损失越小）
        score = -val_loss

        # 第一次训练：初始化最佳分数并保存模型
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        # 验证损失上升：计数器+1
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"[早停] 计数：{self.counter}/{self.patience} | 当前最小损失：{self.val_loss_min:.4f}")
            # 计数器达到阈值：触发早停
            if self.counter >= self.patience:
                self.early_stop = True
        # 验证损失下降：更新最佳分数并保存模型
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """保存验证损失下降的模型"""
        if self.verbose:
            print(f"[早停] 验证损失下降：{self.val_loss_min:.4f} → {val_loss:.4f} | 保存模型到 {self.path}")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# -----------------------------------------------------------------------------
# 5. 训练与评估流程
# -----------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, node_feats, all_edge_index, 
                epochs=100, lr=1e-4, patience=5, device='cpu'):
    """模型训练流程（解决设备不匹配、批次处理）"""
    # 1. 初始化优化器、损失函数、早停
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()  # 二分类损失
    early_stopping = EarlyStopping(
        patience=patience, 
        verbose=True, 
        path=f"{model.__class__.__name__}_best.pt"
    )

    # 2. 转移所有张量到目标设备（确保CPU/CUDA一致）
    model = model.to(device)
    node_feats = node_feats.to(device)
    all_edge_index = all_edge_index.to(device)

    # 3. 训练历史记录
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }

    # 4. 开始训练
    for epoch in range(epochs):
        # -------------------------- 训练阶段 --------------------------
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} | 训练"):
            data = data.to(device)  # 转移数据到设备
            optimizer.zero_grad()   # 清空梯度

            # 获取批次内所有边的索引（支持批次处理）
            source_indices = data.edge_index[0]
            target_indices = data.edge_index[1]

            # 过滤无效索引（双重保险）
            max_node_idx = node_feats.shape[0] - 1
            valid_mask = (source_indices <= max_node_idx) & (target_indices <= max_node_idx)
            if not valid_mask.any():
                continue  # 跳过空批次
            source_indices = source_indices[valid_mask]
            target_indices = target_indices[valid_mask]

            # 模型预测与损失计算
            logits = model.predict(node_feats, all_edge_index, source_indices, target_indices)
            labels = data.y.squeeze()[valid_mask]  # 过滤标签维度
            loss = criterion(logits, labels)

            # 反向传播与参数更新
            loss.backward()
            optimizer.step()

            # 记录训练指标
            train_loss += loss.item()
            preds = logits.argmax(dim=1)  # 取概率最大的类别
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # -------------------------- 验证阶段 --------------------------
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():  # 禁用梯度计算
            for data in val_loader:
                data = data.to(device)
                source_indices = data.edge_index[0]
                target_indices = data.edge_index[1]

                # 过滤无效索引
                max_node_idx = node_feats.shape[0] - 1
                valid_mask = (source_indices <= max_node_idx) & (target_indices <= max_node_idx)
                if not valid_mask.any():
                    continue
                source_indices = source_indices[valid_mask]
                target_indices = target_indices[valid_mask]

                # 模型预测与损失计算
                logits = model.predict(node_feats, all_edge_index, source_indices, target_indices)
                labels = data.y.squeeze()[valid_mask]
                loss = criterion(logits, labels)

                # 记录验证指标
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # -------------------------- 指标计算 --------------------------
        # 训练集指标
        train_loss_avg = train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)  # 平衡类别评估

        # 验证集指标
        val_loss_avg = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)

        # 保存历史记录
        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        # 打印日志
        print(f"\n[Epoch {epoch+1}]")
        print(f"训练：损失={train_loss_avg:.4f} | 准确率={train_acc:.4f} | F1={train_f1:.4f}")
        print(f"验证：损失={val_loss_avg:.4f} | 准确率={val_acc:.4f} | F1={val_f1:.4f}")

        # 早停检查
        early_stopping(val_loss_avg, model)
        if early_stopping.early_stop:
            print(f"[早停] 验证损失连续 {patience} 轮未下降，停止训练")
            break

    # 加载最佳模型权重
    model.load_state_dict(torch.load(early_stopping.path))
    return model, history


def evaluate_model(model, test_loader, node_feats, all_edge_index, device='cpu'):
    """模型评估流程"""
    model.eval()
    model = model.to(device)
    node_feats = node_feats.to(device)
    all_edge_index = all_edge_index.to(device)

    test_preds, test_labels = [], []
    with torch.no_grad():
        for data in tqdm(test_loader, desc="评估测试集"):
            data = data.to(device)
            source_indices = data.edge_index[0]
            target_indices = data.edge_index[1]

            # 过滤无效索引
            max_node_idx = node_feats.shape[0] - 1
            valid_mask = (source_indices <= max_node_idx) & (target_indices <= max_node_idx)
            if not valid_mask.any():
                continue
            source_indices = source_indices[valid_mask]
            target_indices = target_indices[valid_mask]

            # 模型预测
            logits = model.predict(node_feats, all_edge_index, source_indices, target_indices)
            labels = data.y.squeeze()[valid_mask]
            
            preds = logits.argmax(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    # 计算评估指标
    acc = accuracy_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds)

    # 人工抽样100条验证（修复语法错误）
    sample_size = min(100, len(test_preds))
    sample_indices = random.sample(range(len(test_preds)), sample_size)  # 正确生成抽样索引
    sample_preds = [test_preds[i] for i in sample_indices]
    sample_labels = [test_labels[i] for i in sample_indices]
    sample_acc = accuracy_score(sample_labels, sample_preds)

    print(f"\n[测试集评估] 准确率：{acc:.4f} | F1 分数：{f1:.4f}")
    print(f"[人工抽样验证] 100 条样本准确率：{sample_acc:.4f}")

    return {
        'accuracy': acc,
        'f1': f1,
        'sample_accuracy': sample_acc,
        'predictions': test_preds,
        'labels': test_labels
    }


def plot_training_history(history, model_name):
    """绘制训练历史曲线"""
    plt.figure(figsize=(12, 4))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title(f'{model_name} 损失曲线')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title(f'{model_name} 准确率曲线')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{model_name}_训练曲线.png')
    plt.close()
    print(f"[可视化] 训练曲线已保存为 {model_name}_训练曲线.png")

def main():
    # 1. 基础配置
    set_seed(42)  # 固定随机种子
    BATCH_SIZE = 16  # 小批次减少内存占用
    HIDDEN_DIM = 64  # 模型隐藏层维度
    EPOCHS = 100  # 最大训练轮次
    LEARNING_RATE = 1e-4  # 学习率
    PATIENCE = 5  # 早停容忍轮次

    # 2. 设备选择（优先CUDA，失败则自动切换CPU）
    try:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 验证CUDA可用性（避免假阳性）
        if DEVICE == 'cuda':
            torch.tensor([1.]).to(DEVICE)
        print(f"[设备信息] 使用 {DEVICE} 进行训练")
    except Exception as e:
        DEVICE = 'cpu'
        print(f"[设备信息] CUDA不可用（{str(e)}），切换到 CPU 训练")

    # 3. 加载图数据（从build模块获取预处理后的图）
    print("\n[数据加载] 开始加载融合后的图数据...")
    try:
        # 初始化图构造器（需确保数据路径正确）
        ENRON_DATA_DIR = os.path.join(project_root, "enron_processed_data")  # 可根据实际路径调整
        FACEBOOK_DATA_DIR = os.path.join(project_root, "facebook_processed_data")
        graph_constructor = CombinedGraphConstructor(ENRON_DATA_DIR, FACEBOOK_DATA_DIR)
        
        # 执行图构建流程
        graph_constructor.define_graph_structure()  # 定义图结构（节点/边属性）
        graph_constructor.build_pyg_graph()         # 构建PyG格式图
        
        # 提取关键数据
        node_features = graph_constructor.pyg_data.x  # 节点特征矩阵 (num_nodes, input_dim)
        all_edge_index = graph_constructor.pyg_data.edge_index  # 全局边索引 (2, num_edges)
        combined_edges = graph_constructor.combined_edges  # 融合后的边数据表
        combined_users = graph_constructor.combined_users  # 融合后的用户数据表
        user_id_map = graph_constructor.user_id_map        # 用户ID→连续索引映射表
        num_nodes = node_features.shape[0]                 # 总节点数
        
        print(f"[数据加载] 完成！节点数：{num_nodes} | 边数：{all_edge_index.shape[1]} | 节点特征维度：{node_features.shape[1]}")
    except Exception as e:
        print(f"[数据加载] 失败！错误信息：{str(e)}")
        return

    # 4. 数据集标注与划分
    print("\n[数据集处理] 开始标注和划分数据...")
    try:
        # 标注带标签的边数据集（正/负样本）
        labeled_df = create_labeled_dataset(combined_edges, combined_users, user_id_map)
        if len(labeled_df) == 0:
            print("[数据集处理] 错误：未生成有效标注数据")
            return
        
        # 划分训练/验证/测试集
        train_df, val_df, test_df = split_dataset(labeled_df)
        
        # 转换为PyG Data对象列表
        train_data = create_pyg_data(train_df, num_nodes)
        val_data = create_pyg_data(val_df, num_nodes)
        test_data = create_pyg_data(test_df, num_nodes)
        
        # 创建DataLoader（批次加载）
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        
        print(f"[数据集处理] 完成！训练批次：{len(train_loader)} | 验证批次：{len(val_loader)} | 测试批次：{len(test_loader)}")
    except Exception as e:
        print(f"[数据集处理] 失败！错误信息：{str(e)}")
        return

    # 5. 训练GCN模型
    print("\n" + "="*50)
    print("[模型训练] 开始训练 GCN 模型...")
    print("="*50)
    try:
        # 初始化GCN模型（输入维度=节点特征维度）
        input_dim = node_features.shape[1]
        gcn_model = GCN(input_dim=input_dim, hidden_dim=HIDDEN_DIM, dropout=0.3)
        
        # 执行训练
        trained_gcn, gcn_history = train_model(
            model=gcn_model,
            train_loader=train_loader,
            val_loader=val_loader,
            node_feats=node_features,
            all_edge_index=all_edge_index,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            patience=PATIENCE,
            device=DEVICE
        )
        
        # 绘制训练曲线
        plot_training_history(gcn_history, "GCN")
        
        # 评估GCN模型
        print("\n[模型评估] 开始评估 GCN 模型...")
        gcn_results = evaluate_model(
            model=trained_gcn,
            test_loader=test_loader,
            node_feats=node_features,
            all_edge_index=all_edge_index,
            device=DEVICE
        )
    except Exception as e:
        print(f"[GCN训练/评估] 失败！错误信息：{str(e)}")
        return

    # 6. 训练GAT模型
    print("\n" + "="*50)
    print("[模型训练] 开始训练 GAT 模型...")
    print("="*50)
    try:
        # 初始化GAT模型（多注意力头）
        gat_model = GAT(input_dim=input_dim, hidden_dim=HIDDEN_DIM, heads=4, dropout=0.3)
        
        # 执行训练
        trained_gat, gat_history = train_model(
            model=gat_model,
            train_loader=train_loader,
            val_loader=val_loader,
            node_feats=node_features,
            all_edge_index=all_edge_index,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            patience=PATIENCE,
            device=DEVICE
        )
        
        # 绘制训练曲线
        plot_training_history(gat_history, "GAT")
        
        # 评估GAT模型
        print("\n[模型评估] 开始评估 GAT 模型...")
        gat_results = evaluate_model(
            model=trained_gat,
            test_loader=test_loader,
            node_feats=node_features,
            all_edge_index=all_edge_index,
            device=DEVICE
        )
    except Exception as e:
        print(f"[GAT训练/评估] 失败！错误信息：{str(e)}")
        return

    # 7. 模型对比总结
    print("\n" + "="*50)
    print("[项目总结] 模型性能对比")
    print("="*50)
    print(f"{'模型':<10} {'测试集准确率':<15} {'测试集F1':<15} {'抽样100条准确率':<15}")
    print("-"*50)
    print(f"{'GCN':<10} {gcn_results['accuracy']:.4f}{'':<7} {gcn_results['f1']:.4f}{'':<7} {gcn_results['sample_accuracy']:.4f}{'':<7}")
    print(f"{'GAT':<10} {gat_results['accuracy']:.4f}{'':<7} {gat_results['f1']:.4f}{'':<7} {gat_results['sample_accuracy']:.4f}{'':<7}")
    print("="*50)
    print("\n[项目总结] 所有流程执行完成！最佳模型已保存为 GCN_best.pt 和 GAT_best.pt")

# 程序入口
if __name__ == "__main__":
    main()