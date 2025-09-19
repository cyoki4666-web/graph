import os
import pandas as pd
import numpy as np
import networkx as nx
from faker import Faker
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import random
from datetime import datetime, timedelta

# 设置随机种子，保证结果可复现
random.seed(42)
np.random.seed(42)
# 设置 matplotlib 支持中文显示，解决中文乱码问题
plt.rcParams["font.family"] = ["SimHei"]  
plt.rcParams["axes.unicode_minus"] = False  

class SocialNetworkPreprocessor:
    def __init__(self, facebook_data_dir, output_dir='processed_data'):
        """初始化预处理类
        
        Args:
            facebook_data_dir (str): Facebook数据集目录
            output_dir (str): 处理后数据保存目录
        """
        self.facebook_data_dir = 'D:/cc/graph1.0/datasets/facebook'
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化Faker用于生成模拟数据
        self.fake = Faker('zh_CN')
        Faker.seed(42)
        
        # 存储处理后的数据
        self.facebook_users = None
        self.facebook_edges = None
        self.simulated_users = None
        self.simulated_edges = None
        self.combined_users = None
        self.combined_edges = None
        
    def load_facebook_data(self):
        """加载Facebook数据集"""
        print("开始加载Facebook数据集...")
        
        # 收集所有用户ID和属性
        all_users = {}
        ego_nodes = []
        
        # 遍历所有ego节点的数据文件
        for filename in os.listdir(self.facebook_data_dir):
            if filename.endswith('.egofeat'):
                # 提取ego节点ID
                ego_id = int(filename.split('.')[0])
                ego_nodes.append(ego_id)
                
                # 加载ego节点属性
                with open(os.path.join(self.facebook_data_dir, filename), 'r') as f:
                    ego_feats = list(map(int, f.readline().split()))
                    all_users[ego_id] = ego_feats
                
                # 加载邻居节点属性
                feat_file = f"{ego_id}.feat"
                if os.path.exists(os.path.join(self.facebook_data_dir, feat_file)):
                    with open(os.path.join(self.facebook_data_dir, feat_file), 'r') as f:
                        for line in f:
                            parts = list(map(int, line.split()))
                            user_id = parts[0]
                            feats = parts[1:]
                            if user_id not in all_users:
                                all_users[user_id] = feats
        
        # 创建用户数据框
        user_ids = list(all_users.keys())
        max_feats = max(len(feats) for feats in all_users.values())
        
        # 填充属性，确保所有用户有相同数量的属性
        user_data = []
        for user_id in user_ids:
            feats = all_users[user_id]
            # 如果属性数量不足，用0填充
            if len(feats) < max_feats:
                feats += [0] * (max_feats - len(feats))
            user_data.append([user_id] + feats)
        
        # 创建DataFrame
        columns = ['user_id'] + [f'attr_{i}' for i in range(max_feats)]
        self.facebook_users = pd.DataFrame(user_data, columns=columns)
        
        # 加载边数据
        edges = []
        for ego_id in ego_nodes:
            edge_file = f"{ego_id}.edges"
            edge_path = os.path.join(self.facebook_data_dir, edge_file)
            if os.path.exists(edge_path):
                with open(edge_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:  # 跳过空行
                            continue
                            
                        try:
                            # 处理一行可能包含多个邻居ID的情况
                            neighbor_ids = list(map(int, line.split()))
                            
                            for neighbor_id in neighbor_ids:
                                # 随机生成交互频率（1-500次）
                                interaction_freq = random.randint(1, 500)
                                # 随机生成交互时间
                                start_date = datetime(2023, 1, 1)
                                end_date = datetime(2024, 1, 1)
                                time_between_dates = end_date - start_date
                                days_between_dates = time_between_dates.days
                                random_number_of_days = random.randrange(days_between_dates)
                                interaction_time = start_date + timedelta(days=random_number_of_days)
                                
                                edges.append({
                                    'source': ego_id,
                                    'target': neighbor_id,
                                    'interaction_freq': interaction_freq,
                                    'last_interaction': interaction_time.strftime("%Y-%m-%d %H:%M:%S")
                                })
                                
                        except ValueError as e:
                            print(f"警告: 在文件 {edge_file} 的第 {line_num} 行解析出错 - {e}")
                            print(f"有问题的行内容: {line}")
                            continue
        
        # 创建边数据框
        self.facebook_edges = pd.DataFrame(edges)
        
        print(f"Facebook数据集加载完成: {len(self.facebook_users)}个用户, {len(self.facebook_edges)}条边")
        return self
    
    def generate_simulated_data(self, num_users=500):
        """使用Faker和NetworkX生成模拟用户数据
        
        Args:
            num_users (int): 生成的用户数量
        """
        print(f"开始生成模拟数据，用户数量: {num_users}")
        
        # 生成用户属性
        users_data = []
        occupations = ['学生', '工程师', '教师', '医生', '商人', '设计师', '作家', '自由职业者', '公务员', '其他']
        
        for i in range(num_users):
            user_id = 10000 + i  # 确保与Facebook用户ID不冲突
            age = random.randint(18, 65)
            occupation = random.choice(occupations)
            gender = random.choice(['男', '女'])
            # 随机生成注册时间
            reg_date = self.fake.date_between(start_date='-5y', end_date='today')
            reg_date = datetime.combine(reg_date, datetime.min.time())
            
            users_data.append({
                'user_id': user_id,
                'age': age,
                'occupation': occupation,
                'gender': gender,
                'registration_date': reg_date.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        self.simulated_users = pd.DataFrame(users_data)
        
        # 使用NetworkX生成社交网络
        G = nx.erdos_renyi_graph(n=num_users, p=0.1, seed=42)
        
        # 生成边数据
        edges_data = []
        for (u, v) in G.edges():
            # 转换为实际用户ID
            source = 10000 + u
            target = 10000 + v
            
            # 随机生成交互频率
            interaction_freq = random.randint(1, 1200)  # 包含一些超过1000的值用于测试异常值处理
            
            # 随机生成最后交互时间
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2024, 1, 1)
            time_between_dates = end_date - start_date
            days_between_dates = time_between_dates.days
            random_number_of_days = random.randrange(days_between_dates)
            interaction_time = start_date + timedelta(days=random_number_of_days)
            
            edges_data.append({
                'source': source,
                'target': target,
                'interaction_freq': interaction_freq,
                'last_interaction': interaction_time.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        self.simulated_edges = pd.DataFrame(edges_data)
        
        print(f"模拟数据生成完成: {len(self.simulated_users)}个用户, {len(self.simulated_edges)}条边")
        return self
    
    def clean_data(self):
        """数据清洗：处理异常值、重复值、缺失值和格式"""
        print("开始数据清洗...")
        
        # 合并Facebook数据和模拟数据（先处理边数据）
        # 为Facebook用户添加模拟的年龄和职业属性以便统一处理
        if self.facebook_users is not None:
            # 为Facebook用户添加年龄和职业属性（模拟）
            self.facebook_users['age'] = np.random.randint(18, 65, size=len(self.facebook_users))
            occupations = ['学生', '工程师', '教师', '医生', '商人', '设计师', '作家', '自由职业者', '公务员', '其他']
            self.facebook_users['occupation'] = np.random.choice(occupations, size=len(self.facebook_users))
            self.facebook_users['gender'] = np.random.choice(['男', '女'], size=len(self.facebook_users))
            
            # 只保留需要的列
            self.facebook_users = self.facebook_users[['user_id', 'age', 'occupation', 'gender']]
        
        # 合并用户数据
        if self.facebook_users is not None and self.simulated_users is not None:
            # 统一用户数据格式
            self.simulated_users = self.simulated_users[['user_id', 'age', 'occupation', 'gender']]
            self.combined_users = pd.concat([self.facebook_users, self.simulated_users], ignore_index=True)
        elif self.facebook_users is not None:
            self.combined_users = self.facebook_users
        else:
            self.combined_users = self.simulated_users
        
        # 合并边数据
        if self.facebook_edges is not None and self.simulated_edges is not None:
            self.combined_edges = pd.concat([self.facebook_edges, self.simulated_edges], ignore_index=True)
        elif self.facebook_edges is not None:
            self.combined_edges = self.facebook_edges
        else:
            self.combined_edges = self.simulated_edges
        
        # 1. 处理边数据中的异常值（交互频率）
        if 'interaction_freq' in self.combined_edges.columns:
            # 绘制箱线图查看异常值
            plt.figure(figsize=(10, 6))
            plt.boxplot(self.combined_edges['interaction_freq'])
            plt.title('交互频率箱线图（清洗前）')
            plt.savefig(os.path.join(self.output_dir, 'interaction_freq_before_cleaning.png'))
            plt.close()
            
            # 使用箱线图方法检测异常值
            q1 = self.combined_edges['interaction_freq'].quantile(0.25)
            q3 = self.combined_edges['interaction_freq'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # 也可以使用固定阈值1000
            threshold = 1000
            upper_bound = min(upper_bound, threshold)
            
            # 剔除异常值
            outlier_count = len(self.combined_edges[self.combined_edges['interaction_freq'] > upper_bound])
            self.combined_edges = self.combined_edges[
                (self.combined_edges['interaction_freq'] >= lower_bound) & 
                (self.combined_edges['interaction_freq'] <= upper_bound)
            ]
            print(f"剔除交互频率异常值: {outlier_count}条记录")
            
            # 绘制清洗后的箱线图
            plt.figure(figsize=(10, 6))
            plt.boxplot(self.combined_edges['interaction_freq'])
            plt.title('交互频率箱线图（清洗后）')
            plt.savefig(os.path.join(self.output_dir, 'interaction_freq_after_cleaning.png'))
            plt.close()
        
        # 2. 去重
        # 用户数据去重
        user_duplicates = self.combined_users.duplicated('user_id').sum()
        self.combined_users = self.combined_users.drop_duplicates('user_id', keep='first')
        
        # 边数据去重（考虑双向边）
        # 创建一个标准化的边表示，确保(source, target)和(target, source)被视为同一条边
        self.combined_edges['sorted_pair'] = self.combined_edges.apply(
            lambda row: tuple(sorted([row['source'], row['target']])), axis=1
        )
        edge_duplicates = self.combined_edges.duplicated('sorted_pair').sum()
        self.combined_edges = self.combined_edges.drop_duplicates('sorted_pair', keep='first').drop('sorted_pair', axis=1)
        
        print(f"去重处理: 去除重复用户 {user_duplicates} 个, 去除重复边 {edge_duplicates} 条")
        
        # 3. 处理缺失值
        # 处理用户数据缺失值
        if 'age' in self.combined_users.columns:
            # 数值型：年龄用均值填充
            age_mean = self.combined_users['age'].mean()
            age_missing = self.combined_users['age'].isnull().sum()
            self.combined_users['age'] = self.combined_users['age'].fillna(age_mean)
            if age_missing > 0:
                print(f"填充年龄缺失值: {age_missing}条, 使用均值 {age_mean:.1f}")
        
        if 'occupation' in self.combined_users.columns:
            # 类别型：职业用众数填充
            occupation_mode = self.combined_users['occupation'].mode()[0]
            occupation_missing = self.combined_users['occupation'].isnull().sum()
            self.combined_users['occupation'] = self.combined_users['occupation'].fillna(occupation_mode)
            if occupation_missing > 0:
                print(f"填充职业缺失值: {occupation_missing}条, 使用众数 {occupation_mode}")
        
        # 处理边数据缺失值
        if 'interaction_freq' in self.combined_edges.columns:
            # 关键交互数据用0填充
            freq_missing = self.combined_edges['interaction_freq'].isnull().sum()
            self.combined_edges['interaction_freq'] = self.combined_edges['interaction_freq'].fillna(0)
            if freq_missing > 0:
                print(f"填充交互频率缺失值: {freq_missing}条, 使用0填充")
        
        # 4. 统一时间格式
        if 'last_interaction' in self.combined_edges.columns:
            # 尝试将时间列转换为标准格式
            try:
                self.combined_edges['last_interaction'] = pd.to_datetime(
                    self.combined_edges['last_interaction'], 
                    format="%Y-%m-%d %H:%M:%S"
                )
                # 转换回字符串格式，确保统一
                self.combined_edges['last_interaction'] = self.combined_edges['last_interaction'].dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                print("时间格式统一完成")
            except Exception as e:
                print(f"时间格式转换出错: {e}")
        
        # 保存清洗后的数据
        self.combined_users.to_csv(os.path.join(self.output_dir, 'cleaned_users.csv'), index=False)
        self.combined_edges.to_csv(os.path.join(self.output_dir, 'cleaned_edges.csv'), index=False)
        
        print("数据清洗完成")
        return self
    
    def feature_engineering(self):
        """特征工程：处理结构化特征"""
        print("开始特征工程...")
        
        if self.combined_users is None or self.combined_edges is None:
            print("请先加载并清洗数据")
            return self
        
        # 复制数据以避免修改原始数据
        users_processed = self.combined_users.copy()
        edges_processed = self.combined_edges.copy()
        
        # 1. 处理用户特征
        # 归一化年龄（数值型特征）
        if 'age' in users_processed.columns:
            scaler = MinMaxScaler()
            users_processed['age_normalized'] = scaler.fit_transform(
                users_processed['age'].values.reshape(-1, 1)
            )
            print("年龄归一化完成")
        
        # 对职业进行One-Hot编码（类别型特征）
        if 'occupation' in users_processed.columns:
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            occupation_encoded = encoder.fit_transform(users_processed['occupation'].values.reshape(-1, 1))
            
            # 创建编码后的列
            occupation_columns = [f'occupation_{cat}' for cat in encoder.categories_[0][1:]]
            occupation_df = pd.DataFrame(occupation_encoded, columns=occupation_columns)
            
            # 合并回用户数据
            users_processed = pd.concat([users_processed.reset_index(drop=True), occupation_df], axis=1)
            print(f"职业One-Hot编码完成，生成 {len(occupation_columns)} 个特征列")
        
        # 2. 处理边特征
        # 归一化交互频率
        if 'interaction_freq' in edges_processed.columns:
            scaler = MinMaxScaler()
            edges_processed['interaction_freq_normalized'] = scaler.fit_transform(
                edges_processed['interaction_freq'].values.reshape(-1, 1)
            )
            print("交互频率归一化完成")
        
        # 保存处理后的特征数据
        users_processed.to_csv(os.path.join(self.output_dir, 'users_with_features.csv'), index=False)
        edges_processed.to_csv(os.path.join(self.output_dir, 'edges_with_features.csv'), index=False)
        
        print("特征工程完成")
        return self
    
    def validate_data(self):
        """验证数据可用性"""
        print("\n数据验证结果:")
        
        if self.combined_users is not None:
            print(f"用户数据: {len(self.combined_users)} 条记录")
            print("用户数据列:", list(self.combined_users.columns))
            print("用户数据前5行:")
            print(self.combined_users.head())
        
        if self.combined_edges is not None:
            print(f"\n边数据: {len(self.combined_edges)} 条记录")
            print("边数据列:", list(self.combined_edges.columns))
            print("边数据前5行:")
            print(self.combined_edges.head())
        
        # 检查是否有缺失值
        if self.combined_users is not None:
            missing_users = self.combined_users.isnull().sum()
            print("\n用户数据缺失值检查:")
            print(missing_users[missing_users > 0])
        
        if self.combined_edges is not None:
            missing_edges = self.combined_edges.isnull().sum()
            print("\n边数据缺失值检查:")
            print(missing_edges[missing_edges > 0])
        
        # 检查时间格式
        if self.combined_edges is not None and 'last_interaction' in self.combined_edges.columns:
            try:
                # 尝试解析时间格式
                pd.to_datetime(self.combined_edges['last_interaction'], format="%Y-%m-%d %H:%M:%S")
                print("\n时间格式验证: 所有时间均符合标准格式")
            except ValueError as e:
                print(f"\n时间格式验证失败: {e}")
        
        print("\n数据验证完成，数据可用")


if __name__ == "__main__":
    # 请替换为实际的Facebook数据集目录
    FACEBOOK_DATA_DIR = "facebook_data"  # 存放Facebook数据集的文件夹路径
    
    # 创建预处理实例
    preprocessor = SocialNetworkPreprocessor(FACEBOOK_DATA_DIR)
    
    # 执行完整的预处理流程
    (preprocessor
     .load_facebook_data()         # 加载Facebook数据
     .generate_simulated_data()    # 生成模拟数据
     .clean_data()                 # 数据清洗
     .feature_engineering()        # 特征工程
     .validate_data())             # 验证数据
    
    print("\n所有数据预处理步骤已完成！处理后的数据已保存到 'processed_data' 目录")
