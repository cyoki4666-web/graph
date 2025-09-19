import os
import re
import pandas as pd
import numpy as np
import networkx as nx
from faker import Faker
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from datetime import datetime, timedelta
import email
from email.parser import Parser

# 设置 matplotlib 支持中文显示，解决中文乱码问题
plt.rcParams["font.family"] = ["SimHei"]  
plt.rcParams["axes.unicode_minus"] = False  

# 设置随机种子，保证结果可复现
random.seed(42)
np.random.seed(42)

class EnronEmailPreprocessor:
    def __init__(self, enron_data_dir, output_dir='enron_processed_data'):
        """初始化Enron邮件预处理类
        
        Args:
            enron_data_dir (str): Enron数据集目录
            output_dir (str): 处理后数据保存目录
        """
        self.enron_data_dir = 'D:/cc/graph1.0/datasets/email'
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化Faker用于生成模拟数据
        self.fake = Faker('en_US')  # 使用英文生成器，符合企业邮件场景
        Faker.seed(42)
        
        # 存储处理后的数据
        self.enron_users = None           # Enron用户数据
        self.enron_edges = None           # 邮件交互边数据
        self.enron_email_features = None  # 邮件文本特征
        self.simulated_users = None       # 模拟用户数据
        self.simulated_edges = None       # 模拟关系边数据
        self.combined_users = None        # 合并后的用户数据
        self.combined_edges = None        # 合并后的边数据
        
        # 邮件解析相关配置
        self.email_date_pattern = re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')
        
    def load_enron_data(self, max_emails=None):
        """加载Enron邮件数据集
        
        Args:
            max_emails (int, optional): 最大加载邮件数量，None表示全部加载
        """
        print("开始加载Enron Email数据集...")
        
        # 1. 加载边数据（Email-Enron.txt）
        edge_file_path = os.path.join(self.enron_data_dir, "Email-Enron.txt")
        edges = []
        
        if os.path.exists(edge_file_path):
            with open(edge_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):  # 跳过注释和空行
                        continue
                    
                    try:
                        parts = list(map(int, line.split()))
                        if len(parts) >= 2:
                            source, target = parts[0], parts[1]
                            # 初始交互频率设为1，后续会根据实际邮件数量更新
                            edges.append({
                                'source': source,
                                'target': target,
                                'interaction_freq': 1,
                                'last_interaction': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                    except ValueError as e:
                        print(f"解析边数据出错: {e}, 行内容: {line}")
                        continue
        
        # 2. 解析邮件内容（提取用户、时间和主题）
        user_ids = set()
        email_contents = []
        maildir_path = os.path.join(self.enron_data_dir, "maildir")
        
        if os.path.exists(maildir_path):
            email_count = 0
            # 遍历邮件目录
            for root, dirs, files in os.walk(maildir_path):
                for file in files:
                    if max_emails and email_count >= max_emails:
                        break
                        
                    file_path = os.path.join(root, file)
                    try:
                        # 解析邮件内容
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            msg = Parser().parse(f)
                            
                            # 提取发件人
                            from_addr = msg.get('from', 'unknown')
                            # 提取收件人
                            to_addrs = msg.get('to', 'unknown')
                            # 提取日期
                            date_str = msg.get('date', '')
                            # 提取主题
                            subject = msg.get('subject', '').strip()
                            
                            # 简单处理日期格式
                            try:
                                date_obj = email.utils.parsedate_to_datetime(date_str)
                                formatted_date = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                            except:
                                formatted_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                
                            # 提取邮件正文
                            body = ""
                            if msg.is_multipart():
                                for part in msg.walk():
                                    content_type = part.get_content_type()
                                    if content_type == "text/plain" and not part.get("attachment", False):
                                        body = part.get_payload(decode=True).decode('utf-8', errors='ignore').strip()
                                        break
                            else:
                                body = msg.get_payload(decode=True).decode('utf-8', errors='ignore').strip()
                            
                            # 保存邮件特征
                            email_contents.append({
                                'email_id': email_count,
                                'from_addr': from_addr,
                                'to_addrs': to_addrs,
                                'date': formatted_date,
                                'subject': subject,
                                'body': body[:500]  # 只保留前500字符
                            })
                            
                            email_count += 1
                            if email_count % 1000 == 0:
                                print(f"已解析 {email_count} 封邮件")
                                
                    except Exception as e:
                        print(f"解析邮件 {file_path} 出错: {e}")
                        continue
        
        # 3. 创建用户数据（基于邮件地址生成用户ID映射）
        all_emails = [ec['from_addr'] for ec in email_contents]
        for ec in email_contents:
            if ec['to_addrs'] and ec['to_addrs'] != 'unknown':
                all_emails.extend([addr.strip() for addr in ec['to_addrs'].split(',')])
        
        # 去重并创建邮箱到用户ID的映射
        unique_emails = list(set(all_emails))
        email_to_id = {email: i+20000 for i, email in enumerate(unique_emails)}  # 20000+避免与Facebook用户ID冲突
        
        # 创建用户数据
        users_data = []
        departments = ['技术部', '市场部', '财务部', '人力资源', '管理层', '法务部', '销售部', '客服部']
        for email, user_id in email_to_id.items():
            # 从邮箱提取简单信息作为特征
            username = email.split('@')[0] if '@' in email else email
            users_data.append({
                'user_id': user_id,
                'email': email,
                'username': username,
                'age': random.randint(22, 60),  # 企业员工年龄范围
                'department': random.choice(departments),
                'position': random.choice(['经理', '主管', '专员', '助理', '总监', '分析师'])
            })
        
        self.enron_users = pd.DataFrame(users_data)
        
        # 4. 更新边数据（基于实际邮件交互）
        if edges:
            self.enron_edges = pd.DataFrame(edges)
            # 基于实际邮件更新交互频率
            for ec in email_contents:
                if ec['from_addr'] in email_to_id and ec['to_addrs'] != 'unknown':
                    source_id = email_to_id[ec['from_addr']]
                    for to_addr in [addr.strip() for addr in ec['to_addrs'].split(',')]:
                        if to_addr in email_to_id:
                            target_id = email_to_id[to_addr]
                            # 更新交互频率
                            mask = ((self.enron_edges['source'] == source_id) & 
                                    (self.enron_edges['target'] == target_id))
                            if mask.any():
                                self.enron_edges.loc[mask, 'interaction_freq'] += 1
                                self.enron_edges.loc[mask, 'last_interaction'] = ec['date']
                            else:
                                # 添加新边
                                self.enron_edges = pd.concat([self.enron_edges, pd.DataFrame([{
                                    'source': source_id,
                                    'target': target_id,
                                    'interaction_freq': 1,
                                    'last_interaction': ec['date']
                                }])], ignore_index=True)
        
        # 保存邮件特征数据
        self.enron_email_features = pd.DataFrame(email_contents)
        
        print(f"Enron数据集加载完成: "
              f"{len(self.enron_users)}个用户, "
              f"{len(self.enron_edges) if self.enron_edges is not None else 0}条边, "
              f"{len(self.enron_email_features)}封邮件")
        return self
    
    def generate_simulated_data(self, num_users=500):
        """使用Faker和NetworkX生成模拟企业用户数据
        
        Args:
            num_users (int): 生成的用户数量
        """
        print(f"开始生成模拟企业用户数据，用户数量: {num_users}")
        
        # 生成用户属性
        users_data = []
        departments = ['技术部', '市场部', '财务部', '人力资源', '管理层', '法务部', '销售部', '客服部']
        positions = ['经理', '主管', '专员', '助理', '总监', '分析师']
        
        for i in range(num_users):
            user_id = 30000 + i  # 确保与Enron用户ID不冲突
            name = self.fake.name()
            email = self.fake.email()
            age = random.randint(22, 60)
            department = random.choice(departments)
            position = random.choice(positions)
            
            users_data.append({
                'user_id': user_id,
                'email': email,
                'username': name.split()[0],
                'age': age,
                'department': department,
                'position': position
            })
        
        self.simulated_users = pd.DataFrame(users_data)
        
        # 使用NetworkX生成企业社交网络（小世界网络更符合实际）
        G = nx.watts_strogatz_graph(n=num_users, k=5, p=0.1, seed=42)
        
        # 生成边数据
        edges_data = []
        for (u, v) in G.edges():
            # 转换为实际用户ID
            source = 30000 + u
            target = 30000 + v
            
            # 随机生成交互频率（1-1200次）
            interaction_freq = random.randint(1, 1200)
            
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
        
        # 合并用户数据
        if self.enron_users is not None and self.simulated_users is not None:
            self.combined_users = pd.concat([self.enron_users, self.simulated_users], ignore_index=True)
        elif self.enron_users is not None:
            self.combined_users = self.enron_users
        else:
            self.combined_users = self.simulated_users
        
        # 合并边数据
        all_edges = []
        if self.enron_edges is not None:
            all_edges.append(self.enron_edges)
        if self.simulated_edges is not None:
            all_edges.append(self.simulated_edges)
        self.combined_edges = pd.concat(all_edges, ignore_index=True) if all_edges else None
        
        # 1. 处理边数据中的异常值（交互频率）
        if self.combined_edges is not None and 'interaction_freq' in self.combined_edges.columns:
            # 绘制箱线图查看异常值
            plt.figure(figsize=(10, 6))
            plt.boxplot(self.combined_edges['interaction_freq'])
            plt.title('邮件交互频率箱线图（清洗前）')
            plt.savefig(os.path.join(self.output_dir, 'enron_interaction_freq_before.png'))
            plt.close()
            
            # 使用箱线图方法检测异常值
            q1 = self.combined_edges['interaction_freq'].quantile(0.25)
            q3 = self.combined_edges['interaction_freq'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # 使用固定阈值1000
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
            plt.title('邮件交互频率箱线图（清洗后）')
            plt.savefig(os.path.join(self.output_dir, 'enron_interaction_freq_after.png'))
            plt.close()
        
        # 2. 去重
        # 用户数据去重（基于email）
        if self.combined_users is not None:
            user_duplicates = self.combined_users.duplicated('email').sum()
            self.combined_users = self.combined_users.drop_duplicates('email', keep='first')
            # 基于user_id去重
            user_id_duplicates = self.combined_users.duplicated('user_id').sum()
            self.combined_users = self.combined_users.drop_duplicates('user_id', keep='first')
            total_duplicates = user_duplicates + user_id_duplicates
        
        # 边数据去重（考虑双向边）
        edge_duplicates = 0
        if self.combined_edges is not None:
            # 创建标准化的边表示
            self.combined_edges['sorted_pair'] = self.combined_edges.apply(
                lambda row: tuple(sorted([row['source'], row['target']])), axis=1
            )
            edge_duplicates = self.combined_edges.duplicated('sorted_pair').sum()
            self.combined_edges = self.combined_edges.drop_duplicates('sorted_pair', keep='first').drop('sorted_pair', axis=1)
        
        print(f"去重处理: 去除重复用户 {total_duplicates} 个, 去除重复边 {edge_duplicates} 条")
        
        # 3. 处理缺失值
        # 处理用户数据缺失值
        if self.combined_users is not None:
            # 数值型：年龄用均值填充
            if 'age' in self.combined_users.columns:
                age_mean = self.combined_users['age'].mean()
                age_missing = self.combined_users['age'].isnull().sum()
                self.combined_users['age'] = self.combined_users['age'].fillna(age_mean)
                if age_missing > 0:
                    print(f"填充年龄缺失值: {age_missing}条, 使用均值 {age_mean:.1f}")
            
            # 类别型：部门和职位用众数填充
            for col in ['department', 'position']:
                if col in self.combined_users.columns:
                    mode_val = self.combined_users[col].mode()[0]
                    missing = self.combined_users[col].isnull().sum()
                    self.combined_users[col] = self.combined_users[col].fillna(mode_val)
                    if missing > 0:
                        print(f"填充{col}缺失值: {missing}条, 使用众数 {mode_val}")
        
        # 处理边数据缺失值
        if self.combined_edges is not None:
            # 交互频率用0填充
            if 'interaction_freq' in self.combined_edges.columns:
                freq_missing = self.combined_edges['interaction_freq'].isnull().sum()
                self.combined_edges['interaction_freq'] = self.combined_edges['interaction_freq'].fillna(0)
                if freq_missing > 0:
                    print(f"填充交互频率缺失值: {freq_missing}条, 使用0填充")
            
            # 时间用默认时间填充
            if 'last_interaction' in self.combined_edges.columns:
                time_missing = self.combined_edges['last_interaction'].isnull().sum()
                default_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.combined_edges['last_interaction'] = self.combined_edges['last_interaction'].fillna(default_time)
                if time_missing > 0:
                    print(f"填充交互时间缺失值: {time_missing}条, 使用默认时间 {default_time}")
        
        # 4. 统一时间格式
        if self.combined_edges is not None and 'last_interaction' in self.combined_edges.columns:
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
        if self.combined_users is not None:
            self.combined_users.to_csv(os.path.join(self.output_dir, 'cleaned_users.csv'), index=False)
        if self.combined_edges is not None:
            self.combined_edges.to_csv(os.path.join(self.output_dir, 'cleaned_edges.csv'), index=False)
        if self.enron_email_features is not None:
            self.enron_email_features.to_csv(os.path.join(self.output_dir, 'cleaned_emails.csv'), index=False)
        
        print("数据清洗完成")
        return self
    
    def feature_engineering(self):
        """特征工程：处理结构化特征和文本特征"""
        print("开始特征工程...")
        
        if self.combined_users is None or self.combined_edges is None:
            print("请先加载并清洗数据")
            return self
        
        # 1. 处理用户特征
        users_processed = self.combined_users.copy()
        
        # 归一化年龄（数值型特征）
        if 'age' in users_processed.columns:
            scaler = MinMaxScaler()
            users_processed['age_normalized'] = scaler.fit_transform(
                users_processed['age'].values.reshape(-1, 1)
            )
            print("年龄归一化完成")
        
        # 对部门和职位进行One-Hot编码（类别型特征）
        for col in ['department', 'position']:
            if col in users_processed.columns:
                encoder = OneHotEncoder(sparse_output=False, drop='first')
                encoded = encoder.fit_transform(users_processed[col].values.reshape(-1, 1))
                
                # 创建编码后的列
                encoded_columns = [f'{col}_{cat}' for cat in encoder.categories_[0][1:]]
                encoded_df = pd.DataFrame(encoded, columns=encoded_columns)
                
                # 合并回用户数据
                users_processed = pd.concat([users_processed.reset_index(drop=True), encoded_df], axis=1)
                print(f"{col} One-Hot编码完成，生成 {len(encoded_columns)} 个特征列")
        
        # 2. 处理边特征
        edges_processed = self.combined_edges.copy()
        
        # 归一化交互频率
        if 'interaction_freq' in edges_processed.columns:
            scaler = MinMaxScaler()
            edges_processed['interaction_freq_normalized'] = scaler.fit_transform(
                edges_processed['interaction_freq'].values.reshape(-1, 1)
            )
            print("交互频率归一化完成")
        
        # 3. 提取邮件文本TF-IDF特征
        email_tfidf = None
        if self.enron_email_features is not None and len(self.enron_email_features) > 0:
            # 合并主题和正文前100字符作为文本源
            self.enron_email_features['text'] = self.enron_email_features['subject'] + " " + \
                                               self.enron_email_features['body'].str[:100]
            
            # 简单文本预处理
            self.enron_email_features['text'] = self.enron_email_features['text'].str.lower()
            self.enron_email_features['text'] = self.enron_email_features['text'].replace(r'[^\w\s]', ' ', regex=True)
            
            # 提取TF-IDF特征（限制最多1000个特征）
            tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = tfidf.fit_transform(self.enron_email_features['text'])
            
            # 转换为DataFrame
            email_tfidf = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
            )
            # 合并邮件ID
            email_tfidf['email_id'] = self.enron_email_features['email_id'].values
            
            print(f"邮件文本TF-IDF特征提取完成，生成 {tfidf_matrix.shape[1]} 个特征")
        
        # 保存处理后的特征数据
        users_processed.to_csv(os.path.join(self.output_dir, 'users_with_features.csv'), index=False)
        edges_processed.to_csv(os.path.join(self.output_dir, 'edges_with_features.csv'), index=False)
        if email_tfidf is not None:
            email_tfidf.to_csv(os.path.join(self.output_dir, 'email_tfidf_features.csv'), index=False)
        
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
        
        if self.enron_email_features is not None:
            print(f"\n邮件数据: {len(self.enron_email_features)} 条记录")
            print("邮件数据列:", list(self.enron_email_features.columns[:5]))  # 只显示前5列
        
        # 检查缺失值
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
                pd.to_datetime(self.combined_edges['last_interaction'], format="%Y-%m-%d %H:%M:%S")
                print("\n时间格式验证: 所有时间均符合标准格式")
            except ValueError as e:
                print(f"\n时间格式验证失败: {e}")
        
        print("\n数据验证完成，数据可用")


if __name__ == "__main__":
    # 请替换为实际的Enron数据集目录
    ENRON_DATA_DIR = "enron_data"  # 存放Enron数据集的文件夹路径，应包含Email-Enron.txt和maildir
    
    # 创建预处理实例
    preprocessor = EnronEmailPreprocessor(ENRON_DATA_DIR)
    
    # 执行完整的预处理流程
    # 注意：Enron数据集较大，首次运行可能需要较长时间，可通过max_emails参数限制加载数量
    (preprocessor
     .load_enron_data(max_emails=5000)  # 加载Enron数据，限制最多5000封邮件
     .generate_simulated_data()         # 生成模拟数据
     .clean_data()                      # 数据清洗
     .feature_engineering()             # 特征工程
     .validate_data())                  # 验证数据
    
    print("\n所有Enron邮件数据预处理步骤已完成！处理后的数据已保存到 'enron_processed_data' 目录")
