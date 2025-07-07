#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
银行营销数据集预处理脚本
数据集：葡萄牙银行直接营销活动数据
目标：预测客户是否会订阅定期存款
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# 设置字体（使用默认英文字体）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class BankDataPreprocessor:
    """银行营销数据预处理类"""
    
    def __init__(self, data_path):
        """
        初始化预处理器
        
        参数:
        data_path: CSV数据文件路径
        """
        self.data_path = data_path
        self.data = None
        self.processed_data = None
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = None
        
    def load_data(self):
        """加载CSV数据"""
        print("正在加载数据...")
        try:
            # 根据数据格式，使用分号作为分隔符
            self.data = pd.read_csv(self.data_path, sep=';')
            print(f"数据加载成功！数据形状: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
    
    def explore_data(self):
        """数据探索性分析"""
        if self.data is None:
            print("请先加载数据！")
            return
            
        print("\n" + "="*50)
        print("数据洞察")
        print("="*50)
        
        # 基本信息
        print("\n1. 数据基本信息:")
        print(f"数据集形状: {self.data.shape}")
        print(f"特征数量: {self.data.shape[1] - 1}")  # 减去目标变量
        print(f"样本数量: {self.data.shape[0]}")
        
        # 数据类型
        print("\n2. 数据类型:")
        print(self.data.dtypes)
        
        # 缺失值检查
        print("\n3. 缺失值检查:")
        missing_values = self.data.isnull().sum()
        print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "无缺失值")
        
        # 目标变量分布
        print("\n4. 目标变量分布:")
        target_counts = self.data['y'].value_counts()
        print(target_counts)
        print(f"订阅率: {target_counts['yes'] / len(self.data) * 100:.2f}%")
        
        # 数值型特征描述统计
        print("\n5. 数值型特征描述统计:")
        numeric_features = self.data.select_dtypes(include=[np.number]).columns
        print(self.data[numeric_features].describe())
        
        # 分类特征唯一值统计
        print("\n6. 分类特征唯一值统计:")
        categorical_features = self.data.select_dtypes(include=['object']).columns
        for col in categorical_features:
            print(f"{col}: {self.data[col].nunique()} 个唯一值")
            # if self.data[col].nunique() <= 10:
            print(f"  值分布: {dict(self.data[col].value_counts())}")
        
        return self.data.head()
    
    def visualize_data(self):
        """数据可视化"""
        if self.data is None:
            print("请先加载数据！")
            return
            
        print("\n正在生成数据可视化图表...")
        
        # 创建图形
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Bank Marketing Data Exploratory Analysis', fontsize=16, fontweight='bold')
        
        # 1. 目标变量分布
        target_counts = self.data['y'].value_counts()
        axes[0,0].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%')
        axes[0,0].set_title('Target Variable Distribution')
        
        # 2. 年龄分布
        axes[0,1].hist(self.data['age'], bins=30, edgecolor='black', alpha=0.7)
        axes[0,1].set_title('Age Distribution')
        axes[0,1].set_xlabel('Age')
        axes[0,1].set_ylabel('Frequency')
        
        # 3. 余额分布
        axes[0,2].hist(self.data['balance'], bins=50, edgecolor='black', alpha=0.7)
        axes[0,2].set_title('Account Balance Distribution')
        axes[0,2].set_xlabel('Balance (EUR)')
        axes[0,2].set_ylabel('Frequency')
        
        # 4. 职业分布
        job_counts = self.data['job'].value_counts()
        axes[1,0].bar(range(len(job_counts)), job_counts.values)
        axes[1,0].set_title('Job Distribution')
        axes[1,0].set_xticks(range(len(job_counts)))
        axes[1,0].set_xticklabels(job_counts.index, rotation=45, ha='right')
        
        # 5. 教育程度分布
        edu_counts = self.data['education'].value_counts()
        axes[1,1].bar(edu_counts.index, edu_counts.values)
        axes[1,1].set_title('Education Level Distribution')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # 6. 婚姻状况分布
        marital_counts = self.data['marital'].value_counts()
        axes[1,2].bar(marital_counts.index, marital_counts.values)
        axes[1,2].set_title('Marital Status Distribution')
        
        # 7. 联系时长分布
        axes[2,0].hist(self.data['duration'], bins=50, edgecolor='black', alpha=0.7)
        axes[2,0].set_title('Contact Duration Distribution')
        axes[2,0].set_xlabel('Duration (seconds)')
        axes[2,0].set_ylabel('Frequency')
        
        # 8. 联系次数分布
        axes[2,1].hist(self.data['campaign'], bins=20, edgecolor='black', alpha=0.7)
        axes[2,1].set_title('Campaign Contact Count Distribution')
        axes[2,1].set_xlabel('Contact Count')
        axes[2,1].set_ylabel('Frequency')
        
        # 9. 目标变量与年龄关系
        yes_data = self.data[self.data['y'] == 'yes']['age']
        no_data = self.data[self.data['y'] == 'no']['age']
        axes[2,2].hist([yes_data, no_data], bins=20, label=['Subscribed', 'Not Subscribed'], 
                      alpha=0.7, edgecolor='black')
        axes[2,2].set_title('Age vs Subscription Relationship')
        axes[2,2].set_xlabel('Age')
        axes[2,2].set_ylabel('Frequency')
        axes[2,2].legend()
        
        plt.tight_layout()
        plt.savefig('bank_data_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 相关性热力图
        plt.figure(figsize=(12, 10))
        numeric_data = self.data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('Numeric Features Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def clean_data(self):
        """数据清洗"""
        if self.data is None:
            print("请先加载数据！")
            return
            
        print("\n正在进行数据清洗...")
        
        # 复制数据
        self.processed_data = self.data.copy()
        
        # 1. 处理字符串中的引号
        string_columns = self.processed_data.select_dtypes(include=['object']).columns
        for col in string_columns:
            self.processed_data[col] = self.processed_data[col].str.strip('"')
        
        # 2. 检查异常值
        print("检查数值型特征的异常值...")
        numeric_columns = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
        
        for col in numeric_columns:
            if col in self.processed_data.columns:
                Q1 = self.processed_data[col].quantile(0.25)
                Q3 = self.processed_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.processed_data[(self.processed_data[col] < lower_bound) | 
                                             (self.processed_data[col] > upper_bound)]
                print(f"{col}: 发现 {len(outliers)} 个异常值")
        
        # 3. 处理特殊值
        # pdays = -1 表示之前未联系过，这是正常值
        print(f"pdays = -1 的记录数: {(self.processed_data['pdays'] == -1).sum()}")
        
        # 4. 数据类型转换
        print("进行数据类型转换...")
        
        # 确保数值型列为正确类型
        for col in numeric_columns:
            if col in self.processed_data.columns:
                self.processed_data[col] = pd.to_numeric(self.processed_data[col], errors='coerce')
        
        print(f"清洗完成！处理后数据形状: {self.processed_data.shape}")
        return self.processed_data
    
    def feature_engineering(self):
        """特征工程"""
        if self.processed_data is None:
            print("请先进行数据清洗！")
            return
            
        print("\n正在进行特征工程...")
        
        # 1. 创建新特征
        print("创建新特征...")
        
        # 年龄分组
        self.processed_data['age_group'] = pd.cut(self.processed_data['age'], 
                                                 bins=[0, 25, 35, 50, 65, 100], 
                                                 labels=['Young', 'Young-Adult', 'Middle-aged', 'Senior', 'Elderly'])
        
        # 余额分组
        self.processed_data['balance_group'] = pd.cut(self.processed_data['balance'], 
                                                     bins=[-np.inf, 0, 1000, 5000, np.inf], 
                                                     labels=['Debt', 'Low Balance', 'Medium Balance', 'High Balance'])
        
        # 联系时长分组
        self.processed_data['duration_group'] = pd.cut(self.processed_data['duration'], 
                                                      bins=[0, 120, 300, 600, np.inf], 
                                                      labels=['Short', 'Medium', 'Long', 'Very Long'])
        
        # 是否为首次联系
        self.processed_data['first_contact'] = (self.processed_data['pdays'] == -1).astype(int)
        
        # 联系效率 (联系时长/联系次数)
        self.processed_data['contact_efficiency'] = self.processed_data['duration'] / self.processed_data['campaign']
        
        # 2. 编码分类变量
        print("编码分类变量...")
        
        categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 
                             'loan', 'contact', 'month', 'poutcome']
        
        for col in categorical_columns:
            if col in self.processed_data.columns:
                le = LabelEncoder()
                self.processed_data[f'{col}_encoded'] = le.fit_transform(self.processed_data[col])
                self.label_encoders[col] = le
        
        # 3. 独热编码（可选）
        print("创建独热编码...")
        categorical_for_onehot = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']

        for col in categorical_for_onehot:
            if col in self.processed_data.columns:
                dummies = pd.get_dummies(self.processed_data[col], prefix=col)
                self.processed_data = pd.concat([self.processed_data, dummies], axis=1)
        
        # 4. 目标变量编码
        target_le = LabelEncoder()
        self.processed_data['y_encoded'] = target_le.fit_transform(self.processed_data['y'])
        self.label_encoders['y'] = target_le
        
        print(f"特征工程完成！最终数据形状: {self.processed_data.shape}")
        return self.processed_data
    
    def prepare_features(self, use_onehot=False, scale_features=True):
        """准备建模特征"""
        if self.processed_data is None:
            print("请先进行特征工程！")
            return None, None
            
        print("\n准备建模特征...")
        
        # 选择特征
        if use_onehot:
            # 使用独热编码特征
            feature_columns = []
            
            # 数值特征
            numeric_features = ['age', 'balance', 'duration', 'campaign', 'previous', 
                              'contact_efficiency', 'first_contact']
            feature_columns.extend(numeric_features)
            
            # 独热编码特征
            onehot_columns = [col for col in self.processed_data.columns 
                            if any(col.startswith(prefix) for prefix in 
                                  ['job_', 'marital_', 'education_', 'contact_', 'month_', 'poutcome_'])]
            feature_columns.extend(onehot_columns)
            
            # 二元特征
            binary_features = ['default_encoded', 'housing_encoded', 'loan_encoded']
            feature_columns.extend([col for col in binary_features 
                                  if col in self.processed_data.columns])
            
        else:
            # 使用标签编码特征
            feature_columns = ['age', 'job_encoded', 'marital_encoded', 'education_encoded',
                             'default_encoded', 'balance', 'housing_encoded', 'loan_encoded',
                             'contact_encoded', 'duration', 'campaign', 'previous',
                             'poutcome_encoded', 'contact_efficiency', 'first_contact']
            
            # 过滤存在的列
            feature_columns = [col for col in feature_columns 
                             if col in self.processed_data.columns]
        
        # 提取特征和目标变量
        X = self.processed_data[feature_columns]
        y = self.processed_data['y_encoded']
        
        # 特征缩放
        if scale_features:
            print("进行特征缩放...")
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
        
        self.feature_names = feature_columns
        
        print(f"最终特征数量: {len(feature_columns)}")
        print(f"特征列表: {feature_columns[:10]}...")  # 显示前10个特征
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """数据集划分"""
        print(f"\n划分数据集 (训练集:{1-test_size:.0%}, 测试集:{test_size:.0%})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"训练集形状: {X_train.shape}")
        print(f"测试集形状: {X_test.shape}")
        print(f"训练集订阅率: {y_train.mean():.3f}")
        print(f"测试集订阅率: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, output_path="processed_bank_data.csv"):
        """保存处理后的数据"""
        if self.processed_data is None:
            print("没有处理后的数据可保存！")
            return
            
        self.processed_data.to_csv(output_path, index=False)
        print(f"处理后的数据已保存至: {output_path}")
        
        # 保存特征列表
        if self.feature_names:
            with open("feature_names.txt", "w", encoding='utf-8') as f:
                for feature in self.feature_names:
                    f.write(f"{feature}\n")
            print("特征列表已保存至: feature_names.txt")
    
    def generate_summary_report(self):
        """生成数据预处理总结报告"""
        if self.processed_data is None:
            print("请先完成数据预处理！")
            return
            
        report = f"""
        
        =====================================
        银行营销数据预处理总结报告
        =====================================
        
        原始数据:
        - 样本数量: {self.data.shape[0]:,}
        - 特征数量: {self.data.shape[1]-1}
        - 缺失值: {'无' if self.data.isnull().sum().sum() == 0 else '存在'}
        
        处理后数据:
        - 样本数量: {self.processed_data.shape[0]:,}
        - 总特征数量: {self.processed_data.shape[1]-1}
        - 建模特征数量: {len(self.feature_names) if self.feature_names else 'N/A'}
        
        目标变量分布:
        - 订阅 (yes): {(self.data['y'] == 'yes').sum():,} ({(self.data['y'] == 'yes').mean()*100:.2f}%)
        - 未订阅 (no): {(self.data['y'] == 'no').sum():,} ({(self.data['y'] == 'no').mean()*100:.2f}%)
        
        新增特征:
        - 年龄分组 (age_group)
        - 余额分组 (balance_group)
        - 联系时长分组 (duration_group)
        - 首次联系标识 (first_contact)
        - 联系效率 (contact_efficiency)
        
        编码方式:
        - 标签编码: 分类变量转数值
        - 独热编码: 创建二元特征矩阵
        - 特征缩放: StandardScaler标准化
        
        数据质量:
        - 异常值处理: 已检测并标记
        - 数据类型: 已规范化
        - 特征相关性: 已分析
        
        =====================================
        """
        
        print(report)
        
        # 保存报告
        with open("preprocessing_report.txt", "w", encoding='utf-8') as f:
            f.write(report)
        print("预处理报告已保存至: preprocessing_report.txt")


def main():
    """主函数 - 完整的数据预处理流程"""
    print("银行营销数据预处理开始...")
    print("="*50)
    
    # 1. 初始化预处理器
    data_path = "../bank_data_analysis-master/bank-full.csv"  # 使用完整数据集
    # data_path = "bank_data_analysis-master/bank.csv"  # 或使用子集进行快速测试
    
    preprocessor = BankDataPreprocessor(data_path)
    
    # 2. 加载数据
    data = preprocessor.load_data()
    if data is None:
        return
    
    # 3. 数据探索
    sample_data = preprocessor.explore_data()
    print("\n前5行数据样例:")
    print(sample_data)
    
    # 4. 数据可视化
    preprocessor.visualize_data()
    
    # 5. 数据清洗
    cleaned_data = preprocessor.clean_data()
    
    # 6. 特征工程
    engineered_data = preprocessor.feature_engineering()
    
    # 7. 准备建模特征
    X, y = preprocessor.prepare_features(use_onehot=False, scale_features=True)
    
    # 8. 数据集划分
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # 9. 保存处理后的数据
    preprocessor.save_processed_data()
    
    # 10. 生成总结报告
    preprocessor.generate_summary_report()
    
    print("\n" + "="*50)
    print("数据预处理完成！")
    print("="*50)
    
    return preprocessor, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # 运行完整预处理流程
    preprocessor, X_train, X_test, y_train, y_test = main()
    
    print(f"\n数据预处理结果:")
    print(f"训练集: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"测试集: X_test {X_test.shape}, y_test {y_test.shape}") 