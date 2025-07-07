## 数据集概述

这是一份**葡萄牙银行直接营销活动**的历史数据集，主要特点如下：

### 基本信息
- **数据集名称**: Bank Marketing
- **创建者**: Paulo Cortez (Univ. Minho) 和 Sérgio Moro (ISCTE-IUL) @ 2012
- **研究目的**: 预测客户是否会订阅银行定期存款产品
- **营销方式**: 基于电话营销活动

### 数据集版本
- **bank-full.csv**: 包含所有样本，按日期排序（2008年5月至2010年11月），共45,211条记录
- **bank.csv**: 随机抽取10%的样本，共4,521条记录，用于计算密集型算法测试

## 完整属性描述

### 输入变量（16个特征）

#### 客户基本信息
1. **age** (数值型): 客户年龄
2. **job** (分类型): 职业类型
   - 取值: "admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student", "blue-collar", "self-employed", "retired", "technician", "services"
3. **marital** (分类型): 婚姻状况
   - 取值: "married", "divorced", "single" (注: "divorced"包括离异和丧偶)
4. **education** (分类型): 教育程度
   - 取值: "unknown", "secondary", "primary", "tertiary"
5. **default** (二元型): 是否有信用违约
   - 取值: "yes", "no"
6. **balance** (数值型): 年平均余额（欧元）
7. **housing** (二元型): 是否有住房贷款
   - 取值: "yes", "no"
8. **loan** (二元型): 是否有个人贷款
   - 取值: "yes", "no"

#### 当前营销活动相关
9. **contact** (分类型): 联系方式类型
   - 取值: "unknown", "telephone", "cellular"
10. **day** (数值型): 最后联系日期（月份中的第几天）
11. **month** (分类型): 最后联系月份
    - 取值: "jan", "feb", "mar", ..., "nov", "dec"
12. **duration** (数值型): 最后联系时长（秒）

#### 其他属性
13. **campaign** (数值型): 本次活动中对该客户的联系次数（包括最后一次联系）
14. **pdays** (数值型): 上次营销活动后经过的天数（-1表示之前未联系过）
15. **previous** (数值型): 本次活动前对该客户的联系次数
16. **poutcome** (分类型): 上次营销活动的结果
    - 取值: "unknown", "other", "failure", "success"

### 输出变量（目标变量）
17. **y** (二元型): 客户是否订阅了定期存款
    - 取值: "yes", "no"

### 数据质量
- **缺失值**: 无缺失值
- **实例数量**: 45,211条（完整数据集）或4,521条（子集）