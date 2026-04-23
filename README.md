# 用户增量分析与流失预测

> 基于电信用户流失数据集的端到端用户增长分析项目，覆盖 EDA、RFM 用户分层、生命周期分析、LTV 预测、多模型流失预测与 ROI 量化框架，并完整迁移至游戏用户增长场景。

## 项目背景

本项目来源于 **Statistical Machine Learning** 课程，目标是构建一套可复用的用户增长分析框架，通过对电信用户行为数据的深度分析，提炼流失信号与留存策略。

后期出于个人学习目的，额外添加分析内容，将分析逻辑迁移至游戏用户运营场景（仅作参考）。

---

## 数据集

| 项目 | 内容 |
|------|------|
| 来源 | [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| 文件名 | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| 记录数 | 7,043 条 |
| 字段数 | 21 个 |
| 目标变量 | `Churn`（流失 Yes / 未流失 No，流失率 26.54%） |

> **注意**：数据集文件不含于本仓库，请从 Kaggle 链接下载后放置于项目根目录。

---

## 分析框架（五阶段）

```
阶段一  数据探索与清洗（EDA）
  └── 缺失值处理 → 流失分布 → 特征相关性 → 关键流失信号挖掘

阶段二  用户分层分析
  ├── 任务 2.1  RFM 模型（8类用户群，流失率 1.4%~45.5%）
  ├── 任务 2.2  生命周期分段（入网预警期/入网期/成长期/成熟期/衰退期）
  └── 任务 2.3  LTV 预测（LTV = 平均月消费 × 预期生命周期）

阶段三  流失预测模型
  ├── 任务 3.1  特征工程（衍生特征 + One-Hot 编码，32 维）
  ├── 任务 3.2  训练/测试集划分（70/30，stratify 分层抽样）
  ├── 任务 3.3  逻辑回归（基准模型）
  ├── 任务 3.4  随机森林（非线性对比）
  └── 任务 3.5  XGBoost / LightGBM（四模型横向对比）

阶段四  业务洞察与策略建议
  ├── 任务 4.1  Actionable Insights（5级风险用户画像）
  ├── 任务 4.2  差异化挽留策略设计
  └── ROI 矩阵（CAC / 挽留率 / LTV 三因素，优先级排序）

阶段五  游戏场景迁移
  ├── 任务 5.1  电信→游戏字段映射表（dim_user / fact_log 表设计）
  ├── 任务 5.2  游戏流失预测框架
  └── 任务 5.3  A/B 测试方案设计（游戏流失用户召回）
```

---

## 核心结论

### 流失信号

| 特征 | 发现 |
|------|------|
| 合同类型 | 月付用户流失率最高，两年付用户最低 |
| 在网时长 | 0–6 个月新用户流失风险最高（入网期） |
| 网络服务 | Fiber Optic 用户流失率高达 42% |
| 增值服务 | 服务使用广度与留存率呈强正相关 |
| 月消费 | 流失用户月均消费（$74.44）显著高于留存用户（$61.27） |

### 模型性能对比

| 模型 | AUC-ROC | Recall（流失） | Precision（流失） | F1 |
|------|---------|--------------|-----------------|-----|
| **逻辑回归** | **0.8443** | **0.7968** | 0.5109 | 0.6226 |
| 随机森林 | 0.8390 | 0.7291 | 0.5587 | 0.6326 |
| XGBoost | 0.8355 | 0.7433 | 0.5353 | 0.6224 |
| LightGBM | 0.8338 | 0.7736 | 0.5306 | 0.6294 |

> **关键发现**：逻辑回归在 AUC 和 Recall 上均优于或持平集成模型，说明该数据集的决策边界接近线性；在可解释性要求高的场景下，逻辑回归是最优选择。

### ROI 矩阵（触达成本 CAC = 20 元）

| 用户分群 | 平均 LTV | 挽留率 20% | 挽留率 30% | 挽留率 40% |
|---------|---------|-----------|-----------|-----------|
| 成熟期 | $2,000 | ROI = 19x | ROI = 29x | ROI = **39x** |
| 成长期 | $1,200 | ROI = 11x | ROI = 17x | ROI = 23x |
| 衰退期 | $400 | ROI = 3x | ROI = 5x | ROI = 7x |

> **策略建议**：预算有限时，优先触达成熟期与成长期用户；衰退期用户"全量挽留"性价比较低。

---

## 技术栈

| 类别 | 工具 |
|------|------|
| 数据处理 | Python · Pandas · NumPy |
| 可视化 | Matplotlib · Seaborn |
| 机器学习 | Scikit-learn · XGBoost · LightGBM |
| 统计分析 | SciPy（A/B 测试功效分析） |
| 开发环境 | Jupyter Notebook |

---

## 项目结构

```
.
├── 用户增量分析.ipynb          # 主分析 Notebook（45 个 Cell）
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # 数据集（需自行下载）
└── README.md
```

---

## 快速开始

### 1. 克隆仓库

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. 安装依赖

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm scipy jupyter
```

### 3. 下载数据集

前往 [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) 下载 `WA_Fn-UseC_-Telco-Customer-Churn.csv`，放入项目根目录。

### 4. 运行 Notebook

```bash
jupyter notebook 用户增量分析.ipynb
```

按顺序执行全部 Cell 即可复现所有分析结果。

---

## 游戏场景迁移说明

本项目的核心分析逻辑可直接迁移至游戏用户增长场景：

| 电信字段 | 游戏对应字段 | 说明 |
|---------|------------|------|
| `tenure` | `user_lifetime_days` | 用户生命周期长度 |
| `MonthlyCharges` | `avg_monthly_pay` | 月付费强度 |
| `TotalCharges` | `total_pay_amount` | 累计付费价值 |
| `Contract` | `payment_type` | 月卡 / 季卡 / 年卡 / 买断 |
| `Churn` | `is_churned` | 连续 N 天未登录定义为流失 |
| `PhoneService` | `core_feature_used` | 核心功能使用情况 |

游戏流失预测所需数据表：`dim_user`（用户维度）、`fact_log`（行为日志）、`fact_pay`（付费流水）。

---

## A/B 测试设计（附录）

针对游戏流失用户召回场景，项目内置完整 A/B 测试设计器（`ABTestDesigner`），支持：

- 样本量计算（功效分析，α = 0.05，power = 0.8）
- MDE 反推（当前设计可检测最小提升：2.65%）
- 多重检验校正（Bonferroni，5 次分层检验时 α 校正为 0.01）
- 实验周期预估（每日 100 人流失用户 → 需 22 天）

**实验设计摘要**

- 假设：个性化推送召回率（8%）> 通用推送（5%）
- 每组所需样本：1,059 人，总计 2,118 人
- 主指标：7 日召回率；次要指标：召回后 7 日留存率、首次付费率

---

<img width="1583" height="990" alt="image" src="https://github.com/user-attachments/assets/1f948bb2-78dc-4543-ab35-8cf7608fc746" />

<img width="1287" height="490" alt="image" src="https://github.com/user-attachments/assets/11564f73-d4f6-46ec-b754-f5ffdea1ccce" />

<img width="1589" height="1190" alt="image" src="https://github.com/user-attachments/assets/a8cebdcd-5ea5-4781-973a-f4965d5248fd" />

<img width="1589" height="1189" alt="image" src="https://github.com/user-attachments/assets/64c674a3-a575-4603-a91a-d647eae183b2" />

## 逻辑回归
<img width="1780" height="489" alt="image" src="https://github.com/user-attachments/assets/e32d18e0-dd57-43d4-8a1e-2830aa835f8a" />

## 随机森林
<img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/b73c6e16-f5f6-463e-a683-3dd1816e0595" />

## XGBoost和LightGBM
<img width="1790" height="490" alt="image" src="https://github.com/user-attachments/assets/cf010eae-d40e-4cd3-9fcc-35d618c54ad5" />
