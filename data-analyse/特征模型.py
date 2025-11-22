import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

# 设置绘图风格和字体（解决中文乱码）
plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # Windows/Mac兼容
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 1. 环境设置与数据准备
# ==============================================================================
print("正在加载数据...")
# 使用原始数据路径
file_path = r"/Users/zhuyi/PycharmProjects/pythonProject/data-analyse/train.csv"

# 仅读取 R 分析确定的重要列 + 目标列，节省内存
# R分析结果 Top 10: interestRate, dti, installment, postCode, employmentTitle,
#                  revolUtil, annualIncome, revolBal, loanAmnt, grade
cols_to_use = [
    'interestRate', 'dti', 'installment', 'postCode', 'employmentTitle',
    'revolUtil', 'annualIncome', 'revolBal', 'loanAmnt', 'grade',
    'employmentLength',  # 额外保留用于演示清洗，虽然不在Top10但常用于微调
    'isDefault'  # 目标变量
]

try:
    df = pd.read_csv(file_path, usecols=cols_to_use)
except FileNotFoundError:
    print(f"错误：找不到文件 {file_path}")



# --- 数据预处理 (复刻 R 脚本逻辑) ---

# 1. 处理 employmentLength (正则提取数字)
def clean_emp_length(x):
    if pd.isna(x):
        return 0
    # 提取字符串中的第一个数字
    match = re.search(r'\d+', str(x))
    if match:
        return float(match.group())
    return 0


df['employmentLength'] = df['employmentLength'].apply(clean_emp_length)

# 2. 处理 Categorical 变量 (Grade)
# 信用等级是有序的 (Ordinal)，使用映射比 One-Hot 更好
grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
df['grade'] = df['grade'].map(grade_map)

# 3. 处理缺失值
print(f"清洗前数据量: {len(df)}")
df.dropna(inplace=True)
print(f"清洗后数据量: {len(df)}")

# ==============================================================================
# 2. 数据集划分与标准化
# ==============================================================================
X = df.drop('isDefault', axis=1)
y = df['isDefault']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 标准化 (这对逻辑回归很重要，对树模型不是必须但无害)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 将标准化后的数据转回 DataFrame 以便后续使用 (保留列名)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# ==============================================================================
# 3. 模型训练与评估
# ==============================================================================

# 定义模型
# 注意：对于大数据集，HistGradientBoostingClassifier 比 GradientBoostingClassifier 快非常多
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, max_depth=10),  # 限制深度防止过拟合
    'Gradient Boosting (Hist)': HistGradientBoostingClassifier(random_state=42)
}

results = {}
best_auc = 0
best_model_name = ""
best_model_obj = None

print("\n开始模型训练...")

for name, model in models.items():
    print(f"正在训练: {name}...")

    # 逻辑回归使用标准化数据，树模型使用原始数据(通常效果更好且保留物理意义)
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    # 计算指标
    auc = roc_auc_score(y_test, y_prob)
    acc = (y_pred == y_test).mean()

    results[name] = {'model': model, 'auc': auc, 'prob': y_prob}

    print(f"--- {name} 结果 ---")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print("-" * 30)

    # 记录最佳模型
    if auc > best_auc:
        best_auc = auc
        best_model_name = name
        best_model_obj = model

print(f"\n最佳模型是: {best_model_name} (AUC: {best_auc:.4f})")

# ==============================================================================
# 4. 结果可视化
# ==============================================================================

# --- A. ROC 曲线对比 ---
plt.figure(figsize=(10, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['prob'])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {res["auc"]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('模型 ROC 曲线对比')
plt.legend()
plt.grid(True)
plt.savefig('roc_comparison.png')
plt.show()

# --- B. 特征重要性 (针对最佳的树模型) ---
if best_model_name != 'Logistic Regression':
    # 获取特征重要性

    if hasattr(best_model_obj, 'feature_importances_'):
        importances = best_model_obj.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=X.columns[indices], palette='viridis')
        plt.title(f'特征重要性 ({best_model_name})')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('feature_importance_py.png')
        plt.show()

        # 打印数值
        feat_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
        print("\n特征重要性排名:")
        print(feat_imp_df.sort_values(by='Importance', ascending=False))
    else:
        # 针对 HistGradientBoosting 使用 permutation_importance
        from sklearn.inspection import permutation_importance

        print("计算 Permutation Importance")
        result = permutation_importance(best_model_obj, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
        sorted_idx = result.importances_mean.argsort()[::-1]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=result.importances_mean[sorted_idx], y=X.columns[sorted_idx], palette='viridis')
        plt.title(f'特征重要性 (Permutation - {best_model_name})')
        plt.tight_layout()
        plt.savefig('feature_importance_py.png')
        plt.show()

# ==============================================================================
# 5. 模型保存
# ==============================================================================
print("\n正在保存最佳模型和预处理工具...")
joblib.dump(best_model_obj, 'best_credit_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print(f"模型已保存为: best_credit_model.pkl")
print("分析结束！")