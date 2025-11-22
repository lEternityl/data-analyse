# 1. 数据加载与预处理 (Data Loading & Cleaning)
# 加载必要的包
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, corrplot, caret, glmnet, randomForest, ggplot2, factoextra, stringr)
# 读取数据
loan_data <- read.csv("train.csv", stringsAsFactors = FALSE)
# --- 数据预处理关键步骤 ---
# 原始数据包含日期字符串和ID，直接放入模型会报错，需进行清洗
clean_data <- loan_data %>%
  # 1. 移除对预测无用的ID类特征和复杂的日期特征
  select(-id, -issueDate, -earliesCreditLine, -policyCode) %>%

  # 2. 处理 employmentLength (例如 "10+ years" -> 10)
  mutate(
    employmentLength = str_extract(employmentLength, "\\d+"), # 提取数字
    employmentLength = as.numeric(ifelse(is.na(employmentLength), 0, employmentLength))
  ) %>%

  # 3. 转换类别变量为因子 (Factor)
  mutate(
    grade = as.factor(grade),
    subGrade = as.factor(subGrade),
    term = as.factor(term),
    isDefault = as.numeric(isDefault) # 保持数值用于相关性分析，分类模型时再转Factor
  ) %>%

  # 4. 处理缺失值 (简单删除，也可选择中位数填充)
  na.omit()

print("数据预处理完成，概览：")
str(clean_data)


# 2. 相关分析 - 识别关键变量 (Correlation Analysis)
# 计算相关系数矩阵 (只选数值型列)
numeric_vars <- clean_data %>% select_if(is.numeric)
cor_matrix <- cor(numeric_vars, use = "complete.obs")

# 可视化相关系数矩阵
# 调整字体大小以适应较多变量
corrplot(cor_matrix, method = "color", type = "upper",
         order = "hclust", tl.cex = 0.6, tl.col = "black")

# 找出与目标变量 (isDefault) 最相关的特征
target_cor <- cor_matrix[, "isDefault"]
target_cor <- target_cor[!is.na(target_cor)] # 移除NA
# 按绝对值排序，排除 isDefault 本身
target_cor <- sort(abs(target_cor[names(target_cor) != "isDefault"]), decreasing = TRUE)

print("与违约(isDefault)相关性最高的前10个变量：")
print(head(target_cor, 10))

# 3. 方差分析/组间差异 - 分类变量的影响 (ANOVA)
# 分析不同信用等级 (grade) 的平均违约率是否不同
# 虽然 isDefault 是0/1，但用 ANOVA 可以看不同组的均值差异（即违约率差异）
anova_grade <- aov(isDefault ~ grade, data = clean_data)
summary(anova_grade)

# 可视化：不同等级的违约率
ggplot(clean_data, aes(x = grade, y = isDefault, fill = grade)) +
  stat_summary(fun = "mean", geom = "bar") +
  labs(title = "不同信用等级(Grade)的违约率", y = "平均违约率") +
  theme_minimal()


# 分析工作年限 (转为因子进行分析) 对违约率的影响
anova_emp <- aov(isDefault ~ as.factor(employmentLength), data = clean_data)
summary(anova_emp)

# 4. 逻辑回归 - 变量重要性排序 (Logistic Regression)
# 准备数据：逻辑回归需要二分类的目标变量
logit_data <- clean_data
logit_data$isDefault <- as.factor(logit_data$isDefault)

# 建立模型 (这里只选部分核心变量演示，全变量可能运行较慢或不收敛)
# 移除了 subGrade 防止与 grade 多重共线性
logit_model <- glm(isDefault ~ . - subGrade,
                   data = logit_data,
                   family = binomial)

# 变量重要性排序 (基于 z-value 绝对值)
library(broom)
logit_summary <- tidy(logit_model)
logit_summary <- logit_summary %>%
  mutate(importance = abs(statistic)) %>%
  arrange(desc(importance))

print("逻辑回归中显著性最高的前10个特征：")
print(head(logit_summary, 10))

# 5. 随机森林 - 变量重要性 (Random Forest)
set.seed(123)

sample_index <- sample(1:nrow(clean_data), min(5000, nrow(clean_data)))
rf_data <- clean_data[sample_index, ]
rf_data$isDefault <- as.factor(rf_data$isDefault) # 必须转为因子进行分类

rf_model <- randomForest(isDefault ~ . - subGrade, # 同样排除 subGrade
                         data = rf_data,
                         importance = TRUE,
                         ntree = 300)

# 提取变量重要性
var_importance <- importance(rf_model)
var_importance_df <- as.data.frame(var_importance)
var_importance_df$variable <- rownames(var_importance_df)

# 按 MeanDecreaseGini (基尼指数下降量) 排序
var_importance_df <- var_importance_df[order(var_importance_df$MeanDecreaseGini,
                                             decreasing = TRUE),]

print("随机森林变量重要性 (Top 10)：")
print(head(var_importance_df[, c("variable", "MeanDecreaseGini")], 10))

# 可视化
varImpPlot(rf_model, main = "随机森林变量重要性排序", n.var = 15)

# 6. 主成分分析 - 数据降维 (PCA)
# 仅对数值型变量进行 PCA
pca_data <- clean_data %>% select_if(is.numeric) %>% select(-isDefault)

# 执行 PCA
pca_result <- prcomp(pca_data, scale. = TRUE)

# 可视化主成分解释的方差
fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 50), main = "主成分方差解释率")

# 变量在主成分上的贡献图
fviz_pca_var(pca_result,
             col.var = "contrib",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE,
             select.var = list(contrib = 15)) # 只显示贡献最大的15个变量

# 7. 最终变量选择
# 综合逻辑回归和随机森林的结果，选择最重要的变量
top_variables <- var_importance_df$variable[1:10]

print("综合模型建议保留的特征：")
print(top_variables)