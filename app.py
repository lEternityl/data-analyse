import streamlit as st
import pandas as pd
import joblib
import requests
import json

# 页面配置
st.set_page_config(
    page_title="信用风险评估系统",
    page_icon="🏦",
    layout="wide")

# ==============================================================================
# 核心配置
# ==============================================================================
# 已更新为用户提供的绝对路径，并修正了斜杠方向
MODEL_PATH = 'best_credit_model.pkl'
SCALER_PATH = 'scaler.pkl'
PREDICTION_THRESHOLD = 0.25  # 沿用分析中最佳的业务阈值

# DeepSeek API配置
# 此api在本次课程作业后会删除，使用时请重新申请！！！！
DEEPSEEK_API_KEY = "sk-91f27b4f466d44f9ad375dfc6f93e76e"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# R 分析确定的 11 个关键特征（必须与训练时的顺序和名称完全一致）
FEATURE_ORDER = [
    'interestRate', 'dti', 'installment', 'postCode', 'employmentTitle',
    'revolUtil', 'annualIncome', 'revolBal', 'loanAmnt', 'grade',
    'employmentLength'
]

# Grade 映射（复刻训练脚本逻辑）
GRADE_MAP = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}


# ==============================================================================
# 加载模型和标准化器
# ==============================================================================
@st.cache_resource
def load_model_and_scaler():
    """加载模型和标准化器"""
    scaler_path_resolved = SCALER_PATH

    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(scaler_path_resolved)
        return model, scaler
    except FileNotFoundError:
        st.error(
            f"错误：找不到模型文件 ({MODEL_PATH}) 或标准化器文件 ({scaler_path_resolved})。请确保文件已存在且路径正确。")
        st.stop()
    except Exception as e:
        st.error(f"加载模型时出错: {e}")
        st.stop()


model, scaler = load_model_and_scaler()
TRAIN_FEATURES = list(getattr(model, 'feature_names_in_', [])) or FEATURE_ORDER


# ==============================================================================
# AI 分析函数
# ==============================================================================
def get_ai_analysis(credit_data, prediction, probability):
    """调用DeepSeek API获取信用分析"""

    # 检查 Key 是否已替换 
    if DEEPSEEK_API_KEY == "YOUR_ACTUAL_DEEPSEEK_API_KEY_HERE":
        return " DeepSeek API Key 仍是占位符，请将其替换为您的真实密钥。"

    # 转换输入数据为可读的中文格式
    data_display = credit_data.T.rename(
        index={'interestRate': '贷款利率', 'dti': '负债收入比', 'installment': '分期金额',
               'postCode': '邮编代码', 'employmentTitle': '职位名称代码', 'revolUtil': '循环额度利用率',
               'annualIncome': '年收入', 'revolBal': '循环余额', 'loanAmnt': '贷款金额',
               'grade': '信用等级代码', 'employmentLength': '工作年限'}
    ).to_dict()

    prompt = f"""
    基于以下客户信息（部分特征已编码）和信用评估结果，提供专业的信用风险分析：

    **客户信息**:
    {json.dumps(data_display, ensure_ascii=False, indent=2)}

    **评估结果**:
    - 信用风险等级: {'高风险 (建议拒绝)' if prediction == 1 else '低风险 (建议通过)'}
    - 风险概率: {probability:.2%}
    - 预测阈值: {PREDICTION_THRESHOLD}

    请从以下角度提供分析（注意：等级和职位名称是数值编码）：
    1. **主要风险因素分析** (根据输入数据中高风险项)
    2. **信用改善建议**
    3. **信审措施建议** (针对该客户的风险等级)
    4. **总结**

    用专业但易懂的中文回复，面向信贷审批人员。
    """

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system",
             "content": "你是一个专业的信用风险分析师，擅长用数据驱动的方法评估信用风险，并能识别数据中的潜在风险点。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"API调用失败 (状态码: {response.status_code})。请检查 Key 是否有效或服务是否可用。"
    except requests.exceptions.Timeout:
        return "获取AI分析时出错: 请求超时。"
    except Exception as e:
        return f"获取AI分析时出错: {str(e)}"


# ==============================================================================
# 应用界面
# ==============================================================================
st.title("智能信用风险评估系统")
st.markdown("---")

col_main, col_sidebar = st.columns([4, 2])

with col_sidebar:
    st.header("客户信息输入")

    with st.form("credit_form"):
        st.subheader("贷款与信用信息")

        # 1. 贷款金额
        loanAmnt = st.number_input("贷款金额 (loanAmnt)", min_value=1000.0, value=20000.0, step=1000.0)

        # 2. 信用等级
        grade_text = st.selectbox("信用等级 (grade)", options=list(GRADE_MAP.keys()), index=1)

        # 3. 贷款利率
        interestRate = st.slider("贷款利率 (%) (interestRate)", min_value=5.0, max_value=25.0, value=15.0,
                                 step=0.1) / 100

        # 4. 分期金额
        installment = st.number_input("分期金额 (installment)", min_value=10.0, value=650.0, step=10.0)

        # 5. 循环额度利用率
        revolUtil = st.slider("循环额度利用率 (%) (revolUtil)", min_value=0.0, max_value=100.0, value=50.0,
                              step=0.1) / 100

        # 6. 循环余额
        revolBal = st.number_input("循环余额 (revolBal)", min_value=0.0, value=15000.0, step=500.0)

        st.subheader("个人与财务信息")

        # 7. 年收入
        annualIncome = st.number_input("年收入 (annualIncome)", min_value=10000.0, value=60000.0, step=5000.0)

        # 8. 负债收入比
        dti = st.slider("负债收入比 (dti)", min_value=0.0, max_value=50.0, value=20.0, step=0.1)

        # 9. 工作年限
        employmentLength = st.number_input("工作年限 (employmentLength)", min_value=0, max_value=30, value=5)

        # 10. 邮编
        postCode = st.number_input("邮编代码 (postCode)", min_value=0.0, value=3000.0, step=1.0)

        # 11. 职位名称
        employmentTitle = st.number_input("职位名称代码 (employmentTitle)", min_value=0.0, value=1000.0, step=1.0)

        submitted = st.form_submit_button("开始评估")

# 主内容区域
with col_main:
    st.header("风险评估结果")

    if submitted:
        # 1. 数据预处理
        # 将 grade 文本映射为数值
        grade_numeric = GRADE_MAP.get(grade_text)

        # 2. 准备输入 DataFrame，并严格按照 FEATURE_ORDER 排序
        input_data = pd.DataFrame({
            'interestRate': [interestRate],
            'dti': [dti],
            'installment': [installment],
            'postCode': [postCode],
            'employmentTitle': [employmentTitle],
            'revolUtil': [revolUtil],
            'annualIncome': [annualIncome],
            'revolBal': [revolBal],
            'loanAmnt': [loanAmnt],
            'grade': [grade_numeric],
            'employmentLength': [employmentLength]
        })

        input_data = input_data[TRAIN_FEATURES]

        # 3. 预测 (HistGradientBoostingClassifier 使用原始/非标准化数据)
        try:
            # 预测概率 (取类别1的概率)
            probability = model.predict_proba(input_data)[0, 1]

            # 根据自定义阈值生成预测类别
            prediction = 1 if probability >= PREDICTION_THRESHOLD else 0

            # 显示结果
            if prediction == 1:
                st.error(f"预测结果：高风险客户 (拒绝建议)")
                st.metric("风险概率", f"{probability:.2%}", delta=f"阈值: {PREDICTION_THRESHOLD:.2%}")
            else:
                st.success(f"🟢 预测结果：低风险客户 (通过建议)")
                st.metric("风险概率", f"{probability:.2%}", delta=f"阈值: {PREDICTION_THRESHOLD:.2%}")

            # 风险等级指示器
            st.markdown("**风险指示器**")
            st.progress(float(probability))

            # 关键指标显示
            st.subheader("输入关键指标概览")
            col1_1, col1_2, col1_3, col1_4 = st.columns(4)

            with col1_1:
                st.metric("信用等级", grade_text)
            with col1_2:
                st.metric("贷款利率", f"{interestRate:.1%}")
            with col1_3:
                st.metric("负债收入比", f"{dti:.1f}")
            with col1_4:
                st.metric("年收入", f"{annualIncome / 10000:.1f}万")

            st.markdown("---")
            st.header("AI深度分析")

            with st.spinner("AI正在基于评估结果进行专业分析..."):
                ai_analysis = get_ai_analysis(input_data.iloc[0], prediction, probability)

            st.info(ai_analysis)

        except Exception as e:
            st.error(f"预测过程中出错，请检查输入格式或模型文件。错误信息: {str(e)}")

# 底部统计信息
st.markdown("---")
st.subheader("系统信息与阈值")

col3, col4 = st.columns(2)
with col3:
    st.metric("当前模型", model.__class__.__name__)
with col4:
    st.metric("评估阈值", f"{PREDICTION_THRESHOLD:.2f}")
