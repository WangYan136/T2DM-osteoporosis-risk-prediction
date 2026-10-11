import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# 1. 页面设置
# ============================================================

st.set_page_config(
    page_title="老年T2DM患者骨质疏松风险预测工具",
    page_icon="🦴",
    layout="centered"
)


# ============================================================
# 2. 模型基本信息
# ============================================================

MODEL_THRESHOLD = 0.432105

FEATURES = [
    "BMI",
    "FBG_mmol_L",
    "PTH",
    "Ca_mmol_L",
    "VitD25OH",
    "TC_mmol_L"
]


# ============================================================
# 3. 自动寻找最终冻结模型
# ============================================================

current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent

possible_model_paths = [
    project_dir / "results" / "final_XGBoost_6variables.pkl",
    project_dir / "final_XGBoost_6variables.pkl"
]

model_path = None

for p in possible_model_paths:
    if p.exists():
        model_path = p
        break

if model_path is None:
    st.error(
        "未找到 final_XGBoost_6variables.pkl。\n\n"
        "请确认模型文件位于项目根目录或 results 文件夹中。"
    )
    st.stop()


# ============================================================
# 4. 加载最终冻结模型
# ============================================================

@st.cache_resource
def load_model(path):
    return joblib.load(path)


try:
    model = load_model(model_path)
except Exception as e:
    st.error("模型加载失败。")
    st.exception(e)
    st.stop()


# ============================================================
# 5. 页面标题
# ============================================================

st.title("老年2型糖尿病患者骨质疏松风险预测工具")

st.markdown(
    """
    本工具基于 **XGBoost机器学习模型** 构建，
    通过6项临床指标对老年2型糖尿病患者并发骨质疏松的风险进行辅助评估。
    """
)

st.divider()


# ============================================================
# 6. 输入变量
# ============================================================

st.subheader("请输入患者临床指标")

col1, col2 = st.columns(2)


with col1:

    bmi = st.number_input(
        "BMI（kg/m²）",
        min_value=10.0,
        max_value=50.0,
        value=24.50,
        step=0.10,
        format="%.2f"
    )

    pth = st.number_input(
        "甲状旁腺素 PTH（pg/mL）",
        min_value=0.0,
        max_value=300.0,
        value=51.00,
        step=0.10,
        format="%.2f"
    )

    vitd = st.number_input(
        "25-羟维生素D 25(OH)D（ng/mL）",
        min_value=0.0,
        max_value=100.0,
        value=17.50,
        step=0.10,
        format="%.2f"
    )


with col2:

    fbg = st.number_input(
        "空腹血糖 FBG（mmol/L）",
        min_value=1.0,
        max_value=40.0,
        value=7.90,
        step=0.10,
        format="%.2f"
    )

    ca = st.number_input(
        "血钙 Ca（mmol/L）",
        min_value=1.0,
        max_value=4.0,
        value=2.33,
        step=0.01,
        format="%.2f"
    )

    tc = st.number_input(
        "总胆固醇 TC（mmol/L）",
        min_value=1.0,
        max_value=15.0,
        value=4.84,
        step=0.01,
        format="%.2f"
    )


# ============================================================
# 7. 风险预测
# ============================================================

st.divider()

if st.button(
    "计算骨质疏松风险",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # 必须严格保持最终模型训练时的变量名称与顺序
    # --------------------------------------------------------

    patient_data = pd.DataFrame(
        [[
            bmi,
            fbg,
            pth,
            ca,
            vitd,
            tc
        ]],
        columns=FEATURES
    )


    # --------------------------------------------------------
    # 预测概率
    # --------------------------------------------------------

    probability = model.predict_proba(
        patient_data
    )[0, 1]


    # --------------------------------------------------------
    # 输出
    # --------------------------------------------------------

    st.subheader("预测结果")

    st.metric(
        label="模型预测概率",
        value=f"{probability * 100:.1f}%"
    )


    if probability >= MODEL_THRESHOLD:

        st.warning(
            "模型判定：骨质疏松风险预测阳性"
        )

        st.write(
            f"该患者模型预测概率高于预设分类阈值 "
            f"{MODEL_THRESHOLD:.3f}。"
        )

    else:

        st.success(
            "模型判定：骨质疏松风险预测阴性"
        )

        st.write(
            f"该患者模型预测概率低于预设分类阈值 "
            f"{MODEL_THRESHOLD:.3f}。"
        )


    # --------------------------------------------------------
    # 显示输入信息
    # --------------------------------------------------------

    with st.expander("查看本次输入指标"):

        display_df = pd.DataFrame({
            "指标": [
                "BMI",
                "FBG",
                "PTH",
                "Ca",
                "25(OH)D",
                "TC"
            ],

            "数值": [
                bmi,
                fbg,
                pth,
                ca,
                vitd,
                tc
            ],

            "单位": [
                "kg/m²",
                "mmol/L",
                "pg/mL",
                "mmol/L",
                "ng/mL",
                "mmol/L"
            ]
        })

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True
        )


# ============================================================
# 8. 模型说明
# ============================================================

st.divider()

st.subheader("模型说明")

st.markdown(
    f"""
- **预测模型：** XGBoost
- **预测变量：** BMI、FBG、PTH、Ca、25(OH)D、TC
- **分类阈值：** {MODEL_THRESHOLD:.6f}
- **研究对象：** 老年2型糖尿病患者
- **预测结局：** 是否并发骨质疏松
"""
)


# ============================================================
# 9. 使用说明
# ============================================================

st.subheader("使用说明")

st.markdown(
    """
请按照患者实际临床检测结果输入各项指标，
点击“计算骨质疏松风险”后即可获得模型预测结果。

模型预测概率仅反映本研究XGBoost模型计算得到的预测输出，
风险预测阳性不等同于临床确诊。
"""
)


# ============================================================
# 10. 免责声明
# ============================================================

st.info(
    """
本工具基于单中心回顾性临床数据开发，目前仅用于科研及风险辅助评估。
模型尚需进一步开展多中心外部验证，其预测结果不能替代DXA检查、
临床诊断、医生判断或其他专业医疗决策。
"""
)