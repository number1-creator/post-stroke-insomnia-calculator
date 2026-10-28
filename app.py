# -*- coding: utf-8 -*-
"""
卒中后失眠风险在线计算器（Streamlit 部署版，学术化界面）
启动时现场训练随机森林模型，输入 9 个临床特征，输出失眠风险概率与风险分层。
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

st.set_page_config(
    page_title="Post-Stroke Insomnia Risk Calculator",
    page_icon="😴",
    layout="wide",
)

# 最终 9 个特征
FINAL_FEATURES_RAW = ["BI", "mRS", "CD_RISC", "SSRS", "IL_6", "EPQ_E", "EPQ_N", "TNF__", "IL_1_"]

FEATURE_LABELS = {
    "BI": "Barthel Index (0–100, higher = more independent)",
    "mRS": "modified Rankin Scale (0–6, higher = greater disability)",
    "CD_RISC": "Connor–Davidson Resilience Scale (higher = more resilient)",
    "SSRS": "Social Support Rating Scale (higher = more support)",
    "IL_6": "Interleukin-6 (pg/mL)",
    "EPQ_E": "Eysenck Personality Questionnaire — Extraversion",
    "EPQ_N": "Eysenck Personality Questionnaire — Neuroticism",
    "TNF__": "Tumour necrosis factor-α (pg/mL)",
    "IL_1_": "Interleukin-1β (pg/mL)",
}


@st.cache_resource
def train_model():
    """读取训练数据，现场训练随机森林模型（SMOTE + RF）。"""
    train = pd.read_excel("train_data.xlsx")
    train = train.drop(columns=["Num"], errors="ignore")
    train["insomnia"] = train["insomnia"].map({"Yes": 1, "No": 0}).astype(int)

    X = train[FINAL_FEATURES_RAW].copy()
    y = train["insomnia"]
    for c in FINAL_FEATURES_RAW:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=7, min_samples_split=10,
        min_samples_leaf=5, max_features="sqrt", random_state=42, n_jobs=1,
    )
    pipe = ImbPipeline(steps=[
        ("smote", SMOTE(random_state=42, k_neighbors=5)),
        ("model", rf),
    ])
    pipe.fit(X, y)
    return pipe


pipe = train_model()

# ===================== 页面 =====================

# 顶部：标题 + 论文引用信息
st.title("😴 Post-Stroke Insomnia Risk Calculator")
st.markdown(
    "**A machine-learning model for predicting post-stroke insomnia in patients with stroke**"
)
st.caption(
    "Random forest classifier · external-validation AUROC = 0.874 (95% CI 0.839–0.907) · "
    "9 predictors"
)

# 两列布局：左侧输入，右侧说明
col_input, col_sidebar = st.columns([1.2, 1])

with col_input:
    st.markdown("### Patient characteristics")
    st.markdown("Enter the nine clinical variables below:")

    c1, c2 = st.columns(2)
    inputs = {}
    for i, f in enumerate(FINAL_FEATURES_RAW):
        label = FEATURE_LABELS.get(f, f)
        with (c1 if i % 2 == 0 else c2):
            inputs[f] = st.number_input(label, value=0.0, step=0.1, format="%.2f")

    st.markdown("---")
    if st.button("🔍 Predict risk", type="primary", use_container_width=True):
        X = np.array([[inputs[f] for f in FINAL_FEATURES_RAW]])
        proba = float(pipe.predict_proba(X)[0][1])
        risk_pct = proba * 100

        if risk_pct < 20:
            level, color = "Low risk", "#2e7d32"
        elif risk_pct < 50:
            level, color = "Intermediate risk", "#f9a825"
        else:
            level, color = "High risk", "#c62828"

        st.markdown("### Prediction")
        st.markdown(
            f"**Predicted probability of post-stroke insomnia:** "
            f"<span style='font-size:1.6em;font-weight:bold;color:{color}'>{risk_pct:.1f}%</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**Risk category:** <span style='font-size:1.2em;font-weight:bold;color:{color}'>{level}</span>",
            unsafe_allow_html=True,
        )

with col_sidebar:
    st.markdown("### Model overview")
    st.markdown(
        """
- **Model**: Random forest (300 trees)
- **External-validation AUROC**: 0.874 (95% CI 0.839–0.907)
- **Predictors**: 9 clinical variables (neurological, psychological, inflammatory)
- **Risk strata**: Low (<20%), Intermediate (20–50%), High (>50%)
        """
    )

    st.markdown("### Receiver operating characteristic curves")
    st.image("assets/roc_external.png", caption="External validation cohort", use_container_width=True)

    st.markdown("### Disclaimer")
    st.caption(
        "This tool is provided for research and demonstration purposes only and is not "
        "a substitute for clinical judgement. The model was developed on a retrospective "
        "cohort and should be externally validated before clinical use."
    )

    st.markdown("### Reference")
    st.caption(
        "Post-stroke insomnia prediction using machine learning: development and external "
        "validation of a random forest model (manuscript in preparation)."
    )
