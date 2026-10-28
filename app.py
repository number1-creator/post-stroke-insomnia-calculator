# -*- coding: utf-8 -*-
"""
卒中后失眠风险在线计算器（Streamlit 部署版）
加载随附的随机森林模型，输入 9 个临床特征，输出失眠风险概率与风险分层。
"""
import joblib
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Post-Stroke Insomnia Risk Calculator",
    page_icon="😴",
    layout="centered",
)

# 模型文件与本脚本同目录
@st.cache_resource
def load_model():
    pipe = joblib.load("rf_model.pkl")
    meta = joblib.load("features.pkl")
    return pipe, meta["features"]


pipe, FEATURES = load_model()

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

st.title("Post-Stroke Insomnia Risk Calculator")
st.caption(
    "A machine-learning model (random forest, external-validation AUROC = 0.874) "
    "to estimate the probability of post-stroke insomnia from nine clinical variables."
)

st.markdown("### Enter clinical variables")

# 分两列布局
col1, col2 = st.columns(2)
inputs = {}
for i, f in enumerate(FEATURES):
    label = FEATURE_LABELS.get(f, f)
    # 交替放到左右两列
    with (col1 if i % 2 == 0 else col2):
        inputs[f] = st.number_input(label, value=0.0, step=0.1, format="%.2f")

st.markdown("---")

if st.button("Predict", type="primary", use_container_width=True):
    X = np.array([[inputs[f] for f in FEATURES]])
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
        f"<span style='font-size:1.4em;font-weight:bold;color:{color}'>{risk_pct:.1f}%</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**Risk category:** <span style='font-weight:bold;color:{color}'>{level}</span>",
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption(
    "This tool is provided for research and demonstration purposes only and is not "
    "a substitute for clinical judgement."
)
