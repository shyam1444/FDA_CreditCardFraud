import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from utils import load_data, get_sample, prepare_features
from models import run_kmeans, train_xgboost, compute_shap_values, run_rl_simulation
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Fraud Analytics System", layout="wide", page_icon="🛡️")

# Custom CSS for a dynamic, premium look
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 95%;
    }
    .stRadio label {
        font-weight: bold;
        color: #f1f2f6;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 6px;
        transition: 0.3s;
        width: 100%;
        border: none;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
        color: white;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Data Loading -----------------
with st.spinner("Loading core dataset..."):
    df = load_data(r'C:/Users/ShyamVenkatraman/Desktop/FDA/creditcard_balanced.csv')

if df.empty:
    st.error("Please ensure 'creditcard_balanced.csv' is available at the specified path.")
    st.stop()

# Create Sidebar Category
st.sidebar.markdown("<h2 style='color: #ff4b4b;'>🛡️ Control Panel</h2>", unsafe_allow_html=True)
category = st.sidebar.radio("", [
    "Descriptive Analytics", 
    "Diagnostic Analytics", 
    "Predictive Analytics", 
    "Prescriptive Analytics"
])

st.sidebar.markdown("---")
st.sidebar.caption("High-performance fraud detection & optimization system powered by XGBoost, SHAP, and Q-Learning.")

# Helper header
st.markdown("<h1 style='text-align: center; margin-bottom: 20px;'>💳 Enterprise Fraud Analytics Command Center</h1>", unsafe_allow_html=True)

# ----------------- DESCRIPTIVE ANALYTICS -----------------
if category == "Descriptive Analytics":
    st.markdown("### 📊 Descriptive Analytics\nUnderstand historic fraud rates, loss amounts, and typical transaction profiles.")
    
    col1, col2, col3 = st.columns(3)
    fraud_count = df['Class'].sum()
    total_count = len(df)
    fraud_rate = fraud_count / total_count * 100
    total_loss = df[df['Class'] == 1]['Amount'].sum()
    
    col1.metric("Total Transactions Logged", f"{total_count:,}")
    col2.metric("Confirmed Fraud Cases", f"{fraud_count:,} ({fraud_rate:.2f}%)")
    col3.metric("Total Historic Financial Loss", f"${total_loss:,.2f}")
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("Transaction Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=df[df['Amount'] < 500], x='Amount', hue='Class', bins=50, kde=True, ax=ax, palette=['#1f77b4', '#ff4b4b'])
        ax.set_title("Transaction Amounts (<$500) by Class", color='white')
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        st.pyplot(fig, use_container_width=True)
        
    with col_chart2:
        st.subheader("K-Means Segments")
        st.markdown("Segmenting normal behavior to spot outliers via `k=3` grouping.")
        if st.button("Run K-Means Clustering Array"):
            with st.spinner("Clustering..."):
                n_normal = len(df[df['Class'] == 0])
                subset = df[df['Class'] == 0].sample(n=min(10000, n_normal), random_state=42)
                X_subset = subset.drop(['Class', 'Time'], axis=1, errors='ignore')
                
                kmeans = run_kmeans(X_subset, n_clusters=3)
                subset['Cluster'] = kmeans.labels_
                
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                sns.scatterplot(x='V1', y='V2', hue='Cluster', data=subset, palette='viridis', alpha=0.6, ax=ax2)
                fig2.patch.set_facecolor('#0e1117')
                ax2.set_facecolor('#0e1117')
                ax2.tick_params(colors='white')
                ax2.xaxis.label.set_color('white')
                ax2.yaxis.label.set_color('white')
                
                # Make legend font white
                legend = ax2.get_legend()
                if legend:
                    for text in legend.get_texts():
                        text.set_color("white")

                st.pyplot(fig2, use_container_width=True)
                
                largest_cluster = subset['Cluster'].value_counts().idxmax()
                largest_pct = (subset['Cluster'].value_counts().max() / len(subset)) * 100
                st.success(f"**Insight:** **{largest_pct:.1f}%** of normal transactions fall into Cluster {largest_cluster}. This implies that the vast majority of legitimate customer behavior follows a highly distinct pattern, making deviations extremely suspicious.")

# ----------------- DIAGNOSTIC ANALYTICS -----------------
if category == "Diagnostic Analytics":
    st.markdown("### 🔍 Diagnostic Analytics\nFind root causes of fraud spikes and false positives using SHAP feature explainability.")
    
    col_btn, col_empty = st.columns([1, 2])
    with col_btn:
        run_diag = st.button("Train Diagnostic Core & Generate SHAP Map")
        
    if run_diag:
        with st.spinner("Extracting parameters and formatting tree map..."):
            fraud = df[df['Class'] == 1]
            n_normal = len(df[df['Class'] == 0])
            normal = df[df['Class'] == 0].sample(n=min(len(fraud)*2, n_normal), random_state=42)
            balanced_df = pd.concat([fraud, normal])
            
            X, y = prepare_features(balanced_df)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = train_xgboost(X_train, y_train)
            
            X_shap_sample = X_test.sample(n=min(300, len(X_test)), random_state=42)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap_sample)
            
            col_shap, col_text = st.columns([2, 1])
            with col_shap:
                st.subheader("SHAP Feature Hierarchy Plot")
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                fig3.patch.set_facecolor('#0e1117')
                ax3.set_facecolor('#0e1117')
                ax3.tick_params(colors='white')
                shap.summary_plot(shap_values, X_shap_sample, show=False)
                # Ensure labels are white for dark mode compatibility manually isn't trivial for shap plots
                st.pyplot(fig3, use_container_width=True)
                
            with col_text:
                vals = np.abs(shap_values).mean(0)
                top_feature_idx = np.argmax(vals)
                top_feature_name = X_shap_sample.columns[top_feature_idx]
                st.info(f"**Key Driver Identified:**\n\nThe SHAP explainability model marks **{top_feature_name}** as the absolute strongest root cause of fraud outcomes in your dataset.\n\nAnalysts should aggressively screen variances in the `{top_feature_name}` sector.")

# ----------------- PREDICTIVE ANALYTICS -----------------
if category == "Predictive Analytics":
    st.markdown("### 🔮 Predictive Analytics\nTrain an XGBoost topology strictly calibrated for precision imbalanced event-catching.")
    
    if 'xgb_model' not in st.session_state:
        st.session_state['xgb_model'] = None
        
    col_btn, col_empty = st.columns([1, 2])
    with col_btn:
        run_pred = st.button("Initialize & Train Predictive Engine (XGBoost)")
        
    if run_pred:
        with st.spinner("Engineering weights & bootstrapping engine..."):
            df_sample = get_sample(df, frac=0.1)
            X_pred, y_pred = prepare_features(df_sample)
            X_train, X_test, y_train, y_test = train_test_split(X_pred, y_pred, test_size=0.2, stratify=y_pred, random_state=42)
            
            model = train_xgboost(X_train, y_train)
            st.session_state['xgb_model'] = model
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.success("Predictive Engine Online.")
            
    if st.session_state['xgb_model'] is not None:
        model = st.session_state['xgb_model']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred_class = (y_prob > 0.5).astype(int)
        
        col_cm, col_report, col_pr = st.columns([1, 1.5, 1.5])
        
        with col_cm:
            st.subheader("Confusion Map")
            cm = confusion_matrix(y_test, y_pred_class)
            fig_cm, ax_cm = plt.subplots(figsize=(3, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax_cm, cbar=False)
            fig_cm.patch.set_facecolor('#0e1117')
            ax_cm.title.set_color('white')
            ax_cm.xaxis.label.set_color('white')
            ax_cm.yaxis.label.set_color('white')
            ax_cm.tick_params(colors='white')
            st.pyplot(fig_cm, use_container_width=True)
            
        with col_report:
            st.subheader("Report Diagnostics")
            report = classification_report(y_test, y_pred_class, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='Greens'), use_container_width=True)
            
            recall_score = report.get('1', {}).get('recall', 0) * 100
            st.info(f"The engine flags **{recall_score:.1f}%** of all true fraud occurrences securely.")
            
        with col_pr:
            st.subheader("Precision-Recall Falloff")
            precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
            fig_pr, ax_pr = plt.subplots(figsize=(4, 3))
            ax_pr.plot(recall, precision, marker='.', color='#ff4b4b', label='XGBoost')
            fig_pr.patch.set_facecolor('#0e1117')
            ax_pr.set_facecolor('#0e1117')
            ax_pr.xaxis.label.set_color('white')
            ax_pr.yaxis.label.set_color('white')
            ax_pr.tick_params(colors='white')
            st.pyplot(fig_pr, use_container_width=True)

# ----------------- PRESCRIPTIVE ANALYTICS -----------------
if category == "Prescriptive Analytics":
    st.markdown("### 📈 Prescriptive Analytics\nDetermine definitive business action boundaries (Approve vs. Deny) via Q-Learning Agents.")
    
    if st.session_state.get('xgb_model') is None:
        st.warning("Please train the Predictive Engine locally in the Predictive Tab first to source probabilities.")
    else:
        col_btn, _ = st.columns([1, 2])
        with col_btn:
            run_rl = st.button("Trigger Q-Learning Optimizer")
            
        if run_rl:
            with st.spinner("Processing simulated actions via reinforcement grid..."):
                y_prob = st.session_state['xgb_model'].predict_proba(st.session_state['X_test'])[:, 1]
                y_true = st.session_state['y_test'].values
                
                q_table, rewards = run_rl_simulation(y_true, y_prob)
                
                policy_data = []
                for state, actions in q_table.items():
                    best_action = max(actions, key=actions.get)
                    policy_data.append({
                        "Prob Bin": state,
                        "Approve Q-Val": round(actions["Approve"], 2),
                        "Review Q-Val": round(actions["Review"], 2),
                        "Deny Q-Val": round(actions["Deny"], 2),
                        "Best Action": best_action
                    })
                policy_df = pd.DataFrame(policy_data)
                policy_df['SortKey'] = policy_df['Prob Bin'].apply(lambda x: int(x.split('_')[1]))
                policy_df = policy_df.sort_values('SortKey').drop(columns=['SortKey']).reset_index(drop=True)
                
                col_tbl, col_rew = st.columns([1, 1])
                
                with col_tbl:
                    st.subheader("Learned Action Policy")
                    st.dataframe(policy_df.style.apply(lambda x: ['background: lightgreen; color: black; font-weight: bold' if v == x['Best Action'] else '' for v in x], axis=1), use_container_width=True)
                    
                with col_rew:
                    st.subheader("Q-Agent Reward Tracking")
                    fig_rew, ax_rew = plt.subplots(figsize=(5, 3))
                    ax_rew.plot(rewards, color='#1f77b4')
                    fig_rew.patch.set_facecolor('#0e1117')
                    ax_rew.set_facecolor('#0e1117')
                    ax_rew.xaxis.label.set_color('white')
                    ax_rew.yaxis.label.set_color('white')
                    ax_rew.tick_params(colors='white')
                    st.pyplot(fig_rew, use_container_width=True)
                    
                intervention_bins = policy_df[policy_df['Best Action'].isin(['Deny', 'Review'])]
                if not intervention_bins.empty:
                    lowest = intervention_bins.iloc[0]['Prob Bin']
                    prob_threshold = int(lowest.split('_')[1]) * 10
                    st.success(f"**Action Directed:** The Q-Learning algorithm mathematically concludes that any transaction possessing a baseline fraud probability of **{prob_threshold}% or higher** severely eclipses potential revenue and MUST trigger a standard 'Deny' or 'Review' gate. Transactions under this threshold maximize yield and are safe to Auto-Approve.")
