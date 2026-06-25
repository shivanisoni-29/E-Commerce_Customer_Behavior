
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="E-Commerce Customer Behavior", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load("models/random_forest_baseline.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    return model, encoder

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/cleaned_ecommerce.csv")

    duration_map = {
        "Very Short": 0,
        "Short": 1,
        "Long": 2,
        "Very Long": 3
    }

    df["session_duration_bucket"] = df["session_duration_bucket"].map(duration_map)

    return df

model, label_encoder = load_model()
df = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to",["Home","Dashboard","Prediction","Model Performance"])

if page=="Home":
    st.title("E-Commerce Customer Behavior Analysis")
    st.write("Dataset shape:",df.shape)
    st.dataframe(df.head(10))
    st.write(df.describe())

elif page == "Dashboard":
    st.title("📊 E-Commerce Behavior Analytics Dashboard")

    import plotly.express as px

    df_num = df.select_dtypes(include=np.number)

    # -----------------------------
    # SIDEBAR FILTERS (REAL DASHBOARD FEEL)
    # -----------------------------
    st.sidebar.markdown("### ℹ️ Feature Guide")
    st.sidebar.markdown(""" - **Device Type**: Device used by the customer *(0 = Desktop, 1 = Mobile, 2 = Tablet)*  
- **User Type**: Type of user *(0 = New, 1 = Returning)*
""")
    st.sidebar.subheader("Filters")

    if "device_type" in df.columns:
        device_filter = st.sidebar.multiselect(
            "Device Type",
            df["device_type"].unique(),
            default=df["device_type"].unique()
        )
        df_filtered = df[df["device_type"].isin(device_filter)]
    else:
        df_filtered = df.copy()

    # -----------------------------
    # KPI CARDS (INTERVIEW IMPORTANT)
    # -----------------------------
    st.subheader("Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Users", len(df_filtered))
    col2.metric("Avg Pages Viewed", round(df_filtered["pages_viewed"].mean(), 2))
    col3.metric("Avg Time (sec)", round(df_filtered["time_on_site_sec"].mean(), 2))
    col4.metric("Avg Discount %", round(df_filtered["discount_percent"].mean(), 2))

    # -----------------------------
    # DISTRIBUTION ANALYSIS
    # -----------------------------
    st.subheader("Feature Distributions")

    feature = st.selectbox("Select Feature", df_num.columns)

    fig = px.histogram(
        df_filtered,
        x=feature,
        nbins=30,
        color_discrete_sequence=["#636EFA"]
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # RELATIONSHIP ANALYSIS
    # -----------------------------
    st.subheader("Relationship Analysis")

    col1, col2 = st.columns(2)

    with col1:
        x_axis = st.selectbox("X-axis", df_num.columns, index=0)
    with col2:
        y_axis = st.selectbox("Y-axis", df_num.columns, index=1)

    fig = px.scatter(
        df_filtered,
        x=x_axis,
        y=y_axis,
        color="device_type" if "device_type" in df.columns else None,
        opacity=0.6
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # CORRELATION HEATMAP (UPGRADED)
    # -----------------------------
    st.subheader("Feature Correlation Heatmap")

    corr = df_num.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # SEGMENT INSIGHT (WHAT INTERVIEWS LIKE)
    # -----------------------------
    st.subheader("Behavior by Device Type")

    if "device_type" in df.columns:
        seg = df_filtered.groupby("device_type")["pages_viewed"].mean().reset_index()

        fig = px.bar(
            seg,
            x="device_type",
            y="pages_viewed",
            color="device_type"
        )

        st.plotly_chart(fig, use_container_width=True)

elif page=="Prediction":
    st.title("Purchase Prediction")

    feature_cols=model.feature_names_in_.tolist()

    sample=df.iloc[0].copy()

    device=st.selectbox("Device Type",["Desktop","Mobile","Tablet"])
    user=st.selectbox("User Type",["New","Returning"])
    unit_price=st.number_input("Unit Price",0.0,100000.0,100.0)
    quantity=st.number_input("Quantity",1,100,1)
    discount=st.slider("Discount (%)",0.0,100.0,0.0)
    pages=st.number_input("Pages Viewed",1,100,5)
    time_site=st.number_input("Time on Site (sec)",0.0,5000.0,300.0)
    cart=st.number_input("Added to Cart",0,100,1)

    device_map={"Desktop":0,"Mobile":1,"Tablet":2}
    user_map={"New":0,"Returning":1}

    sample["device_type"]=device_map[device]
    sample["user_type"]=user_map[user]
    sample["unit_price"]=unit_price
    sample["quantity"]=quantity
    sample["discount_percent"]=discount
    sample["pages_viewed"]=pages
    sample["time_on_site_sec"]=time_site
    sample["added_to_cart"]=cart

    X=pd.DataFrame([sample])[feature_cols]
    st.write(X)
    print(X.dtypes)
    print(X.iloc[0])

    if st.button("Predict"):
        pred=model.predict(X)[0]
        prob=model.predict_proba(X)[0]
        st.success("Purchase" if pred==1 else "No Purchase")
        st.metric("Confidence",f"{max(prob)*100:.2f}%")

elif page == "Model Performance":

    st.title("Model Performance")

    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        "Baseline": [0.7108, 0.3839, 0.4755, 0.4248, 0.7569],
        "Optimized": [0.6060, 0.3562, 0.9341, 0.5157, 0.7595]
    })

    st.dataframe(metrics)

    fig, ax = plt.subplots()
    x = np.arange(len(metrics))

    ax.bar(x - 0.2, metrics["Baseline"], 0.4, label="Baseline")
    ax.bar(x + 0.2, metrics["Optimized"], 0.4, label="Optimized")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics["Metric"])
    ax.legend()

    st.pyplot(fig)
