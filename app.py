import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# Page Configuration for Professional App
st.set_page_config(
    page_title="Student Expense Analyzer Pro", 
    page_icon="📊",
    layout="wide"
)

# Custom Premium Styling (Dashboard Look)
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #ff4b4b; color: white; font-weight: bold; height: 3em; }
    .css-1r6g72q { background-color: #161b22; }
    div[data-testid="metric-container"] {
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Model Load karna
model_path = "student_expense_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("⚠️ AI Core Machine Learning Model (.pkl file) missing!")
    st.stop()

# --- PRIVATE SESSION STATE FOR DATA PRIVACY ---
if "private_expenses" not in st.session_state:
    st.session_state.private_expenses = pd.DataFrame(
        columns=["Date", "Breakfast", "Dinner", "Fruits", "Books_Study", "Shopping", "Total"]
    )

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://icons8.com", width=80)
st.sidebar.title("App Navigation")
page = st.sidebar.radio("Go to:", ["📝 Expense Input & Prediction", "📊 Advanced Analytics Dashboard"])

st.sidebar.markdown("---")
st.sidebar.info("🔒 **Data Privacy Active:** Aapka data sirf aapke device par save ho raha hai.")

# =========================================================
# PAGE 1: INPUT & PREDICTION
# =========================================================
if page == "📝 Expense Input & Prediction":
    st.title("🎓 Lahore Student Expense Tracker Pro")
    st.markdown("Apne rozmarrah ke kharche add karain aur AI prediction check karain.")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📋 Add Daily Details")
        date = st.date_input("Select Date")
        
        bf_option = st.radio("Kya aaj Breakfast kiya?", ("Yes", "No"))
        bf_price = 0
        bf_skipped = 1
        if bf_option == "Yes":
            bf_price = st.number_input("Breakfast Cost (PKR):", min_value=0, value=120, step=10)
            bf_skipped = 0

        dinner_price = st.number_input("Dinner Cost (PKR):", min_value=0, value=250, step=10)
        fruits_price = st.number_input("Fruits / Snacks Cost (PKR):", min_value=0, value=0, step=10)
        books_price = st.number_input("Books / Study Material (PKR):", min_value=0, value=0, step=50)
        shopping_price = st.number_input("Shopping / Personal Care (PKR):", min_value=0, value=0, step=100)
        
        save_btn = st.button("🚀 Save & Predict Total")

    with col2:
        st.subheader("🤖 AI Analytics Engine")
        if save_btn:
            features = np.array([[bf_skipped, bf_price, dinner_price, fruits_price, books_price, shopping_price]])
            predicted_total = model.predict(features)[0]
            actual_total = bf_price + dinner_price + fruits_price + books_price + shopping_price
            
            # Save data
            new_row = pd.DataFrame([{
                "Date": str(date), "Breakfast": bf_price, "Dinner": dinner_price, 
                "Fruits": fruits_price, "Books_Study": books_price, "Shopping": shopping_price, "Total": actual_total
            }])
            st.session_state.private_expenses = pd.concat([st.session_state.private_expenses, new_row], ignore_index=True)
            
            # Status Metrics
            st.success("🎉 Kharcha successfully log kar diya gaya hai!")
            m1, m2 = st.columns(2)
            m1.metric(label="Actual Total Spent", value=f"{actual_total} PKR")
            m2.metric(label="AI Predicted Target", value=f"{predicted_total:.2f} PKR")
            
            # Advice box based on spending
            if actual_total > 500:
                st.warning("⚠️ Today's budget is a bit high! Try to optimize expenses tomorrow.")
            else:
                st.info("💡 Great job! Your spending is well within the healthy student budget limit.")
        else:
            st.info("Form fill kar ke 'Save & Predict' dabayein taake AI insights yahan show hon.")

# =========================================================
# PAGE 2: ADVANCED ANALYTICS DASHBOARD
# =========================================================
else:
    st.title("📊 Financial Analytics Dashboard")
    st.markdown("Aapke monthly kharchon ka tafseeli graph aur analysis.")
    st.markdown("---")
    
    df_db = st.session_state.private_expenses
    
    if not df_db.empty:
        # Top Row Metrics
        total_monthly = df_db["Total"].sum()
        avg_daily = df_db["Total"].mean()
        
        c1, c2, c3 = st.columns(3)
        c1.metric(label="💵 Is Mahine Ka Kul Karcha", value=f"{total_monthly} PKR")
        c2.metric(label="📅 Total Days Logged", value=f"{len(df_db)} Days")
        c3.metric(label="📈 Average Daily Expense", value=f"{avg_daily:.2f} PKR")
        
        st.markdown("---")
        
        # Graphs Section
        g1, g2 = st.columns([1.5, 1])
        
        with g1:
            st.subheader("📉 Daily Spending Trend")
            fig_line = px.line(df_db, x="Date", y="Total", title="Spending Over Time", markers=True, template="plotly_dark")
            st.plotly_chart(fig_line, use_container_width=True)
            
        with g2:
            st.subheader("🍰 Expense Distribution")
            breakdown = {
                "Category": ["Breakfast", "Dinner", "Fruits", "Books/Study", "Shopping"],
                "Amount": [
                    df_db["Breakfast"].sum(), df_db["Dinner"].sum(), 
                    df_db["Fruits"].sum(), df_db["Books_Study"].sum(), df_db["Shopping"].sum()
                ]
            }
            df_breakdown = pd.DataFrame(breakdown)
            fig_pie = px.pie(df_breakdown, values="Amount", names="Category", title="Where is your money going?", hole=0.4, template="plotly_dark")
            st.plotly_chart(fig_pie, use_container_width=True)
            
        st.markdown("---")
        st.subheader("📋 Detailed Ledger Data")
        st.dataframe(df_db, use_container_width=True)
        
        # Professional Export Feature
        csv_data = df_db.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Data as CSV (Excel)",
            data=csv_data,
            file_name="student_monthly_expenses.csv",
            mime="text/csv"
        )
    else:
        st.warning("📊 Dashboard Analytics dekhne ke liye pehle Input page par ja kar kuch kharche add karain.")
