import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="Lahore Student Expense Tracker", 
    page_icon="🎓",
    layout="centered"
)

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #FF4B4B; color: white; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 Lahore Student Monthly Expense Tracker")

# Mobile Instructions Help Box
with st.expander("📱 Mobile Phone Par As A Downloadable App Kaise Chalayein?"):
    st.markdown("""
    ### 📱 Mobile App Kaise Banayein?
    Agar aapko direct link kholne par setting mein option nahi mil raha, toh yeh steps follow karain:
    
    **Android (Google Chrome):**
    1. WhatsApp se yeh link copy karain aur direct **Google Chrome App** khol kar wahan upar paste karain.
    2. Jab website khul jaye, toh Chrome ke upar daen taraf **3 dots (...)** par click karain.
    3. List mein neeche dekhein, wahan **`Add to Home screen`** ya **`Install app`** likha hoga. Us par click kar dein!
    
    **iPhone (Safari Browser):**
    1. Link ko direct **Safari Browser** mein open karain.
    2. Neeche center mein **Share button** (square dabba aur upar arrow) par click karain.
    3. Thoda neeche scroll karke **`Add to Home Screen`** par click kar dein!
    """)

# Model Load karna
model_path = "student_expense_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("Machine Learning Model (.pkl file) nahi mili!")
    st.stop()

# Local database file (CSV) setup taake data save rahe
db_file = "my_expenses.csv"
if not os.path.exists(db_file):
    df_db = pd.DataFrame(columns=["Date", "Breakfast", "Dinner", "Fruits", "Books_Study", "Shopping", "Total"])
    df_db.to_csv(db_file, index=False)

# --- USER INPUT SECTION ---
st.header("📝 Aaj Ka Kharcha Add Karain")

date = st.date_input("Tareekh select karain")

# Breakfast Option
bf_option = st.radio("Kya aaj Breakfast kiya?", ("Yes", "No"))
bf_price = 0
bf_skipped = 1
if bf_option == "Yes":
    bf_price = st.number_input("Breakfast ki keemat (PKR):", min_value=0, value=100, step=10)
    bf_skipped = 0

# Other Portions
dinner_price = st.number_input("Dinner ki keemat (PKR):", min_value=0, value=200, step=10)
fruits_price = st.number_input("Fruits ki keemat (PKR):", min_value=0, value=0, step=10)
books_price = st.number_input("Books / Pen / Marker ka karcha (PKR):", min_value=0, value=0, step=50)
shopping_price = st.number_input("Shopping / Face wash / Other (PKR):", min_value=0, value=0, step=100)

# --- SAVE & PREDICT ---
if st.button("Save Daily Expense"):
    features = np.array([[bf_skipped, bf_price, dinner_price, fruits_price, books_price, shopping_price]])
    predicted_total = model.predict(features)
    actual_total = bf_price + dinner_price + fruits_price + books_price + shopping_price
    
    new_row = [date, bf_price, dinner_price, fruits_price, books_price, shopping_price, actual_total]
    df_db = pd.read_csv(db_file)
    df_db.loc[len(df_db)] = new_row
    df_db.to_csv(db_file, index=False)
    
    st.success(f"🎉 Kharcha save ho gaya! Actual Total: {actual_total} PKR")
    st.info(f"🤖 ML Model Prediction (Estimated Total): {predicted_total:.2f} PKR")

# --- MONTHLY ANALYSIS ---
st.markdown("---")
st.header("📊 Aapka Monthly Hisab Kitab")

df_db = pd.read_csv(db_file)
if not df_db.empty:
    st.dataframe(df_db)
    
    total_monthly = df_db["Total"].sum()
    st.metric(label="💵 Is Mahine Ka Kul Karcha", value=f"{total_monthly} PKR")
    
    st.subheader("Portion Wise Breakdown")
    breakdown = {
        "Portion": ["Breakfast", "Dinner", "Fruits", "Books/Study", "Shopping"],
        "Total Spent (PKR)": [
            df_db["Breakfast"].sum(),
            df_db["Dinner"].sum(),
            df_db["Fruits"].sum(),
            df_db["Books_Study"].sum(),
            df_db["Shopping"].sum()
        ]
    }
    st.table(pd.DataFrame(breakdown))
else:
    st.warning("Abhi tak koi data add nahi kiya gaya.")
