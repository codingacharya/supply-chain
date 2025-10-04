# streamlit_supply_chain.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Dynamic Global Supply Chain Optimizer", layout="wide")
st.title("Dynamic Global Supply Chain Optimizer")

# --- Load Data ---
st.subheader("Upload Supply Chain Dataset (CSV)")
uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset Loaded Successfully!")
    st.dataframe(df.head(10))

    # --- Data Stats ---
    st.subheader("Dataset Statistics")
    st.write(df.describe())

    # --- Risk Distribution Visualization ---
    st.subheader("Risk Level Distribution")
    risk_counts = df['Risk_Level'].value_counts()
    st.bar_chart(risk_counts)

    # --- Inventory vs Forecasted Demand ---
    st.subheader("Inventory vs Forecasted Demand")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Inventory', y='Forecasted_Demand', hue='Risk_Level', s=100, ax=ax)
    plt.xlabel("Inventory")
    plt.ylabel("Forecasted Demand")
    st.pyplot(fig)

    # --- Lead Time Distribution ---
    st.subheader("Lead Time Distribution")
    st.bar_chart(df['Lead_Time_Days'])

    # --- Correlation Heatmap ---
    st.subheader("Feature Correlation")
    corr = df.select_dtypes(include=np.number).corr()
    fig2, ax2 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    # --- Prediction Model ---
    st.subheader("Demand Prediction Model")
    st.write("Predict Forecasted Demand based on Inventory, Lead Time, Unit Cost, Transportation Cost")

    # Prepare features
    df_model = df.copy()
    df_model = pd.get_dummies(df_model, columns=['Warehouse', 'Supplier', 'Shipping_Route', 'Risk_Level'], drop_first=True)
    X = df_model.drop(columns=['Product_ID','Product_Name','Forecasted_Demand'])
    y = df_model['Forecasted_Demand']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("### Model Metrics")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

    # --- Feature Importance ---
    st.subheader("Feature Importance")
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    st.dataframe(feature_importances.head(10))

    fig3, ax3 = plt.subplots(figsize=(8,4))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(10), ax=ax3)
    st.pyplot(fig3)

    # --- Predict New Input ---
    st.subheader("Predict Forecasted Demand for New Data")
    inventory = st.number_input("Inventory", min_value=0, value=100)
    lead_time = st.number_input("Lead Time (days)", min_value=1, value=7)
    unit_cost = st.number_input("Unit Cost USD", min_value=1, value=50)
    transport_cost = st.number_input("Transportation Cost USD", min_value=1, value=10)

    warehouse_options = df['Warehouse'].unique().tolist()
    warehouse_sel = st.selectbox("Warehouse", warehouse_options)
    supplier_options = df['Supplier'].unique().tolist()
    supplier_sel = st.selectbox("Supplier", supplier_options)
    shipping_options = df['Shipping_Route'].unique().tolist()
    shipping_sel = st.selectbox("Shipping Route", shipping_options)
    risk_options = df['Risk_Level'].unique().tolist()
    risk_sel = st.selectbox("Risk Level", risk_options)

    input_df = pd.DataFrame({
        'Inventory':[inventory],
        'Lead_Time_Days':[lead_time],
        'Unit_Cost_USD':[unit_cost],
        'Transportation_Cost_USD':[transport_cost]
    })

    # Add one-hot encoded columns
    for col in X.columns:
        if col.startswith('Warehouse_'):
            input_df[col] = 1 if col == f'Warehouse_{warehouse_sel}' else 0
        if col.startswith('Supplier_'):
            input_df[col] = 1 if col == f'Supplier_{supplier_sel}' else 0
        if col.startswith('Shipping_Route_'):
            input_df[col] = 1 if col == f'Shipping_Route_{shipping_sel}' else 0
        if col.startswith('Risk_Level_'):
            input_df[col] = 1 if col == f'Risk_Level_{risk_sel}' else 0
        if col not in input_df.columns:
            input_df[col] = 0

    pred_demand = model.predict(input_df)[0]
    st.success(f"Predicted Forecasted Demand: {pred_demand:.2f} units")

    # --- Optimization Module ---
    st.subheader("Supply Chain Optimization Suggestions")

    df['Predicted_Demand'] = model.predict(df_model.drop(columns=['Product_ID','Product_Name','Forecasted_Demand']))
    df['Stock_Surplus'] = df['Inventory'] - df['Predicted_Demand']

    # Inventory Reallocation
    st.write("### Inventory Reallocation Suggestions")
    reallocation_suggestions = []
    for product in df['Product_Name'].unique():
        product_df = df[df['Product_Name'] == product]
        surplus_warehouses = product_df[product_df['Stock_Surplus'] > 0].copy()
        deficit_warehouses = product_df[product_df['Stock_Surplus'] < 0].copy()
        for _, deficit_row in deficit_warehouses.iterrows():
            for _, surplus_row in surplus_warehouses.iterrows():
                move_qty = min(surplus_row['Stock_Surplus'], abs(deficit_row['Stock_Surplus']))
                if move_qty > 0:
                    reallocation_suggestions.append({
                        'Product': product,
                        'From_Warehouse': surplus_row['Warehouse'],
                        'To_Warehouse': deficit_row['Warehouse'],
                        'Quantity': move_qty
                    })
                    surplus_warehouses.loc[surplus_warehouses.index[0], 'Stock_Surplus'] -= move_qty
                    deficit_warehouses.loc[deficit_warehouses.index[0], 'Stock_Surplus'] += move_qty

    if reallocation_suggestions:
        st.dataframe(pd.DataFrame(reallocation_suggestions))
    else:
        st.write("No reallocation needed. Inventory levels are balanced.")

    # Shipping Rerouting
    st.write("### Shipping Rerouting Suggestions")
    reroute_suggestions = []
    for _, row in df.iterrows():
        if row['Lead_Time_Days'] > 10 or row['Risk_Level'] == 'High':
            reroute_suggestions.append({
                'Product': row['Product_Name'],
                'Current_Route': row['Shipping_Route'],
                'Suggestion': 'Consider alternative faster or safer route'
            })
    if reroute_suggestions:
        st.dataframe(pd.DataFrame(reroute_suggestions))
    else:
        st.write("All routes are optimal.")

    # Alternative Suppliers
    st.write("### Alternative Supplier Suggestions")
    alt_supplier_suggestions = []
    for _, row in df.iterrows():
        if row['Risk_Level'] == 'High':
            alt_suppliers = df[(df['Product_Name'] == row['Product_Name']) & (df['Supplier'] != row['Supplier'])]
            if not alt_suppliers.empty:
                alt_supplier_suggestions.append({
                    'Product': row['Product_Name'],
                    'Current_Supplier': row['Supplier'],
                    'Suggested_Supplier': alt_suppliers.iloc[0]['Supplier']
                })
    if alt_supplier_suggestions:
        st.dataframe(pd.DataFrame(alt_supplier_suggestions))
    else:
        st.write("All suppliers are reliable.")
