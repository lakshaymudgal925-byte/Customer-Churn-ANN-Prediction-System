import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from scikeras.wrappers import KerasClassifier
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Churn ANN Prediction System", 
    layout='wide', 
    page_icon='🏦',
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
    }
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1E90FF;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton>button {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #45a049, #3d8b40);
    }
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1976d2;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.pipeline = None
    st.session_state.df = None
    st.session_state.X_test = None
    st.session_state.y_test = None
    st.session_state.y_pred = None

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Choose an option:",
    ['🏠 Home', '🤖 ANN Model', '📊 Data Visualizations', '🔮 Make Prediction', '📈 Statistics']
)

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Churn_Modelling.csv')
        return df
    except FileNotFoundError:
        st.error("Churn_Modelling.csv not found. Please ensure the file is in the same directory.")
        return None
    
df  = load_data()

if df is not None:
    st.session_state.df = df
    
@st.cache_resource
def build_and_train_model(df):
    # Prepare features and target
    X = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
    y = df['Exited']
    
    # Identify Categorical & Numerical Columns
    categorical_cols = ['Geography', 'Gender']
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    
    # Preprocessing using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ]
    )
    
    # Create ANN Model Function
    def build_ann():
        model = Sequential()
        model.add(Dense(16, activation='relu', input_dim=11))
        model.add(Dropout(0.2))
        model.add(Dense(9, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    ann = KerasClassifier(model=build_ann, epochs=50, batch_size=32, verbose=0)
    
    # Create Full Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('model', ann)
    ])
    
    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    with st.spinner("Training ANN model... This may take a moment..."):
        pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    return pipeline, X_test, y_test, y_pred, y_pred_proba, X

#Home page
if option == '🏠 Home':
    st.header('🏦 Customer Churn ANN Prediction System')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🧠 Welcome to the Customer Churn Prediction System!
        
        This application uses an **Artificial Neural Network (ANN)** integrated with a Scikit-learn Pipeline 
        to predict whether a customer is likely to leave the bank.
        
        ---
        
        ### 🎯 Problem It Solves:
        
        Customer churn is one of the biggest challenges in the banking and subscription industries.
        Acquiring new customers is significantly more expensive than retaining existing ones.
        
        This system helps businesses:
        - 🚨 Identify high-risk customers
        - 💰 Reduce revenue loss
        - 📈 Improve customer retention strategies
        - ⚡ Make proactive business decisions
        """)
    
    with col2:
        st.markdown("""
        ### ⚡ Quick Stats
        """)
        if df is not None:
            st.metric("Total Customers", f"{len(df):,}")
            st.metric("Churn Rate", f"{(df['Exited'].mean()*100):.1f}%")
            st.metric("Features", f"{len(df.columns)-4}")
    
    # Model Architecture
    st.markdown("---")
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        ### 🧠 Model Architecture:
        
        - **Input Layer**: 11 features
        - **Hidden Layer 1**: 16 neurons (ReLU) + Dropout 20%
        - **Hidden Layer 2**: 9 neurons (ReLU) + Dropout 20%
        - **Output Layer**: 1 neuron (Sigmoid)
        - **Optimizer**: Adam (lr=0.001)
        - **Loss Function**: Binary Crossentropy
        """)
    
    with col4:
        st.markdown("""
        ### 📊 Prediction Factors:
        
        The model makes predictions based on:
        - Credit Score, Age, Balance
        - Tenure, Products, Activity
        - Geography, Gender
        - Has Credit Card, Estimated Salary
        """)
    
    # Display sample data
    st.markdown("---")
    st.subheader("📋 Sample Data Structure")
    if df is not None:
        display_cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
                       'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
                       'EstimatedSalary', 'Exited']
        st.dataframe(df[display_cols].head(10), use_container_width=True)

# ANN Model page
elif option == '🤖 ANN Model':
    st.header('🤖 ANN Model Training',)
    
    if df is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("info-box")
            st.subheader("Model Configuration")
            st.write("**Architecture:** Feed-forward Neural Network")
            st.write("**Hidden Layers:** 2 layers (16 → 9 neurons)")
            st.write("**Activation:** ReLU (hidden), Sigmoid (output)")
            st.write("**Dropout:** 20% after each hidden layer")
            st.write("**Optimizer:** Adam (learning rate: 0.001)")
            st.write("**Loss Function:** Binary Crossentropy")
            st.write("**Epochs:** 50")
            st.write("**Batch Size:** 32")
            st.markdown('</d>', unsafe_allow_html=True)
            
            if st.button("🚀 Train ANN Model", use_container_width=True):
                pipeline, X_test, y_test, y_pred, y_pred_proba, X = build_and_train_model(df)
                
                st.session_state.model_trained = True
                st.session_state.pipeline = pipeline
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred
                st.session_state.y_pred_proba = y_pred_proba
                st.session_state.feature_names = X.columns.tolist()
                
                st.success("✅ Model trained successfully!")
        
        with col2:
            if st.session_state.model_trained:
                st.subheader("📊 Model Performance")
                
                accuracy = accuracy_score(st.session_state.y_test, st.session_state.y_pred)
                tn, fp, fn, tp = confusion_matrix(st.session_state.y_test, st.session_state.y_pred).ravel()
                
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("Accuracy", f"{accuracy:.2%}")
                    st.metric("Precision", f"{tp/(tp+fp):.2%}" if (tp+fp)>0 else "N/A")
                with metrics_col2:
                    st.metric("Sensitivity (Recall)", f"{tp/(tp+fn):.2%}" if (tp+fn)>0 else "N/A")
                    st.metric("Specificity", f"{tn/(tn+fp):.2%}" if (tn+fp)>0 else "N/A")
                with metrics_col3:
                    st.metric("F1-Score", f"{2*tp/(2*tp+fp+fn):.2%}" if (2*tp+fp+fn)>0 else "N/A")
                    st.metric("False Positive Rate", f"{fp/(fp+tn):.2%}" if (fp+tn)>0 else "N/A")
        
        # Confusion Matrix
        if st.session_state.model_trained:
            st.markdown("---")
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
                fig_cm = px.imshow(cm, text_auto = True,
                                   color_continuous_scale = 'RdBu',
                                   x = ['Predicted Stay', 'Predicted Churn'],
                                   y = ['Actual Stay', 'Actual Churn'])
                fig_cm.update_layout(height = 400,)
                st.plotly_chart(fig_cm, use_container_width=True)
                
            with col4:
                st.subheader("Classification Report")
                report = classification_report(st.session_state.y_test,
                                                   st.session_state.y_pred,
                                                   target_names=['Stay', 'Churn'],
                                                   output_dict=True)
                report_df = pd.DataFrame(report).T
                st.dataframe(report_df.style.format("{:.2%}"), use_container_width = True)

# Data Visualization 
                
elif option == '📊 Data Visualizations':
    st.header('📊 Data Visualization')
    
    if df is not None:
        tab1, tab2, tab3, tab4 = st.tabs(["Churn Analysis", "Customer Demographics", "Financial Metrics", "Correlation Analysis"])
        
        with tab1:
            st.subheader("Churn Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                churn_counts = df['Exited'].value_counts()
                fig_pie = px.pie(values=churn_counts.values, 
                                names=['Stayed', 'Churned'],
                                title="Churn Proportion",
                                color_discrete_sequence=["#4C99AF", "#36ebf4"])
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Churn by Geography
                geo_churn = df.groupby('Geography')['Exited'].mean().reset_index()
                fig_geo = px.bar(geo_churn, x='Geography', y='Exited',
                               title="Churn Rate by Geography",
                               color='Exited',
                               color_continuous_scale='RdYlGn_r')
                st.plotly_chart(fig_geo, use_container_width=True)
        
        with tab2:
            st.subheader("Demographic Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # Age distribution
                fig_age = px.histogram(df, x='Age', color='Exited',
                                     nbins=30,
                                     title="Age Distribution by Churn Status",
                                     marginal='box',
                                     color_discrete_map={0: "#A7AF4C", 1: "#def436"})
                st.plotly_chart(fig_age, use_container_width=True)
            
            with col2:
                # Gender distribution
                gender_churn = pd.crosstab(df['Gender'], df['Exited'], normalize='index') * 100
                fig_gender = px.bar(gender_churn, 
                                   title="Churn Rate by Gender",
                                   labels={'value': 'Percentage', 'Gender': 'Gender'},
                                   color_discrete_sequence=["#AF664C", "#f48f36"])
                st.plotly_chart(fig_gender, use_container_width=True)
        
        with tab3:
            st.subheader("Financial Metrics Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # Balance distribution
                fig_balance = px.violin(df, x='Exited', y='Balance',
                                      title="Balance Distribution by Churn Status",
                                      color='Exited',
                                      box=True,
                                      color_discrete_map={0: "#594CAF", 1: "#367cf4"})
                st.plotly_chart(fig_balance, use_container_width=True)
            
            with col2:
                # Credit Score distribution
                fig_credit = px.box(df, x='Exited', y='CreditScore',
                                  title="Credit Score by Churn Status",
                                  color='Exited',
                                  color_discrete_map={0: '#4CAF50', 1: '#f44336'})
                st.plotly_chart(fig_credit, use_container_width=True)
        
        with tab4:
            st.subheader("Feature Correlation Matrix")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr()
            
            fig_corr = px.imshow(corr_matrix,
                               text_auto=True,
                               color_continuous_scale='RdBu_r',
                               aspect="auto",
                               title="Feature Correlations")
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)

                
elif option == '🔮 Make Prediction':
    st.header("🔮 Make a Prediction")
    
    if not st.session_state.model_trained:
        st.warning('⚠️Please train the model first in the "ANN Model" section')
        if st.button("Go to ANN Model", use_container_width = True):
            st.session_state_.option = '🤖 ANN Model'
            st.rerun()
    else:
        st.markdown("Enter customer details to predict  to predict churn probability")
        col1, col2 = st.columns(2)
        
        with col1:
            geography = st.selectbox("🌎Geography",['France', 'Spain', 'Germany'])
            gender = st.selectbox("👤Gender", ['Male', 'Female'])
            credit_score = st.number_input('💳Credit Score',350, 850, 650)
            age = st.slider('🎂Age', 18, 92, 35)
            tenure = st.slider("📆Tenure (years)", 0, 10, 5) 
            
        with col2:
            balance = st.number_input("💰Balance ($)", min_value = 0.0, max_value = 250898.09, value = 50000.0, step = 1000.0)
            num_products = st.slider("📦Number of Products", 1, 4,2)
            has_cr_card = st.selectbox("💳Has Credit Card", [1,0], format_func = lambda x: 'Yes' if x == 1 else 'No')
            is_active = st.selectbox("⚡Is Active Member", [1,0], format_func = lambda x: 'Yes' if x == 1 else 'No')
            estimated_salary = st.number_input("💵Estimated Salary ($)", min_value = 11.58, max_value = 199992.48, value = 50000.0, step = 1000.0)
            
        if st.button("🔮 Predict Churn Probability", use_container_width = True):
            input_data = pd.DataFrame({
                "CreditScore": [credit_score],
                'Geography': [geography],
                'Gender': [gender],
                'Age': [age],
                'Tenure': [tenure],
                'Balance': [balance],
                'NumOfProducts': [num_products],
                'HasCrCard': [has_cr_card],
                'IsActiveMember': [is_active],
                'EstimatedSalary': [estimated_salary]
            })
                
                #Make Prediction
                
            prediction_proba = st.session_state.pipeline.predict_proba(input_data)[0]
            prediction = st.session_state.pipeline.predict(input_data)[0]
            
            
            #Display Result
            st.markdown("---")
            col_res1, col_res2, col3 = st.columns([1, 2, 1])
            
            with col_res2:
                if prediction == 1:
                    st.markdown(f"""
                                <div class="prediction-result" style="background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);">
                                    ⚠️ The customer is likely to <strong>Churn</strong> with a probability of <strong>{prediction_proba:.2%}</strong>.
                                </div>
                                """, unsafe_allow_html = True)
                    
                else: st.markdown(f"""
                                <div class="prediction-result" style="background: linear-gradient(135deg, #4CAF50 0%, #388E3C 100%);">
                                    ✅ The customer is likely to <strong>Stay</strong> with a probability of <strong>{1-prediction_proba:.2%}</strong>.
                                </div>
                                """, unsafe_allow_html = True)
                
            #Feature importance
            st.markdown("---")
            st.subheader("🔍 Risk Analysis")
            
            #create gauge chart for churn probability
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction_proba*100,
                title = {'text': "Churn Risk Score"},
                domain = {'x': [0,1], 'y': [0,1]},
                gauge = {
                    'axis': {'range':[0,100]},
                    'bar': {'color': 'darkred' if prediction == 1 else 'darkgreen'},
                    'steps': [
                        {'range': [0,30], 'color': '#4CAF50'},
                        {'range': [30,70], 'color': '#FFC107'},
                        {'range': [70,100], 'color': "#f44336"}
                    ],
                    'threshold': {
                        'line': {'color': 'red', 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig_gauge.update_layout(height = 300)
            st.plotly_chart(fig_gauge, use_container_width = True)
            
#Statistics page
elif option == '📈 Statistics':
    st.header("📈 Statistics")
    
    if df is not None:
        tab1, tab2, tab3 = st.tabs(['Dataset Overview','Descriptive Statistics','Feature Analysis'])
        
        with tab1:
            st.subheader('Dataset Information')
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                st.metric("Total Features", len(df.columns))
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep = True).sum() / 1024:.2f} KB")
            with col4:
                st.metric("Missing Values", df.isnull().sum().sum())
                
            st.subheader("Data Types")
            dtype_df = pd.DataFrame(df.dtypes, columns=['Data Type']).reset_index()
            dtype_df.columns = ['Column', 'Data Type']
            st.dataframe(dtype_df, use_container_width=True)
            
            st.subheader("First 10 Rows")
            st.dataframe(df.head(10), use_container_width=True)
            
                
        with tab2:
            st.subheader("Descriptive Statistics")
            
            #Select numeric columns only
            numeric_df = df.select_dtypes(include = [np.number])
            stats_df = numeric_df.describe().T
            stats_df['range'] = stats_df['max'] - stats_df['min']
            stats_df['variance'] = numeric_df.var()
            stats_df['skew'] = numeric_df.skew()
            
            st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
            # Summary by churn status
            st.subheader("Statistics by Churn Status")
            
            churn_stats = df.groupby('Exited')[numeric_df.columns].mean().T
            churn_stats.columns = ['Stayed', 'Churned']
            churn_stats['Difference'] = churn_stats['Churned'] - churn_stats['Stayed']
            churn_stats['% Difference'] = (churn_stats['Difference'] / churn_stats['Stayed'] * 100)
            
            st.dataframe(churn_stats.style.format("{:.2f}"), use_container_width=True)
        
        with tab3:
            st.subheader("Feature Distribution")
            
            #Distribution plots for keys features
            features_to_plot = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
            
            for feature in features_to_plot:
                fig = px.histogram(df, x = feature, color = 'Exited',
                                   nbins = 30,
                                   title = f'{feature} Distribution by Churn Status',
                                   marginal = 'box',
                                   opacity=0.7,
                                   color_discrete_map = {0: '#4CAF50', 1: '#f44336'})
                st.plotly_chart(fig, use_container_width = True)
                
            #categorical features summary
            st.subheader("Categorical Feature Summary")
            cat_cols = ['Geography','Gender','HasCrCard','IsActiveMember','NumOfProducts']
            for col in cat_cols:
                col_counts = df[col].value_counts().reset_index()
                col_counts.columns = [col,'Count']
                col_counts['Percentage'] = col_counts['Count'] / len(df) *100
                
                fig_cat = px.bar(col_counts, x = col, y = 'Count',
                                 text = 'Percentage',
                                 title = f"{col} Distribution",
                                 color = 'Count',
                                color_continuous_scale = 'Viridis')
                fig_cat.update_traces(texttemplate = '%{text:.1f}%', textposition = 'outside')
                st.plotly_chart(fig_cat, use_container_width = True)
                
    else:
        st.error("No Data Available. Please check the data file.")
        
#Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🏦 Customer Churn ANN Prediction System | Built with Streamlit, TensorFlow, and Scikit-learn | https://github.com/lakshaymudgal925-byte</p>
    <p style='font-size: 0.8rem;'>© 2024 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)