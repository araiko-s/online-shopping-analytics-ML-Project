# Restored version of our university Online Shopping project.
# Original sections and model choices are kept; broken file paths and model issues are fixed.
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, f1_score, silhouette_score,
                             davies_bouldin_score, calinski_harabasz_score)
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

PROJECT_DIR = Path(__file__).resolve().parent

st.title('Online shopping')


# Define a function that shows and hides the information when the button is clicked
def toggle_info():
    if st.session_state.show_info:
        st.write("Administrative - Number of pages visited by the visitor about account management")
        st.write(
            "Administrative duration - Total amount of time (in seconds) spent by the visitor on account management related pages")
        st.write(
            "Informational - Number of pages visited by the visitor about Web site, communication and address information of the shopping site")
        st.write(
            "Informational duration - Total amount of time (in seconds) spent by the visitor on informational pages")
        st.write("Product related - Number of pages visited by visitor about product related pages")
        st.write(
            "Product related duration - Total amount of time (in seconds) spent by the visitor on product related pages")
        st.write("Bounce rate - Average bounce rate value of the pages visited by the visitor")
        st.write("Exit rate - Average exit rate value of the pages visited by the visitor")
        st.write("Page value - Average page value of the pages visited by the visitor")
        st.write("Special day - Closeness of the site visiting time to a special day")
        st.write("Operating Systems - Operating system of the visitor")
        st.write("Browser - Browser of the visitor")
        st.write("Region Geographic - region from which the session has been started by the visitor")
        st.write(
            "TrafficType - Traffic source by which the visitor has arrived at the Web site (e.g., banner, SMS, direct)")
        st.write("VisitorType - Visitor type as ‘‘New Visitor,’’ ‘‘Returning Visitor,’’ and ‘‘Other’’")
        st.write("Weekend - Boolean value indicating whether the date of the visit is weekend")
        st.write("Month - Month value of the visit date")
        st.write("Revenue - Class label indicating whether the visit has been finalized with a transaction")
    else:
        st.write("Click the button to show the information")


# Create a button that toggles the display of the information
if "show_info" not in st.session_state:
    st.session_state.show_info = False
if st.button("Show/Hide Information"):
    st.session_state.show_info = not st.session_state.show_info
toggle_info()


csv_path = PROJECT_DIR / "online_shoppers_intention.csv"
if csv_path.exists():
    df = pd.read_csv(csv_path)
else:
    uploaded_csv = st.file_uploader("Upload online_shoppers_intention.csv", type="csv")
    if uploaded_csv is None:
        st.info("Add the CSV to this folder or upload it to continue.")
        st.stop()
    df = pd.read_csv(uploaded_csv)

if st.checkbox('Show dataframe'):
    st.write(df)

st.sidebar.header('User Input Features')
species = st.sidebar.multiselect('Did the sale occur?', df['Revenue'].unique().tolist(), default=df['Revenue'].unique().tolist())

st.subheader('Scatter plot')
col1 = st.selectbox('Which feature on x?', df.columns[0:16])
col2 = st.selectbox('Which feature on y?', df.columns[0:16])

new_df = df[(df['Revenue'].isin(species))]

# create figure using plotly express
fig = px.scatter(new_df, x =col1,y=col2, color='Revenue')
# Plot!

st.plotly_chart(fig)

st.subheader('Histogram')


feature = st.selectbox('Which feature?', df.columns[0:16])
# Filter dataframe
new_df2 = df[(df['Revenue'].isin(species))][feature]
fig2 = px.histogram(new_df, x=feature, color="Revenue", marginal="rug")
st.plotly_chart(fig2)

# Keep the original dataset for charts; encode categories for the ML models.
X = pd.get_dummies(df.drop(columns=['Revenue']), dummy_na=True).astype(float)
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
y = df['Revenue'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

st.title('ML MODELS')
alg = ['Logistic Regression', 'K means clustering','Decision tree']
st.sidebar.subheader('ML MODELS')
classifier = st.sidebar.selectbox('Which algorithm?', alg)
if classifier=='Logistic Regression':
    st.title('Logistic Regression')
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    st.write('Accuracy:', accuracy_score(y_test, pred_lr))

    cm = confusion_matrix(y_test, pred_lr)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_ylabel('Actual label')
    ax.set_xlabel('Predicted label')
    st.pyplot(fig)
    plt.close(fig)

    st.write('Precision:', precision_score(y_test, pred_lr, zero_division=0))
    st.write('Recall:', recall_score(y_test, pred_lr, zero_division=0))
    st.write('F1-Score:', f1_score(y_test, pred_lr, zero_division=0))
    train_acc = lr.score(X_train, y_train)
    test_acc = lr.score(X_test, y_test)
    st.write('train_acc', train_acc)
    st.write('test_acc', test_acc)
    # The original learning curve was a missing PNG. Show the actual scores instead.
    st.caption('Train/test accuracy comparison (not a full learning curve)')
    st.bar_chart(pd.DataFrame({'Accuracy': [train_acc, test_acc]}, index=['Train', 'Test']))

    st.subheader('L1 and L2 regularization')
    C = st.slider('Select C for L1', 0.01, 1.0, 1.0, 0.01)
    lr_lasso = make_pipeline(StandardScaler(), LogisticRegression(
        penalty='l1', solver='liblinear', C=C, max_iter=2000))
    lr_lasso.fit(X_train, y_train)
    y_pred = lr_lasso.predict(X_test)
    st.write('L1 Accuracy:', accuracy_score(y_test, y_pred))
    st.write('L1 Precision:', precision_score(y_test, y_pred, zero_division=0))
    st.write('L1 Recall:', recall_score(y_test, y_pred, zero_division=0))
    st.write('L1 F1:', f1_score(y_test, y_pred, zero_division=0))

    D = st.slider('Select C for L2', 0.01, 1.0, 1.0, 0.01)
    lr_ridge = make_pipeline(StandardScaler(), LogisticRegression(
        penalty='l2', solver='lbfgs', C=D, max_iter=2000))
    lr_ridge.fit(X_train, y_train)
    y_pred = lr_ridge.predict(X_test)
    st.write('L2 Accuracy:', accuracy_score(y_test, y_pred))
    st.write('L2 Precision:', precision_score(y_test, y_pred, zero_division=0))
    st.write('L2 Recall:', recall_score(y_test, y_pred, zero_division=0))
    st.write('L2 F1:', f1_score(y_test, y_pred, zero_division=0))
elif classifier=='K means clustering':
    st.title('KMeans Clustering')
    num_clusters = st.slider('Select number of clusters', 2, 7, 2)
    # Revenue is the outcome, so it should not be used to build clusters.
    scaled_X = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_X)
    clustered_df = df.copy()
    clustered_df['cluster'] = clusters.astype(str)
    fig3 = px.scatter(clustered_df, x=col1, y=col2, color='cluster')
    st.plotly_chart(fig3)
    # Sample for silhouette score to keep the interactive app responsive.
    sil_score = silhouette_score(scaled_X, clusters, sample_size=min(2000, len(X)), random_state=42)
    st.write('Silhouette Score:', sil_score)
    st.write('Davies-Bouldin Score:', davies_bouldin_score(scaled_X, clusters))
    st.write('Calinski-Harabasz Score:', calinski_harabasz_score(scaled_X, clusters))
    st.caption('Clustering uses scaled input features and excludes Revenue.')
elif classifier=='Decision tree':
    st.title('Decision Tree')
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
    tree.fit(X_train, y_train)
    st.write('Decision Tree Classifier Accuracy:', tree.score(X_test, y_test))
    y_pred = tree.predict(X_test)
    st.write('Precision:', precision_score(y_test, y_pred, zero_division=0))
    st.write('Recall:', recall_score(y_test, y_pred, zero_division=0))
    st.write('F1:', f1_score(y_test, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(tree, filled=True, max_depth=3, feature_names=X.columns,
              class_names=['No purchase', 'Purchase'], fontsize=7, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader('Decision Tree Learning Curve')
    sizes, train_scores, test_scores = learning_curve(
        DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42),
        X, y, cv=3, train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=1)
    fig, ax = plt.subplots()
    ax.plot(sizes, train_scores.mean(axis=1), label='Training score')
    ax.plot(sizes, test_scores.mean(axis=1), label='Cross-validation score')
    ax.set_xlabel('Number of training samples')
    ax.set_ylabel('Accuracy')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# Optional reminder of the original university project.
st.sidebar.subheader('University project extra')
with st.sidebar.expander('Original animation'):
    video_path = PROJECT_DIR / 'university_animation.mp4'
    if video_path.exists():
        st.video(str(video_path))
    else:
        st.write('Optional: put university_animation.mp4 next to this file.')
