import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Interactive Linear Regression", layout="wide")

st.title("📈 Interactive Linear Regression Learning")
st.markdown("""
Welcome to the interactive system for visualizing how **Linear Regression** learns, minimizing error, and converging to an optimal solution. Use the sidebar to manipulate the data and parameters!
""")

# --- Helper Functions ---
@st.cache_data
def generate_data(n, noise, outliers):
    np.random.seed(42)  # For reproducibility during interaction
    X = np.linspace(0, 10, n)
    y = 2.5 * X + 10.0 + np.random.normal(0, noise, n)
    if outliers:
        n_out = max(1, int(n * 0.1))
        outlier_indices = np.random.choice(n, n_out, replace=False)
        y[outlier_indices] += np.random.choice([-1, 1], n_out) * (noise * 3 + 20)
    return X, y

def compute_mse(X, y, m, b):
    return np.mean((y - (m * X + b))**2)

# --- Sidebar Controls ---
st.sidebar.header("1. Dataset Controls")
n_samples = st.sidebar.slider("Number of Data Points", 10, 200, 50)
noise_level = st.sidebar.slider("Noise Level", 0.0, 20.0, 5.0)
add_outliers = st.sidebar.checkbox("Add Outliers")

# Generate dataset based on sidebar
X, y = generate_data(n_samples, noise_level, add_outliers)

st.sidebar.markdown("---")
st.sidebar.header("2. Manual Line Fitting")
user_m = st.sidebar.slider("Slope (m)", -5.0, 10.0, 0.0, 0.1)
user_b = st.sidebar.slider("Intercept (b)", -10.0, 30.0, 0.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.header("3. Optimizaton (Gradient Descent)")
learning_rate = st.sidebar.slider("Learning Rate (α)", 0.001, 0.05, 0.01, format="%f")
epochs = st.sidebar.slider("Iterations", 10, 200, 50)

# --- Computations ---
y_pred = user_m * X + user_b
current_mse = compute_mse(X, y, user_m, user_b)

# --- Tabs Structure ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Data & Line Fit", 
    "2. Error (MSE)", 
    "3. Loss Landscape", 
    "4. Gradient Descent", 
    "5. Learning Rate", 
    "6. Noise & Robustness"
])

# --- TAB 1: Data & Line Fit ---
with tab1:
    st.subheader("Data Distribution & Line Fitting")
    st.markdown("**Concept:** Linear Regression assumes $y = mx + b$. Adjust the sliders in the sidebar to see how changing $m$ and $b$ shifts the line.")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Data Points', marker=dict(color='blue', size=8)))
    fig.add_trace(go.Scatter(x=X, y=y_pred, mode='lines', name=f'Hypothesis: y={user_m:.2f}x + {user_b:.2f}', line=dict(color='red', width=3)))
    
    fig.update_layout(height=500, xaxis_title="X", yaxis_title="y", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: Error Visualization ---
with tab2:
    st.subheader("Error / Loss Function (MSE)")
    st.markdown(f"**Concept:** We minimize the distance from the points to the line. Currently, **MSE = {current_mse:.2f}**")
    
    fig2 = go.Figure()
    # Add data points
    fig2.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Data Points', marker=dict(color='blue')))
    # Add line
    fig2.add_trace(go.Scatter(x=X, y=y_pred, mode='lines', name='Hypothesis Line', line=dict(color='red')))
    # Add residual lines
    for i in range(len(X)):
        fig2.add_trace(go.Scatter(
            x=[X[i], X[i]], y=[y[i], y_pred[i]], 
            mode='lines', line=dict(color='gray', dash='dot'), showlegend=False
        ))
    
    fig2.update_layout(height=500, xaxis_title="X", yaxis_title="y", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

# --- TAB 3: Loss Landscape ---
with tab3:
    st.subheader("Loss Surface (Parameter Space Visualization)")
    st.markdown("**Concept:** The Error is a function of parameters $J(m, b)$. We navigate this bowl-shaped surface to find the lowest point (the minimum error).")
    
    m_vals = np.linspace(-5, 10, 50)
    b_vals = np.linspace(-10, 30, 50)
    M, B = np.meshgrid(m_vals, b_vals)
    Z = np.zeros_like(M)
    for i in range(len(m_vals)):
        for j in range(len(b_vals)):
            Z[j, i] = compute_mse(X, y, M[j, i], B[j, i])
            
    fig3 = go.Figure(data=[go.Surface(z=Z, x=M, y=B, colorscale='Viridis', opacity=0.8)])
    # Add Current Position marker
    fig3.add_trace(go.Scatter3d(
        x=[user_m], y=[user_b], z=[current_mse],
        mode='markers', marker=dict(color='red', size=8, symbol='circle'),
        name='Current Parameters'
    ))
    fig3.update_layout(
        scene=dict(xaxis_title='Slope (m)', yaxis_title='Intercept (b)', zaxis_title='MSE'),
        height=600, margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig3, use_container_width=True)

# --- TAB 4: Gradient Descent ---
with tab4:
    st.subheader("Gradient Descent Optimization")
    st.markdown("**Concept:** Gradient descent iteratively updates $m$ and $b$ in the direction of steepest descent.")
    
    if st.button("Run Gradient Descent Animation"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        plot_placeholder = st.empty()
        
        m_curr, b_curr = user_m, user_b
        history_m, history_b, history_loss = [m_curr], [b_curr], [current_mse]
        
        for i in range(epochs):
            # Compute gradients
            y_curr_pred = m_curr * X + b_curr
            dm = (-2/n_samples) * sum(X * (y - y_curr_pred))
            db = (-2/n_samples) * sum(y - y_curr_pred)
            
            # Update params
            m_curr = m_curr - learning_rate * dm
            b_curr = b_curr - learning_rate * db
            loss = compute_mse(X, y, m_curr, b_curr)
            
            history_m.append(m_curr)
            history_b.append(b_curr)
            history_loss.append(loss)
            
            # Animate every 5 steps
            if i % max(1, epochs//20) == 0 or i == epochs - 1:
                progress_bar.progress((i + 1) / epochs)
                status_text.text(f"Iteration {i+1}/{epochs} | Loss: {loss:.2f} | m={m_curr:.2f}, b={b_curr:.2f}")
                
                fig4 = go.Figure(data=[go.Contour(z=Z, x=m_vals, y=b_vals, colorscale='Viridis', opacity=0.5)])
                fig4.add_trace(go.Scatter(x=history_m, y=history_b, mode='lines+markers', marker=dict(color='red', size=4), name='Optimization Path'))
                fig4.update_layout(height=500, xaxis_title="Slope (m)", yaxis_title="Intercept (b)", template="plotly_white")
                plot_placeholder.plotly_chart(fig4, use_container_width=True)
        st.success("Gradient Descent Completed!")

# --- TAB 5: Learning Rate ---
with tab5:
    st.subheader("Learning Rate & Convergence Behavior")
    st.markdown("**Concept:** The Learning Rate $\\alpha$ dictates step size. Too small = slow convergence; too large = divergence.")
    
    st.markdown("Comparing convergence with different learning rates starting from your manually set $m$ and $b$:")
    
    lrs_to_test = [0.001, 0.01, 0.05]
    fig5 = go.Figure()
    
    for lr_test in lrs_to_test:
        m_t, b_t = user_m, user_b
        losses = []
        for _ in range(epochs):
            y_t_pred = m_t * X + b_t
            dm = (-2/n_samples) * sum(X * (y - y_t_pred))
            db = (-2/n_samples) * sum(y - y_t_pred)
            m_t -= lr_test * dm
            b_t -= lr_test * db
            # Limit loss to prevent overflow visualization
            losses.append(min(compute_mse(X, y, m_t, b_t), 1e5)) 
        fig5.add_trace(go.Scatter(y=losses, mode='lines', name=f'LR = {lr_test}'))
        
    fig5.update_layout(height=500, xaxis_title="Iteration", yaxis_title="MSE Loss", template="plotly_white")
    st.plotly_chart(fig5, use_container_width=True)

# --- TAB 6: Noise & Robustness ---
with tab6:
    st.subheader("Effect of Noise & Outliers")
    st.markdown("**Concept:** Linear Regression fits the data minimizing MSE, making it sensitive to large outlier points.")
    
    # Calculate OLS using sklearn
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y)
    y_ols = model.predict(X.reshape(-1, 1))
    
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Data (Toggle Outliers in Sidebar)', marker=dict(color='blue')))
    fig6.add_trace(go.Scatter(x=X, y=y_ols, mode='lines', name=f'Optimal OLS Fit (m={model.coef_[0]:.2f})', line=dict(color='green', width=3)))
    
    fig6.update_layout(height=500, xaxis_title="X", yaxis_title="y", template="plotly_white")
    st.plotly_chart(fig6, use_container_width=True)
    st.info("Try checking/unchecking 'Add Outliers' in the sidebar to see how the optimal green line gets heavily pulled away from the main cluster by extreme points!")

