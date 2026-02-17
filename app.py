import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AgroGrade AI",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. SESSION STATE & NAVIGATION
# ==========================================
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

def navigate_to(page_name):
    st.session_state.page = page_name

# ==========================================
# 3. CUSTOM CSS (STYLED NAV & THEME)
# ==========================================
st.markdown("""
    <style>
    /* GLOBAL FONTS & COLORS */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #050505;
        color: #E0E0E0;
    }
    
    /* REMOVE DEFAULT PADDING */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 5rem;
    }

    /* CUSTOM NAVIGATION BAR */
    /* This targets the container holding the nav items to ensure vertical centering */
    [data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    
    .nav-logo {
        font-size: 1.8rem;
        font-weight: 800;
        color: #fff;
        display: flex;
        align-items: center;
        margin-bottom: 0;
    }
    
    .nav-logo span {
        color: #00E676;
        margin-left: 5px;
    }

    /* CUSTOMIZE STREAMLIT BUTTONS FOR NAV */
    div.stButton > button {
        background-color: transparent;
        color: #E0E0E0;
        border: 1px solid transparent;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:hover {
        background-color: rgba(0, 230, 118, 0.1);
        color: #00E676;
        border: 1px solid rgba(0, 230, 118, 0.3);
    }
    
    div.stButton > button:focus {
        border-color: #00E676;
        color: #00E676;
        box-shadow: none;
    }
    
    /* GLASS CARDS */
    .glass-card {
        background: rgba(20, 20, 20, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2rem;
        height: 100%;
    }
    
    .result-card {
        background: rgba(0, 230, 118, 0.05);
        border: 1px solid #00E676;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 0 30px rgba(0, 230, 118, 0.1);
        margin-bottom: 20px;
    }
    
    /* TEAM CARD */
    .team-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: 0.3s;
        height: 100%;
    }
    
    .team-card:hover {
        border-color: #00E676;
        transform: translateY(-5px);
    }
    
    /* HIDE DEFAULT HEADER */
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 4. MODEL LOADING
# ==========================================
# Define the custom Patches layer exactly as it appears in your training code
class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
    
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

@st.cache_resource
def load_models():
    custom = {"Patches": Patches}
    try:
        # Load the models you trained and saved
        cnn = tf.keras.models.load_model("cnn_model.keras")
        vit = tf.keras.models.load_model("vit_model.keras", custom_objects=custom)
        hybrid = tf.keras.models.load_model("hybrid_model.keras", custom_objects=custom)
        return cnn, vit, hybrid
    except Exception as e:
        return None, None, None

cnn, vit, hybrid = load_models()

# Model Dictionary
models = {
    "EfficientNetB0 (Texture)": cnn, 
    "ViT (Structure)": vit, 
    "Hybrid Fusion (Proposed)": hybrid
}

# Updated Class Names based on your training folder structure ['A', 'B', 'C']
CLASS_NAMES = ['Grade A', 'Grade B', 'Grade C']
IMG_SIZE = 224

# ==========================================
# 5. NAVIGATION LAYOUT (Fixed Alignment)
# ==========================================
# Using a container to group the nav elements
with st.container():
    # Adjusted column ratios for better spacing
    c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1])
    
    with c1:
        st.markdown('<div class="nav-logo">🍃 Agro<span>Grade</span></div>', unsafe_allow_html=True)
    
    with c2:
        if st.button("Home"): navigate_to("Home")
    with c3:
        if st.button("Analysis"): navigate_to("Analysis")
    with c4:
        if st.button("Research"): navigate_to("Research")
    with c5:
        if st.button("About"): navigate_to("About")

st.markdown("---")

# ==========================================
# PAGE: HOME
# ==========================================
if st.session_state.page == "Home":
    
    col_hero_text, col_hero_img = st.columns([1.2, 1])
    
    with col_hero_text:
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        st.markdown('<h1 style="font-size: 3.5rem; font-weight: 800; background: linear-gradient(to right, #fff, #888); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Automated Tobacco<br>Leaf Grading</h1>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size: 1.1rem; color: #9CA3AF; line-height: 1.6;">
        An intelligent system utilizing Hybrid Deep Learning. We combine <b>EfficientNetB0</b> for texture analysis and <b>Vision Transformers</b> for global context to achieve 95.2% accuracy.
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Start Grading", key="hero_btn"):
            navigate_to("Analysis")
            st.rerun()

    with col_hero_img:
        # Placeholder visual
        st.markdown("""
        <div style="width: 100%; height: 350px; background: radial-gradient(circle, rgba(0,230,118,0.1) 0%, rgba(0,0,0,0) 70%); border-radius: 20px; display: flex; align-items: center; justify-content: center;">
            <div style="font-size: 8rem;">🍃</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Value Proposition Cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#00E676;">⚡ Fast Inference</h3>
            <p style="color:#999; font-size:0.9rem;">Processed in < 50ms using optimized EfficientNet backbones.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#00E676;">🧠 Hybrid AI</h3>
            <p style="color:#999; font-size:0.9rem;">Fuses Local CNN texture features with Global ViT Attention.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#00E676;">🎯 3-Grade System</h3>
            <p style="color:#999; font-size:0.9rem;">Calibrated specifically for Grade A, B, and C classification.</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# PAGE: ANALYSIS
# ==========================================
# ==========================================
# PAGE: ANALYSIS
# ==========================================
elif st.session_state.page == "Analysis":
    
    st.markdown("## 🔍 Leaf Quality Analysis")
    
    # Mode Selection for prediction type
    mode = st.radio("Select Analysis Mode:", ["Single Prediction (Fast)", "Model Comparison (Detailed)"], horizontal=True)
    st.markdown("---")

    row1_col1, row1_col2 = st.columns([1, 1.5], gap="large")
    
    with row1_col1:
        st.markdown("### 1. Image Input")
        
        # WEBCAM FIX: Using st.radio to explicitly toggle source avoids layout jumps
        input_source = st.radio("Select Source:", ["Upload Image", "Use Camera"], horizontal=True)
        
        image = None
        
        if input_source == "Upload Image":
            st.markdown("""
            <div style="border: 1px dashed #444; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 10px;">
                <span style="color: #666;">Supports JPG, PNG</span>
            </div>
            """, unsafe_allow_html=True)
            uploaded = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
            if uploaded:
                image = Image.open(uploaded).convert("RGB")
        
        elif input_source == "Use Camera":
            st.warning("Ensure your browser allows camera access.")
            cam_img = st.camera_input("Take a photo")
            if cam_img:
                image = Image.open(cam_img).convert("RGB")
        
        if image:
            st.markdown("### Preview")
            st.image(image, use_container_width=True)

    with row1_col2:
        st.markdown("### 2. Grading Results")
        
        if image:
            # Preprocessing
            img_arr = np.array(image.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32)
            img_arr = np.expand_dims(img_arr, 0)
            
            # Check if models loaded
            if hybrid is None:
                st.error("⚠️ Models failed to load. Please ensure 'cnn_model.keras', 'vit_model.keras', and 'hybrid_model.keras' are in the directory.")
            else:
                # --- MODE 1: SINGLE PREDICTION ---
                if mode == "Single Prediction (Fast)":
                    
                    # === NEW: DROP DOWN MENU ===
                    selected_model_name = st.selectbox("Select Model for Analysis:", list(models.keys()))
                    model = models[selected_model_name]
                    # ===========================

                    with st.spinner(f"Analyzing with {selected_model_name}..."):
                        # Artificial delay for UX
                        time.sleep(0.5)
                        
                        start_t = time.time()
                        preds = model.predict(img_arr)
                        end_t = time.time()
                        
                        conf = float(np.max(preds)) * 100
                        idx = np.argmax(preds)
                        grade = CLASS_NAMES[idx]
                        latency = (end_t - start_t) * 1000

                    # Result Card
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px; color: #00E676; margin-bottom: 10px;">{selected_model_name} Output</div>
                        <h1 style="font-size: 3.5rem; margin: 0; color: #fff;">{grade}</h1>
                        <div style="margin-top: 20px; display: flex; justify-content: center; gap: 40px;">
                            <div>
                                <div style="color: #888; font-size: 0.8rem;">CONFIDENCE</div>
                                <div style="font-size: 1.5rem; font-weight: 700;">{conf:.1f}%</div>
                            </div>
                            <div>
                                <div style="color: #888; font-size: 0.8rem;">LATENCY</div>
                                <div style="font-size: 1.5rem; font-weight: 700;">{latency:.0f} ms</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Bar Chart
                    st.markdown("#### Class Probabilities")
                    probs = {name: float(p)*100 for name, p in zip(CLASS_NAMES, preds[0])}
                    
                    # Create custom dark theme chart
                    fig, ax = plt.subplots(figsize=(6, 2))
                    fig.patch.set_facecolor('#050505')
                    ax.set_facecolor('#050505')
                    
                    y_pos = np.arange(len(probs))
                    bars = ax.barh(y_pos, list(probs.values()), color='#00E676')
                    
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(list(probs.keys()), color='white')
                    ax.set_xticks([0, 25, 50, 75, 100])
                    ax.tick_params(colors='white', axis='x')
                    
                    for spine in ax.spines.values(): 
                        spine.set_edgecolor('#333')
                    
                    st.pyplot(fig)
                    plt.close(fig) # Close figure to free memory

                # --- MODE 2: MODEL COMPARISON ---
                else:
                    st.markdown("Running comprehensive analysis...")
                    results = []
                    my_bar = st.progress(0)
                    
                    i = 0
                    for name, model in models.items():
                        start_t = time.time()
                        p = model.predict(img_arr)
                        end_t = time.time()
                        
                        conf = float(np.max(p)) * 100
                        g = CLASS_NAMES[np.argmax(p)]
                        results.append({"Model": name, "Grade": g, "Conf": conf})
                        i += 1
                        my_bar.progress(i / 3)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    for res in results:
                        border = "2px solid #00E676" if "Hybrid" in res['Model'] else "1px solid #333"
                        bg = "rgba(0, 230, 118, 0.05)" if "Hybrid" in res['Model'] else "#111"
                        
                        st.markdown(f"""
                        <div style="background: {bg}; padding: 15px; border-radius: 10px; border: {border}; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="color: #888; font-size: 0.9rem;">{res['Model']}</div>
                                <div style="font-size: 1.2rem; font-weight: bold;">{res['Grade']}</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="color: #00E676; font-size: 1.2rem; font-weight: bold;">{res['Conf']:.1f}%</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        else:
            st.info("👈 Please select an image source to begin.")


# ==========================================
# PAGE: RESEARCH
# ==========================================
elif st.session_state.page == "Research":
    st.markdown("# 📘 Research Methodology")
    st.markdown("*Automated Tobacco Leaf Grading Using Hybrid Deep Learning*")
    
    st.markdown("""
    <div class="glass-card" style="border-left: 4px solid #00E676;">
    <b>Abstract:</b> We propose a dual-branch architecture that mitigates the limitations of small datasets (300 images per class) in agricultural grading. 
    By fusing <b>EfficientNetB0</b> (for local texture) and a custom <b>Vision Transformer</b> (for global geometry), we achieved a validation accuracy of 95.2%, significantly outperforming standalone models.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🏗️ Architecture Design")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#00E676">Branch 1: Local Texture (CNN)</h4>
            <ul style="color:#aaa; line-height:1.6">
                <li><b>Model:</b> EfficientNetB0 (Pre-trained on ImageNet)</li>
                <li><b>Role:</b> Extracts high-frequency details like leaf veins, spots, and surface texture.</li>
                <li><b>Output:</b> Global Average Pooling Vector.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#00E676">Branch 2: Global Context (ViT)</h4>
            <ul style="color:#aaa; line-height:1.6">
                <li><b>Model:</b> Custom Vision Transformer</li>
                <li><b>Config:</b> Patch Size 16x16, 4 Attention Heads, Projection Dim 128.</li>
                <li><b>Role:</b> Analyzes the spatial structure and shape of the leaf.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Dataset & Training")
    st.markdown("""
    * **Classes:** 3 (Grade A, Grade B, Grade C)
    * **Total Images:** 900 (Balanced distribution: 300 per class)
    * **Augmentation:** RandomFlip, RandomRotation (0.1), RandomZoom (0.1) to simulate industrial conveyor misalignment.
    * **Fusion Strategy:** Feature Concatenation → Dense (512) → Dropout (0.5) → Output.
    """)

# ==========================================

# PAGE: ABOUT

# ==========================================

elif st.session_state.page == "About":
    st.markdown("## 👥 Meet the Team")
    st.markdown("Developed by students and faculty of **Aditya University**, Department of Information Technology.")
    st.markdown("<br>", unsafe_allow_html=True)

    # Project Guide
    st.markdown("### Project Guide")
    st.markdown("""
    <div class="team-card">
        <div class="team-role">Associate Professor</div>
        <div class="team-name">Dr. M. Rajababu</div>
        <div style="color:#666; font-size:0.8rem; margin-top:5px;">Department of Information Technology</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Researchers & Developers")

    # Team Members Grid
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="team-card">
            <div class="team-role">Lead Developer</div>
            <div class="team-name">T. D. N. Vamsi Reddy</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="team-card">
            <div class="team-role">Researcher</div>
            <div class="team-name">S. Durga Bhavani</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="team-card">
            <div class="team-role">Researcher</div>
            <div class="team-name">S. Tejaswin</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="team-card">
            <div class="team-role">Researcher</div>
            <div class="team-name">A. J. Sai Ganesh</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Contact Us")
    st.write("For more information about this project, please contact: **22a91a12b9@aec.edu.in**")