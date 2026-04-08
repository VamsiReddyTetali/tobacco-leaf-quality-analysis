import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #050505;
        color: #E0E0E0;
    }
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 5rem;
    }

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
    
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 4. MODEL LOADING & CUSTOM LAYERS
# ==========================================
@tf.keras.utils.register_keras_serializable()
class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size=1, **kwargs):
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

@tf.keras.utils.register_keras_serializable()
class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.proj = tf.keras.layers.Dense(embed_dim)
        self.pos = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=embed_dim)

    def call(self, x):
        pos = tf.range(start=0, limit=tf.shape(x)[1])
        return self.proj(x) + self.pos(pos)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches, 
            "embed_dim": self.embed_dim
        })
        return config

@st.cache_resource
def load_models():
    custom = {"Patches": Patches, "PatchEncoder": PatchEncoder}
    try:
        cnn = tf.keras.models.load_model("cnn_final.keras")
        vit = tf.keras.models.load_model("vit_final.keras", custom_objects=custom)
        hybrid = tf.keras.models.load_model("hybrid_final.keras", custom_objects=custom)
        return cnn, vit, hybrid
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

cnn, vit, hybrid = load_models()

models = {
    "EfficientNetB0 (Texture)": cnn, 
    "ViT (Structure)": vit, 
    "Hybrid Fusion (Proposed)": hybrid
}

# 🔥 4-GRADE SYSTEM
CLASS_NAMES = ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']
IMG_SIZE = 224

# ==========================================
# 5. NAVIGATION LAYOUT
# ==========================================
with st.container():
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
        st.markdown("""
        <div style="display: inline-block; background: rgba(0, 230, 118, 0.1); border: 1px solid #00E676; padding: 5px 15px; border-radius: 20px; color: #00E676; font-size: 0.85rem; font-weight: 600; margin-bottom: 10px;">
            🍃 Aditya University IT Project
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<h1 style="font-size: 3.5rem; font-weight: 800; background: linear-gradient(to right, #ffffff, #a0a0a0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.2;">Automated Tobacco<br>Leaf Grading</h1>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size: 1.1rem; color: #9CA3AF; line-height: 1.6;">
        Eliminating the subjectivity and fatigue of manual visual inspection. Our intelligent system utilizes a <b>Hybrid Deep Learning</b> architecture, fusing local texture analysis with global spatial awareness to achieve a validated <b>95.4% accuracy</b> across 4 complex quality grades.
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Start Grading", key="hero_btn"):
            navigate_to("Analysis")
            st.rerun()

    with col_hero_img:
        st.markdown("""
        <div style="width: 100%; height: 100%; min-height: 350px; background: radial-gradient(circle at center, rgba(0,230,118,0.15) 0%, rgba(20,20,20,0.8) 70%); border: 1px solid rgba(255,255,255,0.05); border-radius: 24px; display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative; overflow: hidden;">
            <div style="font-size: 7rem; margin-bottom: 20px; text-shadow: 0 0 30px rgba(0,230,118,0.4);">🍃</div>
            <div style="display: flex; gap: 20px;">
                <div style="background: rgba(0,0,0,0.5); padding: 10px 20px; border-radius: 12px; border: 1px solid #333; text-align: center;">
                    <div style="color: #00E676; font-size: 1.2rem; font-weight: bold;">95.4%</div>
                    <div style="color: #888; font-size: 0.7rem; text-transform: uppercase;">Accuracy</div>
                </div>
                <div style="background: rgba(0,0,0,0.5); padding: 10px 20px; border-radius: 12px; border: 1px solid #333; text-align: center;">
                    <div style="color: #00E676; font-size: 1.2rem; font-weight: bold;">< 150ms</div>
                    <div style="color: #888; font-size: 0.7rem; text-transform: uppercase;">Latency</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#00E676;">⚡ Fast Inference</h3>
            <p style="color:#999; font-size:0.9rem;">Processed in milliseconds using optimized EfficientNet backbones.</p>
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
            <h3 style="color:#00E676;">🎯 4-Grade System</h3>
            <p style="color:#999; font-size:0.9rem;">Calibrated specifically for Grade 1, 2, 3, and 4 classification.</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# PAGE: ANALYSIS
# ==========================================
elif st.session_state.page == "Analysis":
    
    st.markdown("## 🔍 Leaf Quality Analysis")
    
    mode = st.radio("Select Analysis Mode:", ["Single Prediction (Fast)", "Model Comparison (Detailed)"], horizontal=True)
    st.markdown("---")

    row1_col1, row1_col2 = st.columns([1, 1.5], gap="large")
    
    with row1_col1:
        st.markdown("### 1. Image Input")
        
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
            st.image(image, width="stretch")

    with row1_col2:
        st.markdown("### 2. Grading Results")
        
        if image:
            img_arr = np.array(image.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32)
            img_arr = np.expand_dims(img_arr, 0)
            
            if hybrid is None:
                st.error("⚠️ Models failed to load. Please ensure 'cnn_final.keras', 'vit_final.keras', and 'hybrid_final.keras' are in the directory.")
            else:
                # --- MODE 1: SINGLE PREDICTION ---
                if mode == "Single Prediction (Fast)":
                    
                    selected_model_name = st.selectbox("Select Model for Analysis:", list(models.keys()))
                    model = models[selected_model_name]

                    with st.spinner(f"Analyzing with {selected_model_name}..."):
                        time.sleep(0.5)
                        
                        start_t = time.time()
                        preds = model.predict(img_arr)
                        end_t = time.time()
                        
                        conf = float(np.max(preds)) * 100
                        idx = np.argmax(preds)
                        grade = CLASS_NAMES[idx]
                        latency = (end_t - start_t) * 1000

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

                    # Dynamic plotly gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=conf,
                        title={'text': "Confidence Level", 'font': {'color': 'white'}},
                        number={'font': {'color': 'white'}},
                        gauge={
                            'axis': {'range': [None, 100], 'tickcolor': "white"},
                            'bar': {'color': "#00E676"},
                            'bgcolor': "#333",
                            'steps': [
                                {'range': [0, 50], 'color': "#555"},
                                {'range': [50, 85], 'color': "#777"}
                            ],
                            'threshold': {'line': {'color': "gold", 'width': 4}, 'thickness': 0.75, 'value': 90}
                        }
                    ))
                    fig.update_layout(paper_bgcolor="#050505", font={'color': "white"}, height=250, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig, use_container_width=True)

                    if selected_model_name == "Hybrid Fusion (Proposed)" and conf > 85:
                        st.balloons()

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
                        latency = (end_t - start_t) * 1000
                        results.append({"Model": name, "Grade": g, "Conf": conf, "Latency": latency})
                        i += 1
                        my_bar.progress(i / 3)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    for res in results:
                        border = "2px solid #00E676" if "Hybrid" in res['Model'] else "1px solid #333"
                        bg = "rgba(0, 230, 118, 0.05)" if "Hybrid" in res['Model'] else "#111"
                        badge = '<span style="background:#00E676; color:#000; font-size:0.7rem; padding: 2px 8px; border-radius: 10px; margin-left:10px; font-weight:bold;">RECOMMENDED</span>' if "Hybrid" in res['Model'] else ''
                        
                        st.markdown(f"""
                        <div style="background: {bg}; padding: 15px; border-radius: 10px; border: {border}; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="color: #888; font-size: 0.9rem;">{res['Model']} {badge}</div>
                                <div style="font-size: 1.2rem; font-weight: bold;">{res['Grade']}</div>
                            </div>
                            <div style="text-align: right; display: flex; gap: 20px;">
                                <div>
                                    <div style="color: #888; font-size: 0.7rem;">CONFIDENCE</div>
                                    <div style="color: {'#00E676' if 'Hybrid' in res['Model'] else '#fff'}; font-size: 1.2rem; font-weight: bold;">{res['Conf']:.1f}%</div>
                                </div>
                                <div>
                                    <div style="color: #888; font-size: 0.7rem;">LATENCY</div>
                                    <div style="color: #fff; font-size: 1.2rem; font-weight: bold;">{res['Latency']:.0f} ms</div>
                                </div>
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
    <b>Abstract:</b> We propose a dual-branch architecture that mitigates the limitations of small, imbalanced datasets in agricultural grading. 
    By fusing <b>EfficientNetB0</b> (for local texture) and a custom <b>Vision Transformer</b> (for global geometry), we achieved a validation accuracy of 95.4% across 4 quality grades, significantly outperforming standalone models.
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
                <li><b>Config:</b> Patch Size 16x16, 4 Attention Heads, Projection Dim 64.</li>
                <li><b>Role:</b> Analyzes the spatial structure and shape of the leaf.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 📊 Performance & Results")
    
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px; text-align: center;">
            <p style="color:#aaa; margin:0;">Standalone ViT</p>
            <h2 style="color:#fff; margin:0;">81.2%</h2>
            <p style="font-size: 0.8rem; color:#888;">Struggled with small dataset</p>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px; text-align: center;">
            <p style="color:#aaa; margin:0;">Standalone CNN</p>
            <h2 style="color:#fff; margin:0;">93.8%</h2>
            <p style="font-size: 0.8rem; color:#888;">Strong texture extraction</p>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown("""
        <div style="background: rgba(0,230,118,0.1); border: 1px solid #00E676; padding: 20px; border-radius: 10px; text-align: center;">
            <p style="color:#00E676; font-weight:bold; margin:0;">Hybrid Model</p>
            <h2 style="color:#00E676; margin:0;">95.4%</h2>
            <p style="font-size: 0.8rem; color:#00E676;">Optimal Multi-Branch Performance</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <br>
    <p style="color:#E0E0E0; line-height:1.7;">
    <b>Inference Time Analysis:</b> For industrial deployment, speed is just as crucial as accuracy. We measured the prediction latency of each model. As expected, the standalone Vision Transformer is the fastest, and the CNN operates at a moderate speed. Because the Hybrid model processes data through two separate branches simultaneously, it takes slightly more time to generate a prediction. However, this inference still occurs in milliseconds—a negligible tradeoff for the massive boost in absolute accuracy across all 4 grades.
    </p>
    """, unsafe_allow_html=True)

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
    st.markdown("### Engineering Team")

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
            <div class="team-role">ML Model Engineer</div>
            <div class="team-name">S. Durga Bhavani</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="team-card">
            <div class="team-role">Product & Frontend Developer</div>
            <div class="team-name">S. Tejaswin</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="team-card">
            <div class="team-role">Model Integration Engineer</div>
            <div class="team-name">A. J. Sai Ganesh</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Contact Us")
    st.write("For more information about this project, please contact: **rajababu.makineedi@adityauniversity.in**")