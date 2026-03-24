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
    
    # 1. HERO SECTION
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
        <p style="font-size: 1.1rem; color: #9CA3AF; line-height: 1.6; margin-top: 15px;">
        Eliminating the subjectivity and fatigue of manual visual inspection. Our intelligent system utilizes a <b>Hybrid Deep Learning</b> architecture, fusing local texture analysis with global spatial awareness to achieve a validated <b>93.33% accuracy</b>.
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Primary Call to Action
        if st.button("🚀 Start Grading Analysis", key="hero_btn"):
            navigate_to("Analysis")
            st.rerun()

    with col_hero_img:
        # Enhanced visual placeholder with stats
        st.markdown("""
        <div style="width: 100%; height: 100%; min-height: 350px; background: radial-gradient(circle at center, rgba(0,230,118,0.15) 0%, rgba(20,20,20,0.8) 70%); border: 1px solid rgba(255,255,255,0.05); border-radius: 24px; display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative; overflow: hidden;">
            <div style="font-size: 7rem; margin-bottom: 20px; text-shadow: 0 0 30px rgba(0,230,118,0.4);">🍃</div>
            <div style="display: flex; gap: 20px;">
                <div style="background: rgba(0,0,0,0.5); padding: 10px 20px; border-radius: 12px; border: 1px solid #333; text-align: center;">
                    <div style="color: #00E676; font-size: 1.2rem; font-weight: bold;">93.3%</div>
                    <div style="color: #888; font-size: 0.7rem; text-transform: uppercase;">Accuracy</div>
                </div>
                <div style="background: rgba(0,0,0,0.5); padding: 10px 20px; border-radius: 12px; border: 1px solid #333; text-align: center;">
                    <div style="color: #00E676; font-size: 1.2rem; font-weight: bold;">< 300ms</div>
                    <div style="color: #888; font-size: 0.7rem; text-transform: uppercase;">Latency</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # 2. VALUE PROPOSITION CARDS
    st.markdown("### Why AgroGrade AI?")
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#00E676; font-size: 1.3rem;">⚡ High-Speed Inference</h3>
            <p style="color:#999; font-size:0.95rem; line-height: 1.5; margin-top: 10px;">
            Designed for real-world industrial pacing. By utilizing highly optimized EfficientNet backbones, the system processes images in milliseconds, keeping up with active conveyor belts.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#00E676; font-size: 1.3rem;">🧠 Hybrid Intelligence</h3>
            <p style="color:#999; font-size:0.95rem; line-height: 1.5; margin-top: 10px;">
            The best of both worlds. We successfully fused the microscopic texture feature extraction of CNNs with the macroscopic structural awareness of Vision Transformers.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#00E676; font-size: 1.3rem;">🎯 Precision Grading</h3>
            <p style="color:#999; font-size:0.95rem; line-height: 1.5; margin-top: 10px;">
            Calibrated specifically for industry standards. The model accurately classifies leaves into Grade A (High Quality), Grade B (Moderate), and Grade C (Low/Damaged).
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # 3. QUICK GUIDE SECTION
    st.markdown("""
    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 16px; padding: 30px;">
        <h3 style="margin-top: 0; color: #E0E0E0;">Getting Started</h3>
        <div style="display: flex; gap: 30px; margin-top: 20px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 200px;">
                <div style="color: #00E676; font-size: 1.5rem; font-weight: bold; margin-bottom: 10px;">1</div>
                <div style="font-weight: 600; margin-bottom: 5px;">Navigate to Analysis</div>
                <div style="color: #888; font-size: 0.9rem;">Click the 'Start Grading' button or use the top navigation bar.</div>
            </div>
            <div style="flex: 1; min-width: 200px;">
                <div style="color: #00E676; font-size: 1.5rem; font-weight: bold; margin-bottom: 10px;">2</div>
                <div style="font-weight: 600; margin-bottom: 5px;">Upload or Capture</div>
                <div style="color: #888; font-size: 0.9rem;">Provide an image of a cured tobacco leaf using your file explorer or webcam.</div>
            </div>
            <div style="flex: 1; min-width: 200px;">
                <div style="color: #00E676; font-size: 1.5rem; font-weight: bold; margin-bottom: 10px;">3</div>
                <div style="font-weight: 600; margin-bottom: 5px;">View Insights</div>
                <div style="color: #888; font-size: 0.9rem;">Instantly receive the predicted grade, confidence metrics, and comparative model analysis.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
            st.markdown("### Image Preview")
            st.image(image, use_container_width=True, clamp=True)
            # Added Image Metadata for an "Industrial Tool" feel
            st.markdown(f"<p style='color:#888; font-size:0.8rem; text-align:center;'>Original Resolution: {image.size[0]}x{image.size[1]} px</p>", unsafe_allow_html=True)

    with row1_col2:
        st.markdown("### 2. Grading Results")
        
        if image:
            # Preprocessing
            img_arr = np.array(image.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32)
            img_arr = np.expand_dims(img_arr, 0)
            
            if hybrid is None:
                st.error("⚠️ Models failed to load. Please ensure 'cnn_model.keras', 'vit_model.keras', and 'hybrid_model.keras' are in the directory.")
            else:
                # --- MODE 1: SINGLE PREDICTION ---
                if mode == "Single Prediction (Fast)":
                    
                    selected_model_name = st.selectbox("Select Model for Analysis:", list(models.keys()))
                    model = models[selected_model_name]

                    with st.spinner(f"Analyzing with {selected_model_name}..."):
                        time.sleep(0.3) # Slight delay for UX
                        
                        start_t = time.time()
                        preds = model.predict(img_arr)
                        end_t = time.time()
                        
                        conf = float(np.max(preds)) * 100
                        idx = np.argmax(preds)
                        grade = CLASS_NAMES[idx]
                        latency = (end_t - start_t) * 1000

                    # Dynamic styling based on confidence
                    border_color = "#00E676" if conf >= 75 else ("#FFB300" if conf >= 50 else "#F44336")
                    
                    # Result Card
                    st.markdown(f"""
                    <div style="background: rgba(0, 0, 0, 0.4); border: 1px solid {border_color}; border-radius: 16px; padding: 30px; text-align: center; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5); margin-bottom: 20px;">
                        <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 2px; color: {border_color}; margin-bottom: 10px;">{selected_model_name} Output</div>
                        <h1 style="font-size: 3.5rem; margin: 0; color: #fff;">{grade}</h1>
                        <div style="margin-top: 20px; display: flex; justify-content: center; gap: 40px;">
                            <div>
                                <div style="color: #888; font-size: 0.8rem;">CONFIDENCE</div>
                                <div style="font-size: 1.5rem; font-weight: 700; color: {border_color};">{conf:.1f}%</div>
                            </div>
                            <div>
                                <div style="color: #888; font-size: 0.8rem;">INFERENCE TIME</div>
                                <div style="font-size: 1.5rem; font-weight: 700;">{latency:.0f} ms</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Low Confidence Warning Trigger
                    if conf < 60:
                        st.warning("⚠️ **Low Confidence Warning:** The model is unsure. This leaf may contain anomalous defects or require manual expert review.")

                    # Actionable Insights based on Grade
                    insights = {
                        "Grade A": "Premium quality leaf. Recommended for high-end product manufacturing. Market value is optimal.",
                        "Grade B": "Standard quality leaf. Suitable for general commercial use. Minor blemishes detected.",
                        "Grade C": "Low quality or damaged leaf. Recommended for alternative extraction or lower-tier products."
                    }
                    
                    st.info(f"💡 **Industrial Recommendation:** {insights.get(grade, '')}")

                    # Sleek HTML/CSS Progress Bars instead of Matplotlib
                    st.markdown("#### Class Probability Distribution")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    probs = {name: float(p)*100 for name, p in zip(CLASS_NAMES, preds[0])}
                    for class_name, prob in probs.items():
                        bar_color = "#00E676" if class_name == grade else "#444"
                        text_color = "#fff" if class_name == grade else "#888"
                        st.markdown(f"""
                        <div style="margin-bottom: 15px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span style="color: {text_color}; font-size: 0.95rem; font-weight: 600;">{class_name}</span>
                                <span style="color: {text_color}; font-size: 0.95rem;">{prob:.1f}%</span>
                            </div>
                            <div style="width: 100%; background-color: #222; border-radius: 8px; height: 12px; overflow: hidden;">
                                <div style="width: {prob}%; background-color: {bar_color}; height: 100%; border-radius: 8px; transition: width 0.5s ease-in-out;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # --- MODE 2: MODEL COMPARISON ---
                else:
                    st.markdown("Running comprehensive analysis across all architectures...")
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
                    
                    time.sleep(0.3)
                    my_bar.empty()
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Enhanced Comparison Display
                    for res in results:
                        is_hybrid = "Hybrid" in res['Model']
                        border = "2px solid #00E676" if is_hybrid else "1px solid #333"
                        bg = "rgba(0, 230, 118, 0.05)" if is_hybrid else "#111"
                        badge = '<span style="background:#00E676; color:#000; font-size:0.7rem; padding: 2px 8px; border-radius: 10px; margin-left:10px; font-weight:bold;">RECOMMENDED</span>' if is_hybrid else ''
                        
                        st.markdown(f"""
                        <div style="background: {bg}; padding: 20px; border-radius: 12px; border: {border}; margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="color: #888; font-size: 0.85rem; text-transform: uppercase;">{res['Model']} {badge}</div>
                                <div style="font-size: 1.5rem; font-weight: bold; color: #fff; margin-top: 5px;">{res['Grade']}</div>
                            </div>
                            <div style="text-align: right; display: flex; gap: 20px;">
                                <div>
                                    <div style="color: #888; font-size: 0.7rem;">CONFIDENCE</div>
                                    <div style="color: {'#00E676' if is_hybrid else '#fff'}; font-size: 1.2rem; font-weight: bold;">{res['Conf']:.1f}%</div>
                                </div>
                                <div>
                                    <div style="color: #888; font-size: 0.7rem;">LATENCY</div>
                                    <div style="color: #fff; font-size: 1.2rem; font-weight: bold;">{res['Latency']:.0f} ms</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        else:
            st.info("👈 Please select an image source to begin the automated grading process.")


# ==========================================
# PAGE: RESEARCH
# ==========================================
elif st.session_state.page == "Research":
    st.markdown("# 📘 Research & Methodology")
    st.markdown("*Demystifying the AI behind Automated Tobacco Leaf Grading*")
    
    # 1. THE PROBLEM & OBJECTIVE
    st.markdown("""
    <div class="glass-card" style="border-left: 4px solid #00E676; margin-bottom: 20px;">
    <h3>The Industry Challenge</h3>
    <p style="color:#E0E0E0; line-height:1.7;">
    Tobacco leaf grading plays an important role in determining the market value and quality of tobacco products[cite: 21]. However, in most agricultural industries today, this critical process is still performed manually by human experts who visually inspect leaves based on color, texture, and visible defects[cite: 22]. This manual approach has major limitations: it is highly subjective, labor-intensive, and prone to inconsistency due to human fatigue and varying lighting conditions[cite: 23, 70, 71]. Our objective was to engineer an automated, scalable solution that removes human error and delivers reliable grading suitable for real-time industrial deployment[cite: 27, 72].
    </p>
    </div>
    """, unsafe_allow_html=True)

    # 2. THE HYBRID ARCHITECTURE
    st.markdown("### 🧠 The 'Two-Expert' AI Architecture")
    st.markdown("""
    <p style="color:#aaa; line-height:1.6; margin-bottom: 20px;">
    Handcrafted features and basic machine learning models (like SVM or KNN) often fail to capture the fine-grained differences required for accurate grading[cite: 56, 60]. To solve this, we developed a Hybrid Deep Learning system[cite: 24]. You can think of this as a "Two-Expert" committee, where two distinct types of artificial intelligence evaluate the leaf simultaneously before combining their findings.
    </p>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#00E676">🔍 Expert 1: The Texture Detective</h4>
            <p style="color:#aaa; font-size:0.9rem"><b>Convolutional Neural Network (EfficientNetB0)</b></p>
            <p style="color:#E0E0E0; line-height:1.6">
            The first branch of our model uses a highly optimized CNN[cite: 25]. Think of this expert as holding a magnifying glass to the leaf. It specializes in extracting localized, high-frequency texture features[cite: 37]. It scans the surface for microscopic details: the health of the veins, the presence of blemishes, spots, and subtle color variations[cite: 38]. By utilizing transfer learning with a pre-trained EfficientNet backbone, the model already understands basic visual concepts, allowing it to learn the specific traits of tobacco leaves with high efficiency[cite: 39, 80].
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#00E676">📐 Expert 2: The Big Picture Analyst</h4>
            <p style="color:#aaa; font-size:0.9rem"><b>Vision Transformer (ViT)</b></p>
            <p style="color:#E0E0E0; line-height:1.6">
            While the CNN obsesses over the tiny details, our Vision Transformer takes a step back to view the global context[cite: 25]. Instead of scanning pixel-by-pixel, the ViT slices the image into a grid of small patches[cite: 90]. It utilizes a "self-attention mechanism" to understand how these different patches relate to one another mathematically[cite: 41]. This allows the model to comprehend the overall geometry, physical structure, and global shape of the leaf[cite: 42], ensuring it understands the macro-level quality just as well as the micro-level texture.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. FUSION & TRAINING
    st.markdown("""
    <div class="glass-card" style="border: 1px solid rgba(0, 230, 118, 0.3); margin-bottom: 20px;">
        <h4 style="color:#00E676">🤝 Feature Fusion & Model Training</h4>
        <p style="color:#E0E0E0; line-height:1.7;">
        Neither expert is perfect on their own, but together they form a highly accurate system. The local texture vectors from the CNN and the global structural vectors from the ViT are concatenated into a dense neural layer[cite: 90]. To prevent the AI from over-relying on any single feature, we apply a dropout regularization of 0.5[cite: 81, 90]. The model then outputs a final probability distribution to classify the leaf into Grade A (High Quality), Grade B (Moderate Quality), or Grade C (Low/Damaged)[cite: 84, 85, 86]. 
        <br><br>
        To ensure our model is robust enough for a messy factory environment, we trained it on a balanced dataset using extensive data augmentation[cite: 26, 81]. By digitally flipping, rotating, and zooming the training images, we forced the AI to recognize the true physical properties of the leaves regardless of how they are positioned on a conveyor belt.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 4. RESULTS
    st.markdown("### 📊 Performance & Results")
    st.markdown("""
    <p style="color:#E0E0E0; line-height:1.7;">
    Rigorous evaluation proved that combining these architectures achieves higher accuracy compared to individual models[cite: 47]. When tested on our validation dataset, the models performed as follows:
    </p>
    """, unsafe_allow_html=True)

    # Create a nice layout for the results stats
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px; text-align: center;">
            <p style="color:#aaa; margin:0;">Standalone ViT</p>
            <h2 style="color:#fff; margin:0;">65.19%</h2>
            <p style="font-size: 0.8rem; color:#888;">Struggled with small dataset [cite: 116]</p>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px; text-align: center;">
            <p style="color:#aaa; margin:0;">Standalone CNN</p>
            <h2 style="color:#fff; margin:0;">84.44%</h2>
            <p style="font-size: 0.8rem; color:#888;">Moderate performance [cite: 115]</p>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown("""
        <div style="background: rgba(0,230,118,0.1); border: 1px solid #00E676; padding: 20px; border-radius: 10px; text-align: center;">
            <p style="color:#00E676; font-weight:bold; margin:0;">Hybrid Model</p>
            <h2 style="color:#00E676; margin:0;">93.33%</h2>
            <p style="font-size: 0.8rem; color:#00E676;">Highest Accuracy [cite: 114]</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <br>
    <p style="color:#E0E0E0; line-height:1.7;">
    <b>Inference Time Analysis:</b> For industrial deployment, speed is just as crucial as accuracy. We measured the prediction latency of each model. As expected, the standalone Vision Transformer is the fastest, and the CNN operates at a moderate speed[cite: 122, 123]. Because the Hybrid model processes data through two separate branches simultaneously, it takes slightly more time to generate a prediction[cite: 124]. However, this inference still occurs in a fraction of a second—a negligible tradeoff for a massive 9% jump in absolute accuracy, making it a highly viable solution for real-time agricultural automation.
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