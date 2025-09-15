# Pneumonia Detection Streamlit App
# Modern UI with Cool Features

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io

# Set page config for icon and title
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .pneumonia-detected {
        background: linear-gradient(135deg, #f44336, #FF5722);
        color: white;
    }
    .normal-detected {
        background: linear-gradient(135deg, #4CAF50, #8BC34A);
        color: white;
    }
    .confidence-score {
        font-size: 1.2rem;
        margin-top: 10px;
        opacity: 0.9;
    }
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
        color: white;
    }
    .info-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-container {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model (update path as needed)
MODEL_PATH = '../models/pneumonia_classifier_final_model.keras'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# Bilingual dictionaries
EN = {
    'title': 'Pneumonia Detection from Chest X-rays',
    'upload': 'Upload a Chest X-ray Image (JPG/PNG)',
    'analyze': 'Analyze',
    'result': 'Prediction:',
    'normal': 'Normal',
    'pneumonia': 'Pneumonia',
    'language': 'Language',
    'gradcam': 'GradCAM Visualization',
    'error': 'Error processing image.',
    'about': 'About',
    'desc': 'This app uses a deep learning model to detect pneumonia from chest X-ray images. Powered by Keras, TensorFlow, and Streamlit. Developed by Ziad ElShazly.',
    'sample': 'Sample data instructions: Place test images in the data/ folder.',
    'loading': 'Analyzing image...',
    'home': 'Home',
    'detection': 'Detection',
    'info': 'Info',
    'main_title': 'Pneumonia Detection System',
    'welcome': 'Welcome to Pneumonia Detection System!',
    'features': 'Features:',
    'real_time': 'Real-time Detection: Upload any chest X-ray and get instant results',
    'high_accuracy': 'High Accuracy: Powered by DenseNet121 architecture with transfer learning',
    'gradcam_viz': 'GradCAM Visualization: See what the model focuses on for explainable AI',
    'bilingual': 'Bilingual UI: English and Arabic support',
    'confidence': 'Confidence',
    'upload_instructions': 'Upload a chest X-ray image to begin analysis',
    'choose_file': 'Choose an image file...',
    'prediction_complete': 'Prediction Complete!',
    'no_image': 'Please upload an image first.',
    'gradcam_title': 'GradCAM Heatmap Analysis',
    'gradcam_desc': 'This visualization shows which parts of the X-ray the AI model focused on when making its prediction.',
    'purpose': 'Purpose',
    'purpose_desc': 'This application was developed to help identify pneumonia in chest X-ray images using advanced deep learning techniques. It provides fast and accurate predictions to assist healthcare professionals.',
    'tech_stack': 'Technology Stack',
    'usage_guidelines': 'Usage Guidelines',
    'limitations': 'Limitations',
    'developer': 'Developer',
    'disclaimer': 'Disclaimer: This tool is for educational and research purposes only. Always consult healthcare professionals for medical decisions.',
    'prediction_details': 'Prediction Details',
    'class_label': 'Class',
    'model_name': 'Model'
}
AR = {
    'title': 'الكشف عن الالتهاب الرئوي من صور الأشعة السينية',
    'upload': 'رفع صورة أشعة الصدر (JPG/PNG)',
    'analyze': 'تحليل',
    'result': 'النتيجة:',
    'normal': 'سليم',
    'pneumonia': 'التهاب رئوي',
    'language': 'اللغة',
    'gradcam': 'توضيح GradCAM',
    'error': 'حدث خطأ أثناء معالجة الصورة.',
    'about': 'حول التطبيق',
    'desc': 'يستخدم هذا التطبيق نموذج تعلم عميق للكشف عن الالتهاب الرئوي من صور الأشعة السينية للصدر. تم تطويره باستخدام Keras و TensorFlow و Streamlit. المطور: زياد الشاذلي.',
    'sample': 'تعليمات بيانات العينة: ضع صور الاختبار في مجلد data/.',
    'loading': 'جاري تحليل الصورة...',
    'home': 'الرئيسية',
    'detection': 'الكشف',
    'info': 'معلومات',
    'main_title': 'نظام الكشف عن الالتهاب الرئوي',
    'welcome': 'مرحباً بك في نظام الكشف عن الالتهاب الرئوي!',
    'features': 'المميزات:',
    'real_time': 'الكشف الفوري: ارفع أي صورة أشعة للصدر واحصل على النتائج فوراً',
    'high_accuracy': 'دقة عالية: مدعوم بمعمارية DenseNet121 مع التعلم النقلي',
    'gradcam_viz': 'تصور GradCAM: شاهد ما يركز عليه النموذج للذكاء الاصطناعي القابل للتفسير',
    'bilingual': 'واجهة ثنائية اللغة: دعم الإنجليزية والعربية',
    'confidence': 'الثقة',
    'upload_instructions': 'ارفع صورة أشعة الصدر لبدء التحليل',
    'choose_file': 'اختر ملف صورة...',
    'prediction_complete': 'التنبؤ مكتمل!',
    'no_image': 'يرجى رفع صورة أولاً.',
    'gradcam_title': 'تحليل خريطة GradCAM الحرارية',
    'gradcam_desc': 'يوضح هذا التصور أي أجزاء من الأشعة السينية ركز عليها نموذج الذكاء الاصطناعي عند إجراء التنبؤ.',
    'purpose': 'الغرض',
    'purpose_desc': 'تم تطوير هذا التطبيق للمساعدة في تحديد الالتهاب الرئوي في صور الأشعة السينية للصدر باستخدام تقنيات التعلم العميق المتقدمة. يوفر تنبؤات سريعة ودقيقة لمساعدة المهنيين الصحيين.',
    'tech_stack': 'المجموعة التقنية',
    'usage_guidelines': 'إرشادات الاستخدام',
    'limitations': 'القيود',
    'developer': 'المطور',
    'disclaimer': 'إخلاء المسؤولية: هذه الأداة لأغراض تعليمية وبحثية فقط. استشر دائماً المهنيين الصحيين للقرارات الطبية.',
    'prediction_details': 'تفاصيل التنبؤ',
    'class_label': 'الفئة',
    'model_name': 'النموذج'
}

# Sidebar: Language and About
with st.sidebar:
    lang = st.selectbox('🌐 Language / اللغة', ['English', 'العربية'])
    TXT = EN if lang == 'English' else AR
    st.header(TXT['about'])
    st.info(TXT['desc'])
    st.caption(TXT['sample'])
    st.markdown('---')

# Main header
st.markdown(f'<h1 class="main-header">🩺 {TXT["main_title"]}</h1>', unsafe_allow_html=True)

# Navigation menu
selected = option_menu(
    menu_title=None,
    options=[f"🏠 {TXT['home']}", f"🔍 {TXT['detection']}", f"🔥 {TXT['gradcam']}", f"ℹ️ {TXT['info']}"],
    icons=["house", "search", "fire", "info-circle"],
    menu_icon="cast",
    default_index=1,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#FFDBBB"},
        "icon": {"color": "orange", "font-size": "18px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "left",
            "margin": "0px",
            "--hover-color": "#eee",
        },
        "nav-link-selected": {"background-color": "#1f77b4"},
    },
)

def create_confidence_gauge(confidence):
    """Create a gauge chart for confidence score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': TXT['confidence']},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate GradCAM heatmap"""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            # For binary classification, typically use index 0
            pred_index = 0
        # Ensure pred_index is a Python int for indexing
        if hasattr(pred_index, 'numpy'):
            pred_index = int(pred_index.numpy())
        else:
            pred_index = int(pred_index)
        class_channel = predictions[0][pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

if selected == f"🏠 {TXT['home']}":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        ### {TXT['welcome']} 👋
        
        This application uses a **DenseNet121-based deep learning model** to detect pneumonia from chest X-ray images.
        
        #### 🚀 {TXT['features']}
        - **{TXT['real_time']}**
        - **{TXT['high_accuracy']}**
        - **{TXT['gradcam_viz']}**
        - **{TXT['bilingual']}**
        - **Interactive UI**: Beautiful and user-friendly interface
        
        #### 🎯 How to Use:
        1. Navigate to the **{TXT['detection']}** tab
        2. Upload a chest X-ray image
        3. View the prediction results with confidence scores
        4. Check **{TXT['gradcam']}** tab for visual explanations
        
        ---
        *Built with ❤️ using Streamlit and TensorFlow by Ziad ElShazly*
        """)

elif selected == f"🔍 {TXT['detection']}":
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown(f"### 📤 {TXT['upload']}")
        st.markdown(TXT['upload_instructions'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            TXT['choose_file'],
            type=['jpg', 'jpeg', 'png'],
            help=TXT['upload_instructions']
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Add prediction button
            if st.button(f"🔍 {TXT['analyze']}", type="primary"):
                with st.spinner(TXT['loading']):
                    img = image.resize((224, 224))
                    arr = np.array(img) / 255.0
                    arr = np.expand_dims(arr, axis=0)
                    pred = model.predict(arr)[0][0]
                    is_pneumonia = pred > 0.5
                    confidence = pred * 100 if is_pneumonia else (1 - pred) * 100
                    label = TXT['pneumonia'] if is_pneumonia else TXT['normal']
                    
                    # Store results in session state
                    st.session_state.prediction_made = True
                    st.session_state.is_pneumonia = is_pneumonia
                    st.session_state.confidence = confidence
                    st.session_state.label = label
                    st.session_state.last_img = img
                    st.session_state.last_arr = arr
    
    with col2:
        if hasattr(st.session_state, 'prediction_made') and st.session_state.prediction_made:
            is_pneumonia = st.session_state.is_pneumonia
            confidence = st.session_state.confidence
            label = st.session_state.label
            
            # Display prediction result
            if is_pneumonia:
                st.markdown(f'''
                <div class="prediction-box pneumonia-detected">
                    🦠 {TXT['pneumonia'].upper()} DETECTED
                    <div class="confidence-score">{TXT['confidence']}: {confidence:.1f}%</div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="prediction-box normal-detected">
                    ✅ {TXT['normal'].upper()} DETECTED
                    <div class="confidence-score">{TXT['confidence']}: {confidence:.1f}%</div>
                </div>
                ''', unsafe_allow_html=True)
            
            # Display confidence gauge
            st.plotly_chart(create_confidence_gauge(confidence), use_container_width=True)
            
            # Additional information
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown(f"### 📈 {TXT['prediction_details']}")
            st.markdown(f"**{TXT['class_label']}**: {label}")
            st.markdown(f"**{TXT['confidence']}**: {confidence:.2f}%")
            st.markdown(f"**{TXT['model_name']}**: DenseNet121 Transfer Learning")
            st.markdown('</div>', unsafe_allow_html=True)

elif selected == f"🔥 {TXT['gradcam']}":
    st.markdown(f"### 🔥 {TXT['gradcam_title']}")
    st.markdown(TXT['gradcam_desc'])
    
    if hasattr(st.session_state, 'last_arr') and st.session_state.last_arr is not None:
        # Find last conv layer name
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
        
        if last_conv_layer_name:
            heatmap = make_gradcam_heatmap(st.session_state.last_arr, model, last_conv_layer_name)
            
            # Create visualization
            fig, ax = plt.subplots()
            ax.imshow(st.session_state.last_img)
            ax.imshow(heatmap, cmap='jet', alpha=0.5)
            ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            st.image(buf, caption=TXT['gradcam'], use_column_width=True)
            plt.close(fig)
            
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("### 🔍 What is GradCAM?")
            st.markdown("GradCAM (Gradient-weighted Class Activation Mapping) shows which parts of the image the model focuses on when making predictions. Red areas indicate high importance.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info('GradCAM not available for this model architecture.')
    else:
        st.info(TXT['no_image'])

elif selected == f"ℹ️ {TXT['info']}":
    st.markdown(f"### ℹ️ {TXT['about']}")
    
    st.markdown(f"""
    #### 🎯 {TXT['purpose']}
    {TXT['purpose_desc']}
    
    #### 🛠️ {TXT['tech_stack']}
    - **Frontend**: Streamlit
    - **Backend**: TensorFlow/Keras
    - **Model**: DenseNet121 with Transfer Learning
    - **Image Processing**: PIL
    - **Visualization**: Plotly, GradCAM, Matplotlib
    
    #### 📝 {TXT['usage_guidelines']}
    - Upload clear chest X-ray images
    - Ensure good image quality and resolution
    - X-ray should show clear lung structure
    - Supported formats: JPG, JPEG, PNG
    
    #### ⚠️ {TXT['limitations']}
    - Designed for binary pneumonia detection (Normal vs Pneumonia)
    - Performance may vary with image quality and acquisition settings
    - Not a substitute for professional medical diagnosis
    - Should be used as a screening tool only
    
    #### 👨‍💻 {TXT['developer']}
    **Ziad Mahmoud ElShazly**
    - Medical AI Enthusiast
    - GitHub: [github.com/ziadelshazly](https://github.com/ziadelshazly)
    
    ---
    *If you encounter any issues or have suggestions, please feel free to reach out!*
    
    **{TXT['disclaimer']}**
    """)