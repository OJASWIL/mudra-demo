import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import time
import av
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(
    page_title="Mudra Vision",
    page_icon="☸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; margin: 0 !important; }
[data-testid="stAppViewContainer"] { background: #07060d; }
section[data-testid="stMain"] { background: #07060d; padding: 0 !important; }
[data-testid="stMain"] > div:first-child { padding-top: 0 !important; margin-top: 0 !important; }
div[data-testid="stVerticalBlock"] { gap: 0 !important; }

.nav-bar {
    background: #0d0c1a;
    border-bottom: 1px solid rgba(201,151,58,0.3);
    padding: 1rem 2.5rem;
    display: flex; align-items: center; justify-content: space-between;
    position: sticky; top: 0; z-index: 999;
}
.nav-logo { display: flex; align-items: center; gap: 14px; }
.nav-emblem {
    width: 44px; height: 44px; border-radius: 50%;
    background: linear-gradient(135deg, #c9973a, #e8c06a);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; color: #07060d; font-weight: 700;
    box-shadow: 0 0 20px rgba(201,151,58,0.4);
}
.nav-name { font-family:'Playfair Display',serif; font-size:1.2rem; color:#e8c06a; font-weight:700; display:block; }
.nav-sub  { font-family:'Space Mono',monospace; font-size:0.5rem; color:#5a5470; letter-spacing:0.16em; text-transform:uppercase; display:block; margin-top:2px; }
.nav-right { display:flex; align-items:center; gap:2rem; }
.nav-chip  { font-family:'Space Mono',monospace; font-size:0.68rem; color:#7a7490; padding:4px 12px; border:1px solid rgba(255,255,255,0.08); border-radius:20px; }
.nav-chip strong { color:#e8c06a; }
.live-chip { font-family:'Space Mono',monospace; font-size:0.68rem; color:#45b07c; padding:4px 14px; background:rgba(69,176,124,0.1); border:1px solid rgba(69,176,124,0.35); border-radius:20px; font-weight:700; display:flex; align-items:center; gap:6px; }
.live-pulse { width:6px; height:6px; border-radius:50%; background:#45b07c; display:inline-block; animation:pulse 1.5s ease-in-out infinite; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.3;transform:scale(0.7)} }
.gold-bar { height:1px; background:linear-gradient(90deg,transparent,#c9973a,transparent); }

.landing-wrap {
    width: 100%; padding: 5rem 2rem 4rem;
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; text-align: center;
    background: radial-gradient(ellipse 60% 50% at 50% 40%, rgba(201,151,58,0.07) 0%, transparent 70%), #07060d;
    box-sizing: border-box;
}
.land-badge {
    display: inline-flex; align-items: center; gap: 8px;
    font-family: 'Space Mono', monospace; font-size: 0.65rem;
    color: #c9973a; letter-spacing: 0.18em; text-transform: uppercase;
    border: 1px solid rgba(201,151,58,0.3); padding: 6px 18px;
    border-radius: 20px; margin-bottom: 2rem; background: rgba(201,151,58,0.05);
}
.land-h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.8rem, 6vw, 4.8rem);
    font-weight: 700; line-height: 1.08; color: #e8c06a; margin-bottom: 1.5rem;
}
.land-h1 em { font-style: italic; color: #e05252; }
.land-sub {
    font-family: 'Inter', sans-serif; font-size: 1.05rem;
    font-weight: 300; color: rgba(237,233,224,0.55);
    line-height: 1.85; max-width: 520px; margin: 0 auto 3rem;
}
.land-sub strong { color: #e8c06a; font-weight: 500; }
.stats-grid {
    display: flex; gap: 0;
    border: 1px solid rgba(201,151,58,0.2); border-radius: 8px;
    overflow: hidden; margin-bottom: 3rem; background: rgba(15,14,26,0.8);
}
.stat-cell { padding: 1.2rem 2.2rem; text-align: center; border-right: 1px solid rgba(201,151,58,0.15); }
.stat-cell:last-child { border-right: none; }
.stat-n { font-family:'Playfair Display',serif; font-size:2rem; font-weight:700; color:#e8c06a; display:block; }
.stat-l { font-family:'Space Mono',monospace; font-size:0.55rem; color:#5a5470; letter-spacing:0.15em; text-transform:uppercase; margin-top:3px; display:block; }
.model-icons { display:flex; gap:1.5rem; margin-bottom:3rem; justify-content:center; }
.model-icon-card {
    background: rgba(15,14,26,0.9); border: 1px solid rgba(201,151,58,0.15);
    border-radius: 10px; padding: 1rem 1.4rem; text-align: center; min-width: 130px;
}
.model-icon-num  { font-size:1.5rem; margin-bottom:4px; }
.model-icon-name { font-family:'Space Mono',monospace; font-size:0.62rem; color:#7a7490; display:block; }
.model-icon-acc  { font-family:'Space Mono',monospace; font-size:0.75rem; font-weight:700; display:block; margin-top:4px; }
.land-hint { font-family:'Space Mono',monospace; font-size:0.62rem; color:#4a4560; margin-top:1rem; }

/* CENTER BUTTON */
.center-btn {
    display: flex;
    justify-content: center;
    width: 100%;
    margin: 0 auto;
}

.stButton > button {
    background: linear-gradient(135deg, #c9973a, #e8b840) !important;
    color: #07060d !important; border: none !important;
    padding: 0.9rem 3rem !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.88rem !important; font-weight: 700 !important;
    letter-spacing: 0.08em !important; border-radius: 6px !important;
    box-shadow: 0 0 30px rgba(201,151,58,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { box-shadow: 0 0 40px rgba(201,151,58,0.5) !important; }

.det-wrap { padding: 1.4rem 2rem; background: #07060d; }
.section-lbl {
    font-family: 'Space Mono', monospace; font-size: 0.58rem;
    color: #5a5470; letter-spacing: 0.2em; text-transform: uppercase;
    margin-bottom: 10px; display: block;
}

[data-testid="stFileUploader"] {
    background: #12111f !important;
    border: 2px dashed rgba(201,151,58,0.3) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"] label {
    color: #7a7490 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
}

[data-testid="stWebRtcStreamer"],
div[data-testid="stWebRtcStreamer"] > div,
.css-1xarl3l, iframe { background: #07060d !important; }
.stWebRtcStreamer > div { background: #07060d !important; border: none !important; }
video {
    border: 2px solid #c9973a !important;
    border-radius: 4px !important; display: block !important;
    background: #07060d !important;
}
[data-testid="stWebRtcStreamer"] > div > div { background: #07060d !important; min-height: 0 !important; }

.mcard {
    background: #12111f; border-radius: 6px; overflow: hidden;
    margin-bottom: 12px; border: 1px solid rgba(255,255,255,0.05);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.mcard-head { padding: 0.7rem 1.1rem; display: flex; align-items: center; justify-content: space-between; }
.mcard-title { font-family:'Space Mono',monospace; font-size:0.78rem; font-weight:700; }
.mcard-acc   { font-family:'Space Mono',monospace; font-size:0.62rem; color:#5a5470; }
.mcard-body  { padding: 0.9rem 1.1rem 1.1rem; }
.mcard-lbl   { font-family:'Space Mono',monospace; font-size:0.55rem; color:#5a5470; letter-spacing:0.14em; text-transform:uppercase; margin-bottom:4px; }
.mcard-name  { font-family:'Playfair Display',serif; font-size:1.6rem; font-weight:700; color:#fff; margin:4px 0 10px; line-height:1.2; }
.mcard-conf  { font-family:'Space Mono',monospace; font-size:1.1rem; font-weight:700; margin:6px 0; }
.bar    { height:5px; background:rgba(255,255,255,0.07); border-radius:3px; overflow:hidden; margin-top:6px; }
.bar-in { height:100%; border-radius:3px; transition:width 0.3s; }

.agree-card {
    background: #12111f; border: 1px solid rgba(255,255,255,0.05);
    border-radius: 6px; padding: 1rem 1.1rem; margin-bottom: 12px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.agree-val  { font-family:'Space Mono',monospace; font-size:0.9rem; font-weight:700; margin-top:5px; }
.inf-tag    { font-family:'Space Mono',monospace; font-size:0.6rem; color:#4a4560; text-align:right; padding:4px 0; }
.det-footer {
    margin-top: 1.5rem; border-top: 1px solid rgba(201,151,58,0.15);
    padding-top: 0.8rem; display: flex; gap: 1.5rem;
    font-family: 'Space Mono', monospace; font-size: 0.6rem; color: #4a4560;
}

.red { color:#e05252; } .yel { color:#e8b840; } .grn { color:#45b07c; }
.bg-red { background:#e05252; } .bg-yel { background:#e8b840; } .bg-grn { background:#45b07c; }
.hd-red { background:rgba(224,82,82,0.12);  border-left:3px solid #e05252; }
.hd-yel { background:rgba(232,184,64,0.12); border-left:3px solid #e8b840; }
.hd-grn { background:rgba(69,176,124,0.12); border-left:3px solid #45b07c; }

.upload-img-box {
    border: 2px solid rgba(201,151,58,0.3); border-radius: 6px;
    overflow: hidden; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

NUM_CLASSES = 50
CLASS_NAMES = [
    'Alapadmam','Anjali','Aralam','Ardhachandran','Ardhapathaka',
    'Berunda','Bramaram','Chakra','Chandrakala','Chaturam',
    'Garuda','Hamsapaksha','Hamsasyam','Kangulam','Kapith',
    'Kapotham','Karkatta','Kartariswastika','Katakamukha 1','Katakamukha 2',
    'Katakamukha 3','Katakavardhana','Katrimukha','Khatva','Kilaka',
    'Kurma','Matsya','Mayura','Mrigasirsha','Mukulam',
    'Mushti','Nagabandha','Padmakosha','Pasha','Pathaka',
    'Pushpaputa','Sakata','Samputa','Sarpasirsha','Shanka',
    'Shivalinga','Shukatundam','Sikharam','Simhamukham','Suchi',
    'Swastikam','Tamarachudam','Tripathaka','Trishulam','Varaha'
]

class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*28*28,512), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512,256), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(256,num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci,co,3,padding=1), nn.BatchNorm2d(co),
                nn.ReLU(inplace=True), nn.MaxPool2d(2,2)
            )
        self.block1 = block(3,32)
        self.block2 = block(32,64)
        self.block3 = block(64,128)
        self.block4 = block(128,256)
        self.gap = nn.AdaptiveAvgPool2d((4,4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4,512), nn.ReLU(inplace=True), nn.Dropout(0.25),
            nn.Linear(512,256), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(256,num_classes)
        )
    def forward(self, x):
        for b in [self.block1,self.block2,self.block3,self.block4]: x = b(x)
        return self.classifier(self.gap(x))

def get_resnet50(num_classes):
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features,512), nn.ReLU(inplace=True), nn.Dropout(0.4),
        nn.Linear(512,256), nn.ReLU(inplace=True), nn.Dropout(0.3),
        nn.Linear(256,num_classes)
    )
    return m

@st.cache_resource
def load_models():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m1 = BaselineCNN(NUM_CLASSES).to(dev)
    m1.load_state_dict(torch.load('baseline_cnn.pth', map_location=dev, weights_only=True))
    m1.eval()
    m2 = ImprovedCNN(NUM_CLASSES).to(dev)
    m2.load_state_dict(torch.load('improved_cnn.pth', map_location=dev, weights_only=True))
    m2.eval()
    m3 = get_resnet50(NUM_CLASSES).to(dev)
    m3.load_state_dict(torch.load('resnet50_finetuned.pth', map_location=dev, weights_only=True))
    m3.eval()
    return m1, m2, m3, dev

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def run_predict(model, img_pil, dev):
    t = transform(img_pil.convert('RGB')).unsqueeze(0).to(dev)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)
        conf, pred = torch.max(probs, 1)
    return CLASS_NAMES[pred.item()], round(conf.item()*100, 1)

def card_html(num, title, acc, hd, c, bg, name, conf):
    return f"""
    <div class="mcard">
      <div class="mcard-head {hd}">
        <span class="mcard-title {c}">{num} &nbsp;{title}</span>
        <span class="mcard-acc">Val Acc: {acc}%</span>
      </div>
      <div class="mcard-body">
        <div class="mcard-lbl">Predicted Mudra</div>
        <div class="mcard-name">{name}</div>
        <div class="mcard-lbl">Confidence</div>
        <div class="mcard-conf {c}">{conf}%</div>
        <div class="bar"><div class="bar-in {bg}" style="width:{conf}%"></div></div>
      </div>
    </div>"""

def predictions_html(l1, c1, l2, c2, l3, c3, ms=0):
    preds = [l1, l2, l3]
    if l1==l2==l3:
        ag_txt = f"✓ All 3 agree — {l1}"; ag_col = "#45b07c"
    elif max(preds.count(p) for p in preds) >= 2:
        maj = max(set(preds), key=preds.count)
        ag_txt = f"⚡ 2/3 agree — {maj}"; ag_col = "#e8b840"
    else:
        ag_txt = "✗ Models disagree"; ag_col = "#e05252"
    ms_line = f'<div class="inf-tag">Inference: {ms} ms</div>' if ms else ''
    return (
        card_html("①","Baseline CNN","95.90","hd-red","red","bg-red", l1, c1) +
        card_html("②","Improved CNN","96.29","hd-yel","yel","bg-yel", l2, c2) +
        card_html("③","ResNet50 ★",  "99.34","hd-grn","grn","bg-grn", l3, c3) +
        f"""<div class="agree-card">
              <div class="section-lbl">Model Agreement</div>
              <div class="agree-val" style="color:{ag_col}">{ag_txt}</div>
            </div>{ms_line}"""
    )

class MudraProcessor(VideoProcessorBase):
    def __init__(self):
        self.model1 = self.model2 = self.model3 = self.device = None
        self.result = {'l1':'—','c1':0,'l2':'—','c2':0,'l3':'—','c3':0,'ms':0}
        self._frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self._frame_count += 1
        if self._frame_count % 3 == 0 and self.model1 is not None:
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            t0 = time.time()
            l1,c1 = run_predict(self.model1, pil, self.device)
            l2,c2 = run_predict(self.model2, pil, self.device)
            l3,c3 = run_predict(self.model3, pil, self.device)
            ms = int((time.time()-t0)*1000)
            self.result = dict(l1=l1,c1=c1,l2=l2,c2=c2,l3=l3,c3=c3,ms=ms)
            cv2.putText(img, f"ResNet50: {l3} ({c3}%)", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (69,176,124), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

if 'page' not in st.session_state:
    st.session_state.page = 'landing'
if 'mode' not in st.session_state:
    st.session_state.mode = 'upload'

page_label = 'LIVE' if st.session_state.page == 'detection' else 'READY'
st.markdown(f"""
<div class="nav-bar">
  <div class="nav-logo">
    <div class="nav-emblem">☸</div>
    <div>
      <span class="nav-name">Mudra Vision</span>
      <span class="nav-sub">Bharatanatyam Hand Gesture Recognition</span>
    </div>
  </div>
  <div class="nav-right">
    <div class="nav-chip">Models: <strong>3 Active</strong></div>
    <div class="nav-chip">Classes: <strong>50 Mudras</strong></div>
    <div class="live-chip"><span class="live-pulse"></span>{page_label}</div>
  </div>
</div>
<div class="gold-bar"></div>
""", unsafe_allow_html=True)

model1, model2, model3, device = load_models()

if st.session_state.page == 'landing':
    st.markdown("""
    <div class="landing-wrap">
      <div class="land-badge">✦ &nbsp; Real-time Recognition &nbsp; ✦</div>
      <h1 class="land-h1">Experience the <em>Art</em> of<br>Mudras</h1>
      <p class="land-sub">
        Real-time recognition of Bharatanatyam hand gestures using
        <strong>three deep learning models</strong> working in ensemble.
        Use webcam or upload a photo to begin.
      </p>
      <div class="stats-grid">
        <div class="stat-cell"><span class="stat-n">50</span><span class="stat-l">Mudra Classes</span></div>
        <div class="stat-cell"><span class="stat-n">3</span><span class="stat-l">AI Models</span></div>
        <div class="stat-cell"><span class="stat-n">99.3%</span><span class="stat-l">Top Accuracy</span></div>
        <div class="stat-cell"><span class="stat-n">Live</span><span class="stat-l">Real-time</span></div>
      </div>
      <div class="model-icons">
        <div class="model-icon-card">
          <div class="model-icon-num">①</div>
          <span class="model-icon-name">Baseline CNN</span>
          <span class="model-icon-acc red">95.90%</span>
        </div>
        <div class="model-icon-card">
          <div class="model-icon-num">②</div>
          <span class="model-icon-name">Improved CNN</span>
          <span class="model-icon-acc yel">96.29%</span>
        </div>
        <div class="model-icon-card">
          <div class="model-icon-num">③</div>
          <span class="model-icon-name">ResNet50 ★</span>
          <span class="model-icon-acc grn">99.34%</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # CENTER BUTTON — single column centered
    st.markdown('<div style="display:flex;justify-content:center;margin:0 auto;">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2.5, 1.5, 2.5])
    with col2:
        if st.button("▷   Begin Detection"):
            st.session_state.page = 'detection'
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<p class="land-hint" style="text-align:center;padding-bottom:2rem">'
        'Webcam or image upload supported  •  Video processed locally</p>',
        unsafe_allow_html=True
    )

elif st.session_state.page == 'detection':

    st.markdown('<div class="det-wrap">', unsafe_allow_html=True)
    col_cam, col_pred = st.columns([1.2, 1], gap="large")

    with col_cam:
        st.markdown('<span class="section-lbl">📷 &nbsp; Input Mode</span>', unsafe_allow_html=True)
        mode_col1, mode_col2 = st.columns(2)
        with mode_col1:
            if st.button("📤  Upload Image", use_container_width=True):
                st.session_state.mode = 'upload'
                st.rerun()
        with mode_col2:
            if st.button("🎥  Live Webcam", use_container_width=True):
                st.session_state.mode = 'webcam'
                st.rerun()

        st.markdown(
            f'<div style="font-family:Space Mono,monospace;font-size:0.62rem;color:#c9973a;margin-bottom:12px">'
            f'Mode: {"📤 Image Upload" if st.session_state.mode == "upload" else "🎥 Live Webcam"}</div>',
            unsafe_allow_html=True
        )

        if st.session_state.mode == 'upload':
            st.markdown('<span class="section-lbl">📸 &nbsp; Capture Mudra</span>', unsafe_allow_html=True)
            uploaded = st.camera_input("Take a photo", label_visibility="collapsed")
            if uploaded:
                img_pil = Image.open(uploaded)
                st.markdown('<div class="upload-img-box">', unsafe_allow_html=True)
                st.image(img_pil, use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="section-lbl">📷 &nbsp; Live Feed</span>', unsafe_allow_html=True)
            st.markdown(
                '<div style="background:#07060d; border:2px solid #c9973a; border-radius:4px; overflow:hidden;">',
                unsafe_allow_html=True
            )
            RTC_CONFIG = RTCConfiguration({"iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["turn:openrelay.metered.ca:80"],
                 "username": "openrelayproject", "credential": "openrelayproject"},
                {"urls": ["turn:openrelay.metered.ca:443"],
                 "username": "openrelayproject", "credential": "openrelayproject"},
                {"urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
                 "username": "openrelayproject", "credential": "openrelayproject"},
            ]})
            ctx = webrtc_streamer(
                key="mudra-stream",
                video_processor_factory=MudraProcessor,
                rtc_configuration=RTC_CONFIG,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)
            if ctx.video_processor:
                ctx.video_processor.model1 = model1
                ctx.video_processor.model2 = model2
                ctx.video_processor.model3 = model3
                ctx.video_processor.device = device

        st.markdown(
            f'<div style="font-family:Space Mono,monospace;font-size:0.62rem;color:#4a4560;margin-top:8px">'
            f'Device: <span style="color:#7a7490">{str(device).upper()}</span></div>',
            unsafe_allow_html=True
        )

        if st.button("■   Stop Demo"):
            st.session_state.page = 'landing'
            st.rerun()

    with col_pred:
        st.markdown('<span class="section-lbl">🔮 &nbsp; Model Predictions</span>', unsafe_allow_html=True)
        pred_box = st.empty()

        if st.session_state.mode == 'upload':
            if 'uploaded' in dir() and uploaded:
                t0 = time.time()
                l1,c1 = run_predict(model1, img_pil, device)
                l2,c2 = run_predict(model2, img_pil, device)
                l3,c3 = run_predict(model3, img_pil, device)
                ms = int((time.time()-t0)*1000)
                pred_box.markdown(predictions_html(l1,c1,l2,c2,l3,c3,ms), unsafe_allow_html=True)
            else:
                pred_box.markdown(
                    predictions_html("—",0,"—",0,"—",0) +
                    '<div style="font-family:Space Mono,monospace;font-size:0.65rem;color:#5a5470;margin-top:8px">← Take a photo to see predictions</div>',
                    unsafe_allow_html=True
                )
        else:
            if 'ctx' in dir():
                while ctx.state.playing if ctx else False:
                    if ctx.video_processor:
                        r = ctx.video_processor.result
                        pred_box.markdown(
                            predictions_html(r['l1'],r['c1'],r['l2'],r['c2'],r['l3'],r['c3'],r['ms']),
                            unsafe_allow_html=True
                        )
                    time.sleep(0.15)
            pred_box.markdown(
                predictions_html("—",0,"—",0,"—",0) +
                '<div style="font-family:Space Mono,monospace;font-size:0.65rem;color:#5a5470;margin-top:8px">← Start camera to see predictions</div>',
                unsafe_allow_html=True
            )

    st.markdown("""
    <div class="det-footer">
      <span>Bharatanatyam Mudra Classification</span>
      <span>•</span><span>Student: Ojaswi Sharma</span>
      <span>•</span><span>GPU T4 ×2</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)