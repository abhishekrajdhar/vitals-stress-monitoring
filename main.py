import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
from scipy.signal import butter, filtfilt, welch, find_peaks
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import os

#config
TARGET_FS = 30.0
BUFFER_SECONDS = 15
BUFFER_SIZE = int(TARGET_FS * BUFFER_SECONDS)
MIN_SECONDS_FOR_EST = 5.0
BANDPASS = (0.7, 3.5)  # Hz
SPO2_MIN = 80
SPO2_MAX = 100

# Streamlit setup

st.set_page_config(layout="wide", page_title="Contactless Vitals & Stress Dashboard")

# UI layout
st.title("Contactless Vitals & Stress Monitoring Dashboard")
cols = st.columns([2, 1])
left = cols[0]
right = cols[1]

# Controls
with right:
    start_button = st.button("Start")
    stop_button = st.button("Stop")
    save_csv = st.checkbox("Save vitals to CSV", value=False)
    csv_name = st.text_input("CSV filename", value="vitals_data.csv")

# Placeholders
video_placeholder = left.empty()
ppg_placeholder = left.empty()

hr_card = right.empty()
sdnn_card = right.empty()
rmssd_card = right.empty()
stress_card = right.empty()
spo2_card = right.empty()
snr_card = right.empty()

message_placeholder = right.empty()

#gignal processing functions
def bandpass_filter(sig, fs, low, high, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, sig)

def pos_from_rgb(rgb_signal):
    X = np.transpose(rgb_signal)
    mean = X.mean(axis=1, keepdims=True)
    Xn = X/(mean+1e-8) - 1
    S = np.array([[0,1,-1],[-2,1,1]])
    P = S @ Xn
    alpha = (np.std(P[0])+1e-8)/(np.std(P[1])+1e-8)
    h = P[0] - alpha * P[1]
    return (h - h.mean())/(h.std()+1e-8)

def compute_hr(sig, fs):
    f, P = welch(sig, fs=fs, nperseg=min(256, len(sig)))
    mask = (f>=0.7)&(f<=3.5)
    if np.any(mask):
        peak = f[mask][np.argmax(P[mask])]
        return peak*60
    return None

def detect_beats(sig, fs):
    peaks, _ = find_peaks(sig, distance=int(0.4*fs), prominence=0.2)
    return peaks/fs

def compute_hrv(rr_ms):
    if len(rr_ms)<2:
        return None,None
    sdnn = np.std(rr_ms)
    rmssd = np.sqrt(np.mean(np.diff(rr_ms)**2))
    return sdnn,rmssd

#  SKIN MASK 

def skin_mask_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array([0,40,50])
    upper = np.array([50,200,255])
    mask = cv2.inRange(hsv,lower,upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,(5,5))
    return mask.astype(bool)

# SNR COMPUTATION

def compute_snr(sig, fs):
    f, P = welch(sig, fs=fs, nperseg=min(256,len(sig)))
    band = np.sum(P[(f>=0.7)&(f<=3.5)])
    total = np.sum(P)+1e-8
    return float(band/total)

# SpO2 ESTIMATION

def estimate_spo2(rgb_buffer):
    arr = np.array(rgb_buffer)
    if arr.shape[0]<10:
        return None
    ac = np.std(arr,axis=0)
    dc = np.mean(arr,axis=0)
    ratio = (ac[0]/dc[0])/(ac[1]/dc[1])
    spo2 = 110 - 25*ratio
    spo2 = max(SPO2_MIN,min(SPO2_MAX,spo2))
    return round(spo2,1)

# Stress Index COMPUTATION

def compute_stress_index(rr_ms):
    """Baevsky Stress Index"""
    if len(rr_ms)<5:
        return None
    rr = np.array(rr_ms)
    hist, bins = np.histogram(rr, bins=20)
    idx = np.argmax(hist)
    Mo = (bins[idx]+bins[idx+1])/2
    AMo = (hist[idx]/len(rr))*100
    MxDMn = rr.max() - rr.min()
    if Mo==0 or MxDMn==0:
        return None
    SI = AMo/(2*Mo*MxDMn)
    return round(SI,2)

def classify_stress(si):
    if si is None:
        return "No Data", "⚪"
    if si < 80:
        return "Low Stress", "🟢"
    elif si < 150:
        return "Normal Stress", "🟡"
    else:
        return "High Stress", "🔴"

#  MULTI-ROI EXTRACTION 

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1)

def extract_multi_roi(frame):
    h,w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return frame, None

    lm = result.multi_face_landmarks[0]
    pts = np.array([(int(p.x*w), int(p.y*h)) for p in lm.landmark])

    annotated = frame.copy()

    # Face bounding box
    x1,y1 = pts[:,0].min(), pts[:,1].min()
    x2,y2 = pts[:,0].max(), pts[:,1].max()
    cv2.rectangle(annotated,(x1,y1),(x2,y2),(255,0,0),2)

    rois=[]

    # FOREHEAD
    nose = 1
    nx,ny = pts[nose]
    fw,fh=int(w*0.18),int(h*0.10)
    fx1 = max(0,nx-fw//2)
    fy1 = max(0,ny-int(1.8*fh))
    forehead = rgb[fy1:fy1+fh, fx1:fx1+fw]
    cv2.rectangle(annotated,(fx1,fy1),(fx1+fw,fy1+fh),(0,255,0),2)

    # LEFT CHEEK
    lx,ly = pts[234]
    cw,ch=int(w*0.15),int(h*0.12)
    lc = rgb[ly-ch//2:ly+ch//2, lx-cw//2:lx+cw//2]
    cv2.rectangle(annotated,(lx-cw//2,ly-ch//2),(lx+cw//2,ly+ch//2),(0,255,255),2)

    # RIGHT CHEEK
    rx,ry = pts[454]
    rc = rgb[ry-ch//2:ry+ch//2, rx-cw//2:rx+cw//2]
    cv2.rectangle(annotated,(rx-cw//2,ry-ch//2),(rx+cw//2,ry+ch//2),(255,255,0),2)

    for roi in [forehead, lc, rc]:
        if roi is None or roi.size==0:
            continue
        mask = skin_mask_hsv(roi)
        skin_pixels = roi[mask]
        if skin_pixels.size==0:
            mean_rgb = roi.reshape(-1,3).mean(axis=0)
        else:
            mean_rgb = skin_pixels.mean(axis=0)
        rois.append(mean_rgb)

    if len(rois)==0:
        return annotated, None

    return annotated, np.mean(rois,axis=0)

# Main application logic

if "running" not in st.session_state:
    st.session_state.running = False

rgb_buffer = deque(maxlen=BUFFER_SIZE)
time_buffer = deque(maxlen=BUFFER_SIZE)
ppg_buffer = deque(maxlen=BUFFER_SIZE)

log_rows = []

if start_button:
    st.session_state.running = True
    st.session_state.cap = cv2.VideoCapture(0)
if stop_button:
    st.session_state.running = False
    if "cap" in st.session_state:
        st.session_state.cap.release()

if st.session_state.running:
    cap = st.session_state.cap

    fig, ax = plt.subplots(figsize=(6,2))
    ax.set_title("Live PPG Waveform")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    line, = ax.plot([])

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            continue

        annotated, roi_mean = extract_multi_roi(frame)

        t_now = time.time()
        if roi_mean is not None:
            rgb_buffer.append(roi_mean)
            time_buffer.append(t_now)
        elif len(rgb_buffer)>0:
            rgb_buffer.append(rgb_buffer[-1])
            time_buffer.append(t_now)

        duration = time_buffer[-1]-time_buffer[0] if len(time_buffer)>1 else 0
        hr=sdnn=rmssd=spo2=stress=snr=None

        if duration>=MIN_SECONDS_FOR_EST:
            arr=np.array(rgb_buffer)
            sig = pos_from_rgb(arr)
            sig_f = bandpass_filter(sig, TARGET_FS, BANDPASS[0],BANDPASS[1])
            ppg_buffer.append(sig_f[-1])

            hr = compute_hr(sig_f, TARGET_FS)
            beats = detect_beats(sig_f,TARGET_FS)

            if len(beats)>=2:
                rr = np.diff(beats)*1000
                sdnn, rmssd = compute_hrv(rr)
                stress = compute_stress_index(rr)

            snr = compute_snr(sig_f, TARGET_FS)
            spo2 = estimate_spo2(rgb_buffer)

            # log
            log_rows.append({
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "hr": hr, "sdnn": sdnn, "rmssd": rmssd,
                "spo2": spo2, "stress": stress, "snr": snr
            })

        video_placeholder.image(cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB))

        # Plots
        ax.cla()
        ax.grid(True)
        ax.set_title("Live PPG Waveform")
        if len(ppg_buffer)>0:
            sig=np.array(ppg_buffer)[-300:]
            ax.plot(sig)
        ppg_placeholder.pyplot(fig)

        # METRICS
        hr_card.metric("Heart Rate (bpm)", f"{hr:.1f}" if hr else "--")
        sdnn_card.metric("SDNN (ms)", f"{sdnn:.2f}" if sdnn else "--")
        rmssd_card.metric("RMSSD (ms)", f"{rmssd:.2f}" if rmssd else "--")

        # Stress classification
        stress_level, emoji = classify_stress(stress)
        stress_card.metric("Stress Level", f"{stress_level} {emoji}")

        spo2_card.metric("SpO₂ (est.)", f"{spo2}%" if spo2 else "--")

        snr_card.metric("SNR", f"{snr:.2f}" if snr else "--")

        # Save CSV in batches
        if save_csv and len(log_rows)>=30:
            df = pd.DataFrame(log_rows)
            if not os.path.exists(csv_name):
                df.to_csv(csv_name,index=False)
            else:
                df.to_csv(csv_name,mode="a",header=False,index=False)
            log_rows=[]

        time.sleep(1/TARGET_FS)

else:
    st.info("Press **Start** to begin monitoring.")

