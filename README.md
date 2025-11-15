```markdown
# 📡 Contactless Vitals & Stress Monitoring System  
**Real-time Heart Rate, HRV, Stress Index, SpO₂ & PPG Waveform using rPPG + Multi-ROI Skin Tracking**

This project implements a complete **remote photoplethysmography (rPPG)** system that uses a webcam to estimate vital signs **without any physical sensors**.  
The application extracts subtle color changes in facial skin caused by blood flow, processes them using physiological signal algorithms, and displays:

- ❤️ Heart Rate (BPM)  
- 📈 HRV Metrics (SDNN & RMSSD)  
- 🔥 Stress Index (Baevsky)  
- 🧠 Stress Level Classification (Low / Normal / High)  
- 🫁 SpO₂ Estimation (Non-Clinical)  
- 📉 Real-time PPG waveform  
- 🎯 Signal Quality (SNR Indicator)  

👉 All of this is packaged inside an interactive **Streamlit dashboard** with live webcam video.

---

## 🚀 Demo Screenshot (Description)

**Left Panel:**  
- Live annotated video feed  
- Face bounding box  
- Multi-ROI boxes (Forehead, Left Cheek, Right Cheek)

**Right Panel:**  
- Cards for HR, SDNN, RMSSD  
- Stress Index & Stress Level (color-coded)  
- SpO₂ estimation  
- SNR indicator  
- PPG waveform graph  

---

# ⭐ Features

### ✔ Multi-ROI Pulse Extraction  
Uses 3 facial regions for more stable pulse signal:  
- Forehead  
- Left Cheek  
- Right Cheek  

### ✔ Skin Masking  
Only skin pixels are used for signal extraction → improves accuracy.

### ✔ POS Algorithm  
The **Plane-Orthogonal-to-Skin (POS)** method is used to extract the rPPG waveform from RGB variations.

### ✔ Heart Rate Estimation  
Computed from spectral peak of the filtered rPPG signal.

### ✔ HRV Metrics  
HRV derived from beat-to-beat intervals (RR intervals):  
- **SDNN** — Standard deviation of RR intervals  
- **RMSSD** — Short-term parasympathetic activity

### ✔ Stress Index (Baevsky SI)  
A validated HRV metric used for stress analysis:  
```

SI = AMo / (2 * Mo * MxDMn)

```

### ✔ Stress Level Classification  
Based on Stress Index:  
- < 80 → Low  
- 80–150 → Normal  
- > 150 → High

### ✔ SpO₂ Estimation (Non-Clinical)  
Uses the ratio-of-ratios from red and green channels.

### ✔ Signal Quality (SNR)  
Shows GOOD / MEDIUM / POOR quality based on HR-band energy.

### ✔ CSV Logging  
Automatically logs vitals every few seconds.

### ✔ Streamlit Dashboard  
Clean UI, one-click Start/Stop, and interactive live graph.

---

# 🧠 System Architecture

```

Webcam → FaceMesh (Mediapipe) → Multi-ROI Extraction
→ Skin Masking → RGB Averaging → POS Algorithm
→ Bandpass Filter → rPPG Signal → HR / HRV / SI / SpO₂
→ Streamlit Dashboard (metrics, waveform, video)

```

---

# 🏗 Multi-ROI Design

The system uses 3 regions for better tolerance to motion and lighting variations:

1. **Forehead** – best SNR, stable illumination  
2. **Left Cheek** – high blood perfusion  
3. **Right Cheek** – compensates asymmetry  

After masking non-skin pixels, the mean RGB values are averaged across ROIs for balanced signal quality.

---

# ⚙️ Algorithms Used

### **1. POS rPPG Algorithm**  
Transforms RGB time-series into orthogonal color differences:  
- Removes illumination variation  
- Amplifies pulse-related chrominance changes  

### **2. Bandpass Filtering**  
0.7–3.5 Hz → corresponding to 42–210 BPM.

### **3. HRV Computation**  
- RR intervals detected via peak detection  
- SDNN = overall variability  
- RMSSD = beat-to-beat variation  

### **4. Baevsky Stress Index**  
Combines RR mode, amplitude, and range to quantify stress.

### **5. SpO₂ Estimation** *(Heuristic / Non-clinical)*  
```

SpO2 ≈ 110 - 25 * (AC_R/DC_R) / (AC_G/DC_G)

````

---

# 🛠 Installation

### ✓ Clone the repository  
```bash
git clone https://github.com/yourusername/contactless-vitals-monitor.git
cd contactless-vitals-monitor
````

### ✓ Create & activate virtual environment (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### ✓ Install required dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser.

---

# 📁 Project Structure

```
📂 project/
 ├── app.py                 # Streamlit dashboard
 ├── requirements.txt       # Dependencies
 ├── README.md              # Documentation
 ├── vitals_data.csv        # Auto-generated logs (optional)
 └── assets/                # Images / diagrams (optional)
```

---

# 📊 Output Metrics Explained

### **HR (Heart Rate)**

Beats per minute, derived from the rPPG signal’s dominant frequency.

### **SDNN**

Reflects overall HRV → higher is healthier.

### **RMSSD**

Reflects parasympathetic activity → relaxation.

### **Stress Index (Baevsky)**

Physiological stress indicator based on HRV distribution.

### **Stress Level**

Categorized as Low / Normal / High.

### **SpO₂ Estimate**

Non-invasive estimation using color ratios.

### **SNR**

Real-time quality indicator of the rPPG signal.

---

# 📌 Limitations

* SpO₂ estimation is **non-clinical**
* Results degrade with:

  * low lighting
  * strong head movement
  * webcam auto-exposure
* Works best at 30 FPS
* Not a replacement for medical devices

---

# 💡 Future Improvements

* CNN-based robust rPPG extraction
* Deep learning SpO₂ estimation
* Breathing rate detection
* FFT power spectrum visualization
* Cloud deployable version (Streamlit Cloud)
* Support for multiple users
* Face-stabilization pipeline or Kalman filtering

---

# 🙏 Acknowledgements

* Mediapipe FaceMesh by Google
* POS rPPG algorithm from “Plane-Orthogonal-to-Skin”
* Various HRV research papers
* Streamlit for user interface

---

# 🔗 License

MIT License — free to use, modify, and distribute.

---

# 🎉 Final Notes

This project demonstrates a full real-time contactless vitals system using modern computer vision and signal-processing techniques.
The combination of rPPG, HRV analytics, stress estimation, and a Streamlit interface makes this a complete and functional wellness monitoring solution.

```

# Abhishek R. Dubey

(AI Engineer)