# 🧠🏟️ Marsad - مَرْصَد (AI Stadium Intelligence System)

- Marsad is a **hackathon prototype concept** exploring how computer vision and language models could support stadium operations through crowd-density analysis, fan assistance, and player-movement insights. The repository describes proposed capabilities; it is not a deployed stadium-safety system.
  - 🚀 AI-powered Smart Stadium & Fan Experience platform  
  - 🎯 Focused on safety, engagement, and performance  
  - 🏆 Submitted for the AI League Hackathon 2025  

---

## Saudi Arabia Vision 2030 & FIFA 2034 Alignment

The concept was framed around future smart-venue priorities, including large-scale events such as **FIFA World Cup 2034**. This is project positioning, not an official endorsement or deployment:

- 🏟️ Transforming stadiums into intelligent, secure, and data-driven ecosystems  
- 🤖 Leveraging AI to enhance visitor safety, optimize operations, and elevate entertainment  
- Saudi Arabia supporting the smart cities and digital infrastructure tracks of Vision 2030  
- 🌍 Offering a scalable solution for global sporting events and smart venue management  

Marsad demonstrates a possible direction for sports technology; operational readiness, safety performance, and real-world effectiveness have not been validated.

---

## 🚀 Key Features

### 🛡️ Crowd Safety & Emergency Detection
- Prototype crowd-density analysis using YOLOv8
- Proposed alerts for crowd-risk scenarios; not validated for emergency response
- Conceptual alert workflow for stadium staff

### 🎉 Fan Engagement
- AI-powered smart cheering based on crowd energy
- NLP chatbot for stadium assistance
- Personalized highlight reels based on fan preferences

### ⚽ Player Performance Analysis
- Pose estimation using MediaPipe/OpenPose
- Heatmaps of player movement and fatigue estimation
- AI tactical assistant for coaching decisions

---

## 🧠 Tech Stack
- Python, PyTorch, TensorFlow, OpenCV
- YOLOv8 for crowd detection
- MediaPipe / OpenPose for player tracking
- Transformers / RNN for NLP modules
- Flask + Streamlit for front-end simulation

---

## 📁 Repository Structure

```bash
Marsad_AI/
├── dataset/              # Placeholder for raw data
├── models/               # Saved model files
├── notebooks/            # Jupyter notebooks for experiments
├── outputs/               # Inference outputs and demos
├── src/                  # Source code
│   ├── preprocess_all.py
│   ├── train.py
│   ├── inference.py
│   ├── pose_analysis.py
│   ├── chatbot.py
│   ├── heatmap.py
│   └── utils.py
├── Marsad_AI_League_Presentation_2025.pptx # Presentation
├── requirements.txt      # All dependencies
├── README.md             # This file
└── data.yaml             # Dataset configuration (YOLO)
```

---

## 📸 Screenshots & Demo
> ![visual_demo.png](outputs%2Fvisual_demo.png)


## 📬 Contact
Created by [Abdullah Hani Abdellatif Al-Shobaki](https://www.linkedin.com/in/abdulahals/)
