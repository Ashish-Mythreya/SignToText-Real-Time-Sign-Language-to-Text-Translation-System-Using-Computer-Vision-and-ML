# SignToText: Real-Time Sign Language to Text Translation System

**Malla Reddy University** | **Department of Computer Science and Engineering** 
**Batch No:** 22 

---

## 📝 Project Overview
SignToText is a real-time translation system designed to bridge the communication gap between the deaf/hard-of-hearing community and the general public.By utilizing computer vision to track hand movements via a standard webcam, the system eliminates the need for expensive wearable sensors. It employs machine learning to intelligently classify complex hand landmarks into accurate text characters, ensuring a fluid and inclusive communication experience through optimized, low-latency performance.

## 👥 Team Members 
* **Enagandhula Ashish Mythreya** (-2211CS010159) 
* **Devarapalli Nandhitha** (-2211CS010149) 
* **Gali Bhanu Sivani** (-2211CS010172) 
* **Chigurupati Sai Sukumar** (-2211CS010096) 

**Under the Esteemed Guidance of:** Mrs. V. Preethi Reddy 

---

## 🛠️ Technical Stack 
* **Language**: Python 3.8+ 
* **Computer Vision**: OpenCV (Video capturing and frame processing) 
* **Feature Extraction**: MediaPipe (21 hand landmark detection and 3D coordinate mapping) 
* **Machine Learning**: Scikit-learn (Random Forest implementation) 
* **Data Handling**: NumPy and Pandas 

## 📊 Performance 
* **Accuracy**: 98%+ on the test dataset 
* **Inference Speed**: Average processing time of <30ms per frame
* **Frame Rate**: Maintains a stable 30 FPS, ensuring a lag-free user experience 

---

## 🚀 Methodology 
1. **Data Acquisition**: Captured custom hand gesture datasets representing the alphabet/numbers via webcam.
2. **Landmark Mapping**: Used MediaPipe to convert visual frames into 21 key coordinate points.
3. **Feature Normalization**: Applied scaling to make coordinates relative to the palm center, ensuring recognition works at various distances.
4. **Model Training**: Employed a Random Forest Classifier with multiple decision trees for robust classification.

## 🔮 Future Scope 
* Scaling to support continuous sign language recognition for full sentences.
* Integrating Natural Language Processing (NLP) to help in correcting grammar and context.
* Deployment as a lightweight mobile application using TensorFlow Lite.

---
*"Empowering Communication Through Technology"* 
