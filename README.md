# Attendance‑Via‑Face‑Recognition

## 🚀 Project Overview  
This project is a real‑time attendance system leveraging face recognition.  
It enables capturing faces of people (e.g., students, employees), encoding them, and during runtime detecting and recognising multiple faces to mark attendance automatically.

## 🎯 Features  
- Collect faces of individuals and store their encodings.  
- Real‑time face detection & recognition via webcam or video feed.  
- Mark attendance automatically and log it in a CSV.  
- Use of face‑recognition libraries and models for reliable identification.  
- Easy-to‑use Python scripts for training, recognising and logging attendance.

## 🧰 Tech Stack  
- **Language**: Python  
- **Libraries**: e.g., `face_recognition`, `opencv‑python`, `numpy` (see `requirements.txt`).  
- **Files in repo**:  
  - `collect_faces.py` – script to capture face images / data.  
  - `train_encodings.py` – script to encode face data into `encodings.pkl`.  
  - `recognize.py` – script to run live face detection & mark attendance.  
  - `attendance.csv` – CSV file where attendance logs are maintained.  
  - `encodings.pkl` – pickled face encodings for known individuals.  
- **License**: MIT (see `LICENSE` file) — you are free to use/modify under that license.

## 📁 Folder/File Structure  
```
/
├── attendance.csv         ← Attendance log file  
├── collect_faces.py       ← Script to collect face images/data  
├── encodings.pkl          ← Saved face encodings of known users  
├── recognize.py           ← Script to perform real‑time recognition & attendance  
├── requirements.txt       ← Library dependencies  
├── train_encodings.py     ← Script to build encodings from collected data  
└── LICENSE                ← MIT License file  
```

## ✅ Setup & Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/SubhansuPradhan/Attendance‑Via‑Face‑Recognition.git
   cd Attendance‑Via‑Face‑Recognition
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
3. (Optional) Collect face images for your participants by running:  
   ```bash
   python collect_faces.py --name "Your name" --count 50
   ```  
   Follow prompts to capture images for each individual.  
4. Encode collected faces:  
   ```bash
   python train_encodings.py --dataset dataset --encodings encodings.pkl --model-name VGG-Face
   ```  
   This produces `encodings.pkl` with face encodings of known people.  
5. Run live recognition to mark attendance:  
   ```bash
   python recognize.py --encodings encodings.pkl --model-name VGG-Face --tolerance 25 --scale 0.5 --output attendance.csv --detector-backend opencv
   ```  
   A video feed opens; when a known face is detected, the attendance is logged into `attendance.csv`.

## 🧪 Usage Tips  
- Ensure your webcam or video source is properly configured.  
- If lighting is poor, recognition accuracy may drop.  
- Use clear, frontal‑facing images for better training.  
- Regularly back up the `attendance.csv` file if you wish to archive attendance.  
- For multiple faces at once: the system handles more than one face simultaneously (if using suitable libraries and model).  
- If you receive errors: check versions of `face_recognition`, `dlib` and `opencv`. Some require specific OS builds.

## 🔧 Configuration & Deployment  
- You can extend this to use live video feeds from IP cameras or network streams.  
- You may integrate with a database instead of a CSV for larger scale.  
- You can build a web interface or automate daily attendance email reports.  
- If deploying on a server, ensure you handle camera access and GPU/CPU performance accordingly.

## 👥 Contribution  
Contributions are welcome!  
- Fork the repo  
- Create a feature branch (`git checkout -b feature/YourFeature`)  
- Commit your changes (`git commit -m 'Add some feature'`)  
- Push your branch (`git push origin feature/YourFeature`)  
- Open a Pull Request to discuss your changes.

## 📝 Future Enhancements  
- Integrate with biometric access control (e.g., door‑unlock when face recognised).  
- Add facial recognition with mask / occlusion handling.  
- Improve accuracy using deep learning models (e.g., FaceNet, ArcFace).  
- Add reporting dashboards (daily, monthly attendance analytics).  
- Add mobile/web app for scanning via smartphone camera and marking attendance remotely.

## 📄 License  
This project is licensed under the [MIT License](LICENSE) — see the `LICENSE` file for details.

## 🙏 Acknowledgements  
Thanks to all the open‑source libraries and community contributions enabling face recognition in Python. Big shout‑out to the developers of `face_recognition`, `dlib`, `OpenCV`, and other support packages.

---

> _“Computer vision is like seeing with code — this project brings attendance into the future.”_
