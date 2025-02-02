# VioletHacks - WorkOut AI

## 📌 Overview
VioletHacks - WorkOut AI is an AI-powered fitness assistant designed to generate personalized workout routines and provide real-time form analysis using computer vision. This project was developed during the VioletHacks hackathon and aims to offer an intelligent and adaptive approach to fitness.

## 🚀 Features
- 🔹 AI-generated workout plans based on user input
- 🔹 Real-time squat form analysis using a TensorFlow Lite model, MoveNet Lighting Model
- 🔹 Connects to Gemini API for a workout planner
- 🔹 User-friendly interface for selecting muscle groups and workout intensity
- 🔹 Automated process management to free up necessary ports
- 🔹 Flask-based web application

## 🛠️ Tech Stack
- **Backend:** Python, Flask
- **AI/ML:** TensorFlow Lite, Gemini API
- **Computer Vision:** OpenCV
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Local machine, Flask server

## 📥 Installation & Setup

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip
- Flask
- OpenCV
- TensorFlow Lite
- Git

### Clone the Repository
```bash
$ git clone https://github.com/Hayden32-D/WorkOut_AI.git
$ cd WorkOut_AI/VioletHacks
```

### Backend Setup
```bash
$ pip install -r requirements.txt
$ python -m flask --app run main (runs the code)
```

## 🎯 Usage
1. Run the Flask backend.
2. Navigate to `http://127.0.0.1:5000` in your browser.
3. Choose muscle groups and intensity to generate a workout plan.
4. Use FormWatcher to analyze squat form in real-time.
5. View feedback and improve your form.

## 🔬 AI Model
The AI utilizes a **TensorFlow Lite** model to analyze squat form. Key angles (hip, knee) are extracted from webcam input, and a score is assigned based on proper form adherence.

## 📌 Future Improvements
- Integration with additional exercises for form tracking
- Advanced workout customization based on past performance
- Mobile app compatibility
- Add more lifts to the camera vision

## 🤝 Contributors
- **Hayden Douglas** - Lead Developer
- **Keller Bice** - Lead Developer
- **Rohan Solanki** - Lead Developer

## 📞 Contact
For any inquiries, feel free to reach out:
- 📧 Email: [Haydendouglas32@icloud.com]
- 📧 Email: [rohans@vt.edu]
- 📧 Email: [kece05@vt.edu]

---

💡 *Contributions are welcome! Feel free to fork the repo and submit a PR.*
