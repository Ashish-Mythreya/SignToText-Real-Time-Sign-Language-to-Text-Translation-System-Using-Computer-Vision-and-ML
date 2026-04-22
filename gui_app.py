import cv2
import mediapipe as mp
import numpy as np
import pickle
import customtkinter as ctk
from PIL import Image, ImageTk

# --- Theme Configuration ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue") # We will override colors manually for the 'tech' look

class TechySignApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("SignToText v1.0 // ASL INTERPRETER")
        self.geometry("1100x700")
        self.configure(fg_color="#0a0a0a") # Deep black background

        # Mediapipe & Model Setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
        self.mp_drawing = mp.solutions.drawing_utils
        
        try:
            with open('asl_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
        except:
            self.model = None

        self.label_map = {i: chr(65 + i) for i in range(26)}

        # --- UI LAYOUT ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. Left Sidebar (Control & Brand)
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0, fg_color="#121212", border_width=1, border_color="#1f1f1f")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.brand = ctk.CTkLabel(self.sidebar, text="SignToText", font=ctk.CTkFont(family="Orbitron", size=24, weight="bold"), text_color="#00d4ff")
        self.brand.pack(pady=(40, 10))
        
        self.line = ctk.CTkFrame(self.sidebar, height=2, fg_color="#00d4ff", width=150)
        self.line.pack(pady=10)

        # 2. Prediction Display (The Big Hero)
        self.pred_container = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.pred_container.pack(expand=True)
        
        self.pred_label_title = ctk.CTkLabel(self.pred_container, text="CURRENT SIGN", font=ctk.CTkFont(size=12), text_color="gray")
        self.pred_label_title.pack()
        
        self.prediction_box = ctk.CTkLabel(self.pred_container, text="-", font=ctk.CTkFont(size=120, weight="bold"), text_color="#00d4ff")
        self.prediction_box.pack()

        # 3. Status Indicators
        self.status_ball = ctk.CTkLabel(self.sidebar, text="● SYSTEM ACTIVE", font=ctk.CTkFont(size=10), text_color="#57bb8a")
        self.status_ball.pack(pady=20)

        # 4. Main Viewport (Video)
        self.viewport = ctk.CTkFrame(self, fg_color="#0f0f0f", corner_radius=20, border_width=2, border_color="#1f1f1f")
        self.viewport.grid(row=0, column=1, padx=30, pady=30, sticky="nsew")

        self.video_label = ctk.CTkLabel(self.viewport, text="")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        # Start Capture
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def extract_landmarks(self, hand_landmarks):
        # Using the NEW Normalization Logic for accuracy
        base_x = hand_landmarks.landmark[0].x
        base_y = hand_landmarks.landmark[0].y
        palm_size = ((hand_landmarks.landmark[9].x - base_x)**2 + (hand_landmarks.landmark[9].y - base_y)**2)**0.5
        if palm_size == 0: palm_size = 1

        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([(lm.x - base_x)/palm_size, (lm.y - base_y)/palm_size, lm.z])
        return np.array(landmarks)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            # Customizing the Mediapipe drawing style for a 'tech' look
            # Blue lines, white dots
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            predicted_label = "-"
            
            if results.multi_hand_landmarks and self.model is not None:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw a custom 'techy' hand skeleton
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 212, 255), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
                    )
                    
                    features = self.extract_landmarks(hand_landmarks)
                    try:
                        prediction = self.model.predict([features])[0]
                        predicted_label = self.label_map.get(prediction, "?")
                    except: pass

            # Update UI
            self.prediction_box.configure(text=predicted_label)
            
            # Rounded corners for the video feed
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Resize image to fit viewport while maintaining aspect ratio
            img_tk = ImageTk.PhotoImage(image=img)
            self.video_label.img_tk = img_tk
            self.video_label.configure(image=img_tk)

        self.after(10, self.update_frame)

    def on_closing(self):
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = TechySignApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()