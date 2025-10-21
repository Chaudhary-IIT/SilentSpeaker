# models.py
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, TimeDistributed, Flatten,
    Bidirectional, LSTM, Dense, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

class LipReaderModel:
    def __init__(self, input_shape=(75, 64, 64, 1), vocab_size=28):
        """
        input_shape: (T, H, W, C) e.g. 75 frames, 64x64 grayscale width.
        vocab_size: Number of characters + 1 for CTC blank token.
        """
        self.input_shape = input_shape
        self.vocab_size = vocab_size
        self.model = self._build_model()
        self._compile_model()

    def _build_model(self):
        inputs = Input(shape=self.input_shape, name="input_video")

        # 3D CNN for spatio-temporal feature extraction
        x = Conv3D(32, kernel_size=(3, 5, 5), activation="relu", padding="same")(inputs)
        x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x)

        x = Conv3D(64, kernel_size=(3, 3, 3), activation="relu", padding="same")(x)
        x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x)

        # Flatten spatial dims, keep temporal dim
        x = TimeDistributed(Flatten())(x)

        # Bidirectional LSTM to learn temporal dependencies
        x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(x)
        x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(x)

        # Output dense layer with softmax per time step (seq length, vocab_size+1)
        y_pred = Dense(self.vocab_size + 1, activation="softmax", name="y_pred")(x)

        model = Model(inputs=inputs, outputs=y_pred, name="LipReaderModel")
        return model

    def _compile_model(self):
        # CTC loss requires special handling as a Lambda layer
        def ctc_lambda_func(args):
            y_pred, labels, input_length, label_length = args
            # y_pred shape: (batch, time, vocab+1)
            y_pred = y_pred[:, :, :]
            return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

        labels = Input(name="the_labels", shape=[None], dtype="int32")
        input_length = Input(name="input_length", shape=[1], dtype="int32")
        label_length = Input(name="label_length", shape=[1], dtype="int32")

        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")(
            [self.model.output, labels, input_length, label_length]
        )

        self.train_model = Model(
            inputs=[self.model.input, labels, input_length, label_length], outputs=loss_out
        )
        self.train_model.compile(optimizer="adam", loss={"ctc": lambda y_true, y_pred: y_pred})

    def predict(self, video_path):
        """
        Predict text from an input video file.
        1. Extract frames & crop lips.
        2. Preprocess frames to model input (resize, grayscale, normalize).
        3. Feed into model.predict().
        4. Decode CTC output to text.
        """
        frames = self.extract_lip_frames(video_path)
        preprocessed = self.preprocess_frames(frames)
        pred = self.model.predict(np.expand_dims(preprocessed, axis=0))
        text = self.decode_ctc(pred[0])
        return text

    def extract_lip_frames(self, video_path, max_frames=75):
        """
        Extract lip-region frames from video using MediaPipe FaceMesh.
        Returns list of grayscale frames resized to 64x64.
        """
        mp_face_mesh = mp.solutions.face_mesh
        cap = cv2.VideoCapture(video_path)
        lip_frames = []
        with mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

            while len(lip_frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    h, w, _ = frame.shape
                    # Lip landmarks indices from mediapipe
                    lip_landmark_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61] 
                    lip_points = []
                    for idx in lip_landmark_indices:
                        x = int(face_landmarks.landmark[idx].x * w)
                        y = int(face_landmarks.landmark[idx].y * h)
                        lip_points.append((x, y))
                    lip_points = np.array(lip_points)
                    x, y, w_box, h_box = cv2.boundingRect(lip_points)
                    lip_crop = frame[y:y+h_box, x:x+w_box]
                    gray = cv2.cvtColor(lip_crop, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (64, 64))
                    normalized = resized / 255.0
                    lip_frames.append(normalized)
                else:
                    # Append blank frame if no face found
                    lip_frames.append(np.zeros((64, 64)))

            cap.release()

        # Pad frames if less than max_frames
        while len(lip_frames) < max_frames:
            lip_frames.append(np.zeros((64,64)))

        return np.array(lip_frames)

    def preprocess_frames(self, frames):
        # Shape (T,H,W), convert to (T,H,W,1) for model input
        return np.expand_dims(frames, axis=-1)

    def decode_ctc(self, pred):
        """
        Decode softmax outputs from model to text using CTC greedy decoder.
        pred shape: (T, vocab_size+1)
        """
        character_list = "abcdefghijklmnopqrstuvwxyz0123456789,.' "
        out_best = list(np.argmax(pred, axis=1))
        out_best = [k for k, _ in itertools.groupby(out_best)]  # remove duplicates
        out_best = [c for c in out_best if c != self.vocab_size]  # remove blank

        pred_text = ''.join([character_list[i] if i < len(character_list) else '' for i in out_best])
        return pred_text

    def print_summary(self):
        self.model.summary()


# Optional Grad-CAM utils for explainability
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import itertools


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_heatmap(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    overlayed = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlayed

