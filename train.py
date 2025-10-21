import numpy as np
import tensorflow as tf
from models import LipReaderModel

class LipReadingDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, video_paths, transcripts, batch_size=8, max_frames=75, char_to_num=None, shuffle=True):
        self.video_paths = video_paths
        self.transcripts = transcripts
        self.batch_size = batch_size
        self.max_frames = max_frames
        self.char_to_num = char_to_num
        self.shuffle = shuffle
        self.indexes = np.arange(len(video_paths))
        self.on_epoch_end()

    def __len__(self):
        # Number of batches per epoch
        return int(np.floor(len(self.video_paths) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        video_paths_batch = [self.video_paths[k] for k in batch_indexes]
        transcripts_batch = [self.transcripts[k] for k in batch_indexes]

        X, y, input_length, label_length = self.__data_generation(video_paths_batch, transcripts_batch)
        return [X, y, input_length, label_length], np.zeros(self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_video_paths, batch_transcripts):
        # Prepare batch data
        batch_frames = []
        batch_labels = []
        batch_input_len = []
        batch_label_len = []

        for i, video_path in enumerate(batch_video_paths):
            # Extract and preprocess lip frames (must implement extract/preprocess in your code)
            frames = LipReaderModel().extract_lip_frames(video_path, max_frames=self.max_frames)
            frames = LipReaderModel().preprocess_frames(frames)

            batch_frames.append(frames)

            # Encode transcript text to int sequence using char_to_num dict
            label_seq = [self.char_to_num.get(ch, 0) for ch in batch_transcripts[i]]
            batch_labels.append(label_seq)

            batch_input_len.append(self.max_frames)
            batch_label_len.append(len(label_seq))

        # Pad labels so that all sequences have the same length
        max_label_len = max(batch_label_len)
        padded_labels = tf.keras.preprocessing.sequence.pad_sequences(
            batch_labels, maxlen=max_label_len, padding='post', value=0
        )

        X = np.array(batch_frames)
        y = np.array(padded_labels)
        input_length = np.array(batch_input_len).reshape(-1,1)
        label_length = np.array(batch_label_len).reshape(-1,1)

        return X, y, input_length, label_length


# Character mapping (make sure it matches your model's vocabulary)
character_list = "abcdefghijklmnopqrstuvwxyz0123456789,.' "
char_to_num = {ch: i for i, ch in enumerate(character_list)}

# Sample usage to fine-tune
video_paths = ['path/to/video1.mp4', 'path/to/video2.mp4', ...]
transcripts = ['hello', 'hi', ...]

batch_size = 4
data_gen = LipReadingDataGenerator(video_paths, transcripts, batch_size=batch_size, char_to_num=char_to_num)

model = LipReaderModel()  # your model class instance

# Optional: load pretrained weights if available
# model.train_model.load_weights('path_to_weights.h5')

# Train or fine-tune
model.train_model.fit(
    data_gen,
    epochs=10,
)



# Character mapping
character_list = "abcdefghijklmnopqrstuvwxyz '"
char_to_num = {ch: i for i, ch in enumerate(character_list)}

def main():
    video_paths = ['path/to/video1.mp4', 'path/to/video2.mp4', ...]  # Provide your paths here
    transcripts = ['hello', 'hi', ...]  # Corresponding transcripts

    batch_size = 4
    data_gen = LipReadingDataGenerator(video_paths, transcripts, batch_size=batch_size, char_to_num=char_to_num)

    model = LipReaderModel()
    # Optionally load pretrained weights here
    # model.train_model.load_weights('path_to_weights.h5')

    model.train_model.fit(data_gen, epochs=10)

if __name__ == "__main__":
    main()