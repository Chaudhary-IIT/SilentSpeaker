import pandas as pd

# 1️⃣ Load the CSV
df = pd.read_csv(
    r"C:\Users\ADITY\Desktop\hackathon\SilentSpeaker\data\train_manifest.csv",
    header=None,
    names=["video_path", "text"],  # Force correct column names
)

# 2️⃣ Replace old -> new base paths
old_base = r"C:\Users\Naman\Desktop\Files\kagglehub\datasets\mohamedbentalb\lipreading-dataset\versions\1\data"
new_base = r"C:\Users\ADITY\Desktop\data"

df["video_path"] = df["video_path"].str.replace(old_base, new_base, regex=False)

# 3️⃣ Save corrected CSV
out_path = r"C:\Users\ADITY\Desktop\hackathon\SilentSpeaker\data\train_manifest_fixed.csv"
df.to_csv(out_path, index=False)
print(f"✅ Fixed manifest saved successfully at:\n{out_path}")
