import json

# path ke file annotasi Anda
path = "annotation.json"

with open(path, "r") as f:
    data = json.load(f)

# jika formatnya selalu "train", "val", "test"
train_count = len(data.get("train", []))
val_count   = len(data.get("val", []))
test_count  = len(data.get("test", []))
total_count = train_count + val_count + test_count

print("Total samples :", total_count)
print("train         :", train_count)
print("val           :", val_count)
print("test          :", test_count)