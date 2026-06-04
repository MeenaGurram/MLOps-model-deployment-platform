import subprocess

THRESHOLD = 0.30

drift_score = 0.42

if drift_score > THRESHOLD:

    print(
        "Drift detected. Retraining..."
    )

    subprocess.run(
        ["python", "src/train.py"]
    )