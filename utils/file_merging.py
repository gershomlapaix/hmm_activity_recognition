import pandas as pd
import os
import glob

# Folder containing your CSV files
folder_path = r"C:\Users\LENOVO\Documents\La paix\ALU\ML-techniques-II\formative-2\dataset\new\test"   # change to your folder

# Find all accelerometer files
accel_files = glob.glob(os.path.join(folder_path, "*accel*.csv"))

for accel_file in accel_files:

    gyro_file = accel_file.replace("accel", "gyro")

    if os.path.exists(gyro_file):

        accel = pd.read_csv(accel_file)
        gyro = pd.read_csv(gyro_file)

        # Rename columns
        accel = accel.rename(columns={
            "x": "accel_x",
            "y": "accel_y",
            "z": "accel_z"
        })

        gyro = gyro.rename(columns={
            "x": "gyro_x",
            "y": "gyro_y",
            "z": "gyro_z"
        })

        # Keep only gyro sensor values
        gyro = gyro[["gyro_x", "gyro_y", "gyro_z"]]

        merged = pd.concat([accel, gyro], axis=1)

        output = accel_file.replace("accel", "")

        merged.to_csv(output, index=False)

        print("Merged:", output)