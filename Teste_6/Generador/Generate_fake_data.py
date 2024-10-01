import csv
import random
from datetime import datetime, timedelta
import os

def generate_ip():
    return f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"

def generate_timestamp():
    start_date = datetime(2023, 1, 1)
    random_date = start_date + timedelta(days=random.randint(0, 365))
    return random_date.strftime("%Y-%m-%d %H:%M:%S")

def generate_row(is_secure):
    row = {
        "timestamp": generate_timestamp(),
        "src_ip": generate_ip(),
        "dst_ip": generate_ip(),
        "protocol": random.choice(["TCP", "UDP", "ICMP"]),
        "src_port": random.randint(1, 65535),
        "dst_port": random.randint(1, 65535),
        "packets": random.randint(1, 1000),
        "bytes": random.randint(64, 1000000),
        "duration": round(random.uniform(0.1, 300.0), 2),
        "attack_type": "Benign" if is_secure else random.choice(["DoS", "Probe", "R2L", "U2R"])
    }
    return row

def generate_fake_data(filename, num_rows=10000):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ["timestamp", "src_ip", "dst_ip", "protocol", "src_port", "dst_port", 
                      "packets", "bytes", "duration", "attack_type"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for _ in range(num_rows):
            is_secure = random.random() < 0.7  # 70% chance of being secure
            row = generate_row(is_secure)
            writer.writerow(row)

if __name__ == "__main__":
    generate_fake_data("../data/fake_network_data.csv")
    print("Fake data generated and saved to ../data/fake_network_data.csv")