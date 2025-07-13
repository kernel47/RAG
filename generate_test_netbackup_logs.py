import os
import random
from datetime import datetime, timedelta
from pathlib import Path

def generate_netbackup_logs(output_dir="test_netbackup_logs"):
    root_dir = Path(output_dir)
    if root_dir.exists():
        import shutil
        shutil.rmtree(root_dir)
    root_dir.mkdir(parents=True)

    # Directories
    dirs = [
        root_dir / "jobs",
        root_dir / "images",
        root_dir / "policies",
        root_dir / "db" / "error"
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Simulated clients
    clients = [f"euvrli{str(i).zfill(4)}" for i in range(1, 1001)]
    policies = ["win_clients", "linux_clients", "nas_share"]

    # 1. Generate job.list
    job_list_file = root_dir / "jobs" / "job.list"
    def random_job_line(job_id, client):
        policy = random.choice(policies)
        status = random.choice(["0", "1", "2"])
        date = datetime(2025, 4, 1) + timedelta(days=random.randint(0, 90), minutes=random.randint(0, 1440))
        start = date
        end = start + timedelta(minutes=random.randint(10, 60))
        type_ = random.choice(["Full", "Incremental", "Differential"])
        size = random.choice(["1.8GB", "250MB", "5.2GB", "980MB", "3.1GB"])
        files = random.randint(100, 15000)
        return f"{job_id},{client},{policy},{status},{start.strftime('%Y-%m-%d %H:%M')},{end.strftime('%Y-%m-%d %H:%M')},{type_},{size},{files}\n"

    with open(job_list_file, "w") as f:
        for i in range(50000):
            client = random.choice(clients + ["nas-eu-west", "nas-eu-south", "nas-eu-central"])
            f.write(random_job_line(300000 + i, client))

    # 2. Generate image .f files
    for client in clients[:200]:
        image_dir = root_dir / "images" / client
        image_dir.mkdir(parents=True, exist_ok=True)
        image_file = image_dir / f"{1689345123 + random.randint(0, 10000)}.f"
        with open(image_file, "w") as f:
            for i in range(random.randint(5, 15)):
                f.write(f"/opt/app/file_{i}.tar.gz\n")

    # 3. Generate policies
    policy_files = {
        "win_clients": {
            "type": "MS-Windows",
            "clients": clients[:100],
            "include": ["C:\\Users\\", "C:\\Program Files\\", "D:\\"],
            "schedule": "Full every Sunday 01:00"
        },
        "linux_clients": {
            "type": "Standard",
            "clients": clients[100:200],
            "include": ["/etc/", "/opt/", "/home/"],
            "schedule": "Incremental daily 02:00"
        },
        "nas_share": {
            "type": "NDMP",
            "clients": ["nas-eu-west", "nas-eu-central", "nas-eu-south"],
            "include": ["/share1/", "/share2/"],
            "schedule": "Differential every 3 days"
        }
    }

    for policy_name, content in policy_files.items():
        policy_path = root_dir / "policies" / policy_name
        with open(policy_path, "w") as f:
            f.write(f"Policy Name: {policy_name}\n")
            f.write(f"Type: {content['type']}\n")
            f.write("Clients:\n")
            for c in content["clients"]:
                f.write(f"  {c}\n")
            f.write("Include List:\n")
            for path in content["include"]:
                f.write(f"{path}\n")
            f.write(f"Schedule: {content['schedule']}\n")

    # 4. Generate error logs
    error_clients = random.sample(clients, 100)
    for client in error_clients:
        error_log_path = root_dir / "db" / "error" / f"{client}.log"
        with open(error_log_path, "w") as f:
            for _ in range(random.randint(1, 5)):
                ts = datetime(2025, 5, 15, 1, random.randint(0, 59)).strftime("%Y-%m-%d %H:%M:%S")
                status = random.choice(["ERR", "WARN"])
                message = random.choice([
                    "backup failed, status 13 (file read failed)",
                    "retrying backup in 10 min",
                    "media write error",
                    "network timeout during backup"
                ])
                f.write(f"{ts} {status}: {message}\n")

if __name__ == "__main__":
    generate_netbackup_logs()
