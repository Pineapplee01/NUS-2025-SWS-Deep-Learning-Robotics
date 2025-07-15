import subprocess

DEST_IP = "172.20.10.4"
PORT = 5001

cmd = [
    "sudo",
    "libcamera-vid",
    "-t", "0",
    "--inline",
    "--codec", "h264",
    "-o", f"udp://{DEST_IP}:{PORT}"
]

process = subprocess.Popen(cmd)

try:
    process.wait()
except KeyboardInterrupt:
    process.terminate()
    print("Stopped streaming")
