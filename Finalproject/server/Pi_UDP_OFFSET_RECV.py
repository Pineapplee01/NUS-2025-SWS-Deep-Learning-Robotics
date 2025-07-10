# pi_udp_receiver.py
import socket

UDP_IP = "0.0.0.0"       # 监听所有网卡
UDP_PORT = 8888

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening on {UDP_IP}:{UDP_PORT}...")

while True:
    try:
        data, addr = sock.recvfrom(512)
        message = data.decode().strip()
        offset_str, width_str, height_str = message.split(',')
        #int
        offset = int(offset_str)
        box_width = int(width_str)
        box_height = int(height_str)

        print(f"addr:{addr}")
        print (f"raw_msg:{message}")
        print(f"offset: {offset}, box_width: {box_width}, box_height: {box_height}")


        """
        add PID there
        """


    except Exception as e:
        print(f"wrong:{e}")
