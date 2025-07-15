#just listens
import socket

# Incoming from camera
UDP_IP_IN = "0.0.0.0"
UDP_PORT_IN = 8888

# Outgoing to serial controller
UDP_IP_OUT = "127.0.0.1"
UDP_PORT_OUT = 9999

# Socket to listen
sock_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_in.bind((UDP_IP_IN, UDP_PORT_IN))

# Socket to forward
sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"Listening on {UDP_IP_IN}:{UDP_PORT_IN}... Forwarding to {UDP_IP_OUT}:{UDP_PORT_OUT}")

while True:
    data, addr = sock_in.recvfrom(1024)
    message = data.decode().strip()
    print(f"Camera says: {message}")

    # Forward locally
    sock_out.sendto(message.encode(), (UDP_IP_OUT, UDP_PORT_OUT))
