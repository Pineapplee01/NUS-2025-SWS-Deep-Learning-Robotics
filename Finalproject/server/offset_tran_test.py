import socket

#offset transmit
Pi_UDP_IP = "172.20.10.12"
OFFSET_PORT = 8888
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

offset = 5005
box_width = 30
box_height = 10


msg = f"{offset},{box_width},{box_height}"
print(msg)
sock.sendto(msg.encode(), (Pi_UDP_IP, OFFSET_PORT) )
