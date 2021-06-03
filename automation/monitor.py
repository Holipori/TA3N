
import psutil
import socket
import time

# pids = psutil.pids()
# if 268605 in pids:
#     print('True')

s = socket.socket()
host = socket.gethostname()  # get local host name
port = 12345
s.connect((host,port))

while True:
    with open('/home/xinyue/TA3N/status.txt', 'r') as f:
        status = f.read()
    s.send(status.encode())
    time.sleep(10)