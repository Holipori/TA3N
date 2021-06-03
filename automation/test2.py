
import os
import time

a = os.getpid()

file = open('/home/xinyue/TA3N/pid.txt', 'w')
file.write(str(a))
file.close()
time.sleep(5)
print(50 % (100/2) ==0)