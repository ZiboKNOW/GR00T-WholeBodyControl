# gear_sonic/scripts/keyboard_publisher.py
import time
import zmq

ctx = zmq.Context()
pub = ctx.socket(zmq.PUB)
pub.bind("tcp://localhost:5580")
time.sleep(0.5)
print("Keyboard publisher ready. Keys: p=pause, k=start/stop, i=init pose, [/]=toggle hands, t=<prompt>")
while True:
    key = input()
    if key.startswith("t "):
        pub.send_string("prompt:" + key[2:])
        print("Sent prompt:", key[2:])
    else:
        pub.send_string(key)
        print("Sent:", key)