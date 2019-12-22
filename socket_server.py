
import socket
import sys
import cv2
import pickle
import numpy as np
import struct  # new
import zlib
if __name__ == "__main__":
    HOST = ''
    PORT = 8485

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')

    s.bind((HOST, PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')

    conn, addr = s.accept()

    data = b""
    payload_size = struct.calcsize(">L")
    print("payload_size: {}".format(payload_size))

    # Load the model.
    net = cv2.dnn.readNet('face-detection-adas-0001.xml',
                          'face-detection-adas-0001.bin')
    # Specify target device.
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    while True:
        while len(data) < payload_size:
            print("Recv: {}".format(len(data)))
            data += conn.recv(4096)

        print("Done Recv: {}".format(len(data)))
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        print("msg_size: {}".format(msg_size))
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        if frame is None:
            raise Exception('Image not found!')
        # Prepare input blob and perform an inference.
        blob = cv2.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv2.CV_8U)
        net.setInput(blob)
        out = net.forward()
        # Draw detected faces on the frame.
        for detection in out.reshape(-1, 7):
            confidence = float(detection[2])
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])
            if confidence > 0.5:
                cv2.rectangle(frame, (xmin, ymin),
                              (xmax, ymax), color=(0, 255, 0))

        cv2.imshow('ImageWindow', frame)
        conn.sendall(b'ok')
        cv2.waitKey(1)
