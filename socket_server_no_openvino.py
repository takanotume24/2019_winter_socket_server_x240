
import socket
import sys
import cv2
import pickle
import numpy as np
import struct  # new
import zlib
import threading

motor = 0
xmin_to_send = 0
xmax_to_send = 0


def calc_x_center(xmin, xmax):
    return int((xmin + xmax) / 2)


if __name__ == "__main__":

    HOST = ''
    PORT_IMAGE = 8485
    PORT_MOTOR = 1234

    socket_image = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_motor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    print('Socket created')

    socket_image.bind((HOST, PORT_IMAGE))
    socket_motor.bind((HOST, PORT_MOTOR))

    print('Socket bind complete')
    socket_image.listen(10)
    socket_motor.listen(10)

    print('Socket now listening')

    conn_image, _ = socket_image.accept()
    conn_motor, _ = socket_motor.accept()

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
            # print("Recv: {}".format(len(data)))
            data += conn_image.recv(4096)

        # print("Done Recv: {}".format(len(data)))
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        # print("mssg_size: {}".format(msg_size))
        while len(data) < msg_size:
            data += conn_image.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        #if frame is None:
       #     raise Exception('Image not found!')
        # Prepare input blob and perform an inference.
       # blob = cv2.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv2.CV_8U)
       # net.setInput(blob)
       # out = net.forward()

        #max_area = 0
        # faces = out.reshape(-1, 7)
        # Draw detected faces on the frame.
        #for detection in faces:
        #    confidence = float(detection[2])
        #    xmin = int(detection[3] * frame.shape[1])
        #    ymin = int(detection[4] * frame.shape[0])
        #    xmax = int(detection[5] * frame.shape[1])
        #    ymax = int(detection[6] * frame.shape[0])
        #    if confidence > 0.5:
        #        cv2.rectangle(frame, (xmin, ymin),
        #                      (xmax, ymax), color=(0, 255, 0))
        #        area = (xmax - xmin) * (ymax - ymin)
        #        if area > max_area:
        #            max_area = area
        #            xmin_to_send = xmin
        #            xmax_to_send = xmax

        #if(max_area > 0):
        #    x_center = calc_x_center(xmin_to_send, xmax_to_send)
        #else:
        #    x_center = 0
        x_center = 0
        print(x_center)

        conn_motor.sendall(str(x_center).zfill(4).encode())
        conn_motor.recv(1024)
        max_area = 0
        fliped_frame = cv2.flip(frame, 1)
        # resized_frame = cv2.resize(fliped_frame, (frame.shape[1] * 2, frame.shape[0] * 2))
        cv2.namedWindow("camera", cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
        cv2.imshow("camera",fliped_frame)
        conn_image.sendall(b'ok')
        cv2.waitKey(1)
