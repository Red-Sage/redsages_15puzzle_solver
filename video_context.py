import cv2
from contextlib import contextmanager

@contextmanager
def VideoCapture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


@contextmanager
def VideoWrite(*args, **kwargs):
    writer = cv2.VideoWriter(*args, **kwargs)
    try:
        yield writer
    finally:
        writer.release()