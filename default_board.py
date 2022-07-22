import pathlib
import cv2

from image_operations import get_sift_key_points


class DefaultBoard():

    def __init__(self):

        board_path = pathlib.Path().cwd() / 'resources/board_2-0-2.png'
        board_im = cv2.imread(str(board_path))
        self._board_area = [[35, 104], [315, 383]]

        board_im = cv2.cvtColor(board_im, cv2.COLOR_BGR2GRAY)
        board_im = cv2.fastNlMeansDenoising(board_im, board_im, 5)

        kp, des = get_sift_key_points(board_im)

        self.board_im = board_im
        self.key_points = kp
        self.descriptors = des

    def get_board_kp_des(self):
        return self.key_points, self.descriptors

    @property
    def board_area(self):
        return self._board_area


        