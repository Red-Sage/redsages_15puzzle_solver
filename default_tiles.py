import pathlib
import cv2
import numpy as np
from image_operations import get_sift_key_points


class DefaultTiles():

    def __init__(self):

        self.default_im_size = (65, 65)
        self.puzzle_size = 15  # tiles not including the blank
        im_array = np.zeros((*self.default_im_size, self.puzzle_size),
                            dtype=np.uint8
                            )
        key_points = []
        descriptors = []

        tile_path = pathlib.Path().cwd() / 'resources'
        for i in range(self.puzzle_size):
            # Import and process image excluding the 16 tile
            file_name = tile_path / f'{i+1}.png'
            tile_im = cv2.imread(str(file_name))
            tile_im = cv2.resize(tile_im, self.default_im_size)
            tile_im = cv2.cvtColor(tile_im, cv2.COLOR_BGR2GRAY)
            cv2.fastNlMeansDenoising(tile_im, tile_im, 5)

            # Calculate key points
            kp, des = get_sift_key_points(tile_im)

            key_points.append(kp)
            descriptors.append(des)
            im_array[:, :, i] = tile_im

        self.im_array = im_array
        self.key_points = key_points
        self.descriptors = descriptors

    def get_tile_image(self, idx):

        if idx > 0 and idx <= self.im_array.shape[2]:
            return self.im_array[:, :, idx-1]
        else:
            raise ValueError(f'Requested image {idx} does not exist')

    def get_tile_kp_des(self, idx):

        if idx > 0 and idx <= self.im_array.shape[2]:
            return self.key_points[idx-1], self.descriptors[idx-1]
        else:
            raise ValueError(f'Requested image {idx} does not exist')

    @property
    def num_tiles(self):
        return self.im_array.shape[2]

    @property
    def puzzle_shape(self):
        return [4, 4]

