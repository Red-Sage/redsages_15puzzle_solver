from msilib.schema import ActionText
from pickle import FRAME
from traceback import FrameSummary
import cv2
from video_context import VideoCapture, VideoWrite
import multiprocessing
from image_operations import (get_sift_key_points,
                              get_transform_matrix,
                              put_text,
                              remove_key_point,
                              draw_move,
                              )
import default_tiles
import default_board
import numpy as np
import collections
from pathlib import Path
from sklearn.cluster import KMeans
from RedSages_15Puzzle_Agent.agent_q_learning import Agent_Q_Learning
from RedSages_15Puzzle_Agent.redsages_15puzzle.puzzle15 import PuzzleBoard
from time import time


def main(vid, writer):

    tiles = default_tiles.DefaultTiles()
    center_x = tiles.default_im_size[0]/2
    center_y = tiles.default_im_size[1]/2
    center = np.array([[center_x], [center_y], [1]])

    board = default_board.DefaultBoard()
    board_kp, board_des = board.get_board_kp_des()

    agent = Agent_Q_Learning()
    agent.epsilon_init = 0
    agent.load_q_table()

    num_h = 7
    x_history = []
    y_history = []
    search_order = [14, 11, 15, 12, 13, 10, 4, 7, 1, 3, 9, 2, 5, 6, 8]
    draw_labels = True
    loop_count = 0

    for i in range(tiles.num_tiles):
        x_history.append(collections.deque(maxlen=num_h))
        y_history.append(collections.deque(maxlen=num_h))

        x_history[i].append(0)
        y_history[i].append(0)

        best_x = np.zeros(tiles.num_tiles)
        best_x = best_x.reshape(-1, 1)
        best_y = np.zeros(tiles.num_tiles)
        best_y = best_y.reshape(-1, 1)
        t1 = time()

    while True:
        ret, frame = vid.read()
        color_frame = np.copy(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp, des = get_sift_key_points(frame)
        kp = list(kp)

        # Find the board and remove it's key points
        board_m = get_transform_matrix(board_kp, board_des, kp, des)
        if board_m is not None:
            board_upper_left = np.transpose([*board.board_area[0], 1])
            board_lower_right = np.transpose([*board.board_area[1], 1])
            board_upper_left = np.matmul(board_m, board_upper_left)
            board_lower_right = np.matmul(board_m, board_lower_right)
            kp, des = remove_key_point(kp,
                                       des,
                                       board_upper_left,
                                       board_lower_right,
                                       out=True,
                                       )

        if des is not None:

            m = []
            for i in search_order:

                tile_kp, tile_des = tiles.get_tile_kp_des(i)

                this_m = get_transform_matrix(tile_kp, tile_des, kp, des)

                if this_m is not None:
                    location = np.matmul(this_m, center)
                    upper_left = np.matmul(this_m, [[0], [0], [1]])
                    lower_right = np.matmul(this_m,
                                            [
                                             [tiles.default_im_size[0]],
                                             [tiles.default_im_size[1]],
                                             [1],
                                             ]
                                            )
                    x_history[i-1].append(location[0, 0])
                    y_history[i-1].append(location[1, 0])
                    kp, des = remove_key_point(kp,
                                               des,
                                               upper_left.squeeze(),
                                               lower_right.squeeze(),
                                               )

                    m.append(this_m)

                best_x[i-1] = int(np.median(x_history[i-1]))
                best_y[i-1] = int(np.median(y_history[i-1]))

                if draw_labels:
                    put_text(color_frame,
                             str(i),
                             [best_x[i-1], best_y[i-1]],
                             )

        # Use kmeans to determine the grid
        kmeans_x = KMeans(n_clusters=4).fit(best_x)
        kmeans_y = KMeans(n_clusters=4).fit(best_y)

        idx_x = np.argsort(kmeans_x.cluster_centers_.reshape(1, -1))
        idx_y = np.argsort(kmeans_y.cluster_centers_.reshape(1, -1))

        idx_x = idx_x.squeeze()
        idx_y = idx_y.squeeze()

        look_up_x = np.zeros_like(idx_x)
        look_up_y = np.zeros_like(idx_y)

        look_up_x[idx_x] = np.arange(tiles.puzzle_shape[0])
        look_up_y[idx_y] = np.arange(tiles.puzzle_shape[1])

        board_grid = np.ones(tiles.puzzle_shape)*16
        for i in range(1, tiles.num_tiles + 1):
            col = look_up_x[kmeans_x.labels_[i-1]]
            row = look_up_y[kmeans_y.labels_[i-1]]
            board_grid[row, col] = i

        # If all tiles have been found get an action
        if (len(np.unique(board_grid)) ==
           tiles.puzzle_shape[0]*tiles.puzzle_shape[1]):

            agent.puzzle = PuzzleBoard(board_grid)

            action = agent.epsilon_action()

            loc_16 = agent.puzzle.get_tile_loc(16)
            x_16 = kmeans_x.cluster_centers_[idx_x[loc_16[1]]]
            y_16 = kmeans_y.cluster_centers_[idx_y[loc_16[0]]]

            if agent.puzzle.is_complete:
                put_text(color_frame,
                         '16',
                         [x_16, y_16],
                         )
            else:
                draw_move(color_frame,
                          action,
                          int(x_16),
                          int(y_16),
                          tiles.default_im_size,
                          )

        cv2.imshow('frame', color_frame)
        writer.write(color_frame)

        loop_count += 1
        if loop_count % 25 == 0:
            print(f'Frame rate: {25/(time()-t1)} frames/sec')
            t1 = time()

        k = cv2.waitKey(1)
        if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
            break


if __name__ == "__main__":

    vid_size = (640, 480)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    numProcessors = multiprocessing.cpu_count()
    with (VideoCapture(1) as vid,
          VideoWrite('video3.avi', fourcc, 8, vid_size) as writer
          ):

        main(vid, writer)
