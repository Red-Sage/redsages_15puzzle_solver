from default_tiles import DefaultTiles

class PuzzleSolver():

    def __init__(self, vid, pool):
        self.vid = vid
        self.pool = pool
        self.tiles = DefaultTiles

    def find_tiles(self)