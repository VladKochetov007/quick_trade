class WalkForward:
    def __init__(self,
                 total_chunks: int = 10,
                 insample_chunks: int = 3,
                 outofsample_chunks: int = 1):
        assert not (total_chunks - insample_chunks) % outofsample_chunks

