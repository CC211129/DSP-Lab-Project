import numpy as np

def read_frame(filename, frame_number, width, height):
    """
    :param filename: yuv input data stream
    :param frame_number: frame number in sequence. First frame: frame_number=0
    :param width: width of frame in pixels
    :param height: height of frame in pixels
    :return: y, u, v: Numpy uint8 array for each color component
    """
    bytes_per_sample = 1  # 8-bit data requires one byte per color sample
    # Open the input file
    with open(filename, "rb") as f:
        # Skip the first frames until frame_number is reached
        # factor 1.5 for color format 420 (444: factor 3)
        f.seek(int(width * height * 1.5 * frame_number * bytes_per_sample))

        y = np.fromfile(f, dtype=np.uint8, count=int(height*width))
        y = y.reshape(height, width)

        count = width//2 * height//2
        u = np.fromfile(f, dtype=np.uint8, count=count)
        v = np.fromfile(f, dtype=np.uint8, count=count)
        u = u.reshape(height//2, width//2)
        v = v.reshape(height//2, width//2)

    return y, u, v

  