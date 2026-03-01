import numpy as np


def compute_ssd(block1, block2):
    """
    Sum of Squared Differences between two blocks.

    :param block1:  block from current frame
    :param block2:  candidate block from reference frame
    :return:  SSD value
    """
    diff = block1.astype(np.int32) - block2.astype(np.int32)
    return int(np.sum(diff * diff))


def compute_sad(block1, block2):
    """
    Sum of Absolute Differences between two blocks.

    :param block1: block from current frame
    :param block2: candidate block from reference frame
    :return:  SAD value
    """
    return int(np.sum(np.abs(block1.astype(np.int32) - block2.astype(np.int32))))


def motion_estimation(cur_frame, ref_frame, blocksize, search_range,
                      distance_metric='ssd'):
    """
    Full-search block matching (luminance only).
    For each block, all candidates in the search window are evaluated in one
    vectorised numpy operation.

    :param cur_frame: current frame to be predicted
    :param ref_frame: reference frame (previous frame)
    :param blocksize: side length of each square block in pixels (e.g. 4)
    :param search_range: maximum displacement in ±pixels (e.g. 16)
    :param distance_metric: 'ssd' or 'sad'
    :return: mvs  motion vectors [dm, dn] per block
    """
    height, width = cur_frame.shape
    B  = blocksize
    sr = search_range

    num_blocks_h = height // B
    num_blocks_w = width  // B
    mvs = np.zeros((num_blocks_h, num_blocks_w, 2), dtype=np.int32)

    cur_f = cur_frame.astype(np.int32)
    ref_f = ref_frame.astype(np.int32)

    for i in range(num_blocks_h):
        y = i * B
        for j in range(num_blocks_w):
            x = j * B

            dy_min = max(0,          y - sr)
            dy_max = min(height - B, y + sr)
            dx_min = max(0,          x - sr)
            dx_max = min(width  - B, x + sr)

            n_dy = dy_max - dy_min + 1
            n_dx = dx_max - dx_min + 1

            # Build (n_dy, n_dx, B, B) candidate array via advanced indexing
            row_idx = (np.arange(n_dy)[:, None, None, None] + dy_min
                       + np.arange(B)[None, None, :, None])
            col_idx = (np.arange(n_dx)[None, :, None, None] + dx_min
                       + np.arange(B)[None, None, None, :])

            candidates = ref_f[row_idx, col_idx]       # (n_dy, n_dx, B, B)
            cur_block  = cur_f[y:y+B, x:x+B]           # (B, B)
            diff       = cur_block - candidates         # broadcast -> (n_dy, n_dx, B, B)

            if distance_metric == 'ssd':
                costs = np.sum(diff * diff, axis=(2, 3))
            else:
                costs = np.sum(np.abs(diff), axis=(2, 3))

            flat_idx         = np.argmin(costs)
            best_dn, best_dm = np.unravel_index(flat_idx, costs.shape)

            mvs[i, j, 0] = (dx_min + best_dm) - x   # horizontal displacement dm
            mvs[i, j, 1] = (dy_min + best_dn) - y   # vertical   displacement dn

    return mvs


def motion_compensation(ref_frame, blocksize, mvs):
    """
    Generate predicted frame by copying blocks from the reference frame
    according to the motion vectors.

    :param ref_frame:  reference frame
    :param blocksize:  block side length in pixels
    :param mvs:        motion vectors [dm, dn]
    :return:           predicted frame
    """
    height, width          = ref_frame.shape
    B                      = blocksize
    num_blocks_h, num_blocks_w = mvs.shape[:2]
    pred_frame = np.zeros((height, width), dtype=np.uint8)

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            y, x   = i * B, j * B
            dm, dn = int(mvs[i, j, 0]), int(mvs[i, j, 1])
            pred_frame[y:y+B, x:x+B] = ref_frame[y+dn:y+dn+B, x+dm:x+dm+B]

    return pred_frame


def predict_frame(cur_frame, ref_frame, blocksize, search_range,
                  distance_metric='ssd'):
    """
    Full-search motion-compensated prediction for one frame.

    :param cur_frame: frame to be predicted
    :param ref_frame: previous frame used as reference
    :param blocksize: block side length in pixels (e.g. 4)
    :param search_range: search window half-size in pixels (e.g. 16)
    :param distance_metric: 'ssd' or 'sad'
    :return: predicted frame
    """
    mvs  = motion_estimation(cur_frame, ref_frame, blocksize,
                             search_range, distance_metric)
    pred = motion_compensation(ref_frame, blocksize, mvs)
    return pred


def motion_estimation_three_step(cur_frame, ref_frame, blocksize, search_range,
                                 distance_metric='ssd'):
    """
    Three-Step Search (TSS) block matching.
    Step size starts at the largest power-of-2 <= search_range and halves
    each iteration; 8 neighbours are tested per step.

    :param cur_frame: current frame
    :param ref_frame: reference frame
    :param blocksize: block side length in pixels
    :param search_range: maximum search displacement in +-pixels
    :param distance_metric: 'ssd' or 'sad'
    :return: mvs motion vectors [dm, dn] 
             total_positions total number of candidate positions evaluated
    """
    import math
    height, width = cur_frame.shape
    B  = blocksize
    sr = search_range

    num_blocks_h = height // B
    num_blocks_w = width  // B
    mvs          = np.zeros((num_blocks_h, num_blocks_w, 2), dtype=np.int32)

    dist_fn   = compute_ssd if distance_metric == 'ssd' else compute_sad
    init_step = 1 << (int(math.log2(sr)) if sr >= 1 else 0)

    total_positions = 0

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            y, x      = i * B, j * B
            cur_block = cur_frame[y:y+B, x:x+B]

            cy, cx    = 0, 0
            best_dist = dist_fn(cur_block, ref_frame[y:y+B, x:x+B])
            total_positions += 1

            step = init_step
            while step >= 1:
                best_dn, best_dm = cy, cx
                for dn in (-step, 0, step):
                    for dm in (-step, 0, step):
                        if dn == 0 and dm == 0:
                            continue
                        ty, tx = cy + dn, cx + dm
                        if abs(ty) > sr or abs(tx) > sr:
                            continue
                        ry, rx = y + ty, x + tx
                        if ry < 0 or ry+B > height or rx < 0 or rx+B > width:
                            continue
                        d = dist_fn(cur_block, ref_frame[ry:ry+B, rx:rx+B])
                        total_positions += 1
                        if d < best_dist:
                            best_dist        = d
                            best_dn, best_dm = ty, tx
                cy, cx = best_dn, best_dm
                step >>= 1

            mvs[i, j, 0] = cx   # dm
            mvs[i, j, 1] = cy   # dn

    return mvs, total_positions


def predict_frame_three_step(cur_frame, ref_frame, blocksize, search_range,
                             distance_metric='ssd'):
    """
    Motion-compensated prediction using Three-Step Search.

    :param cur_frame: frame to be predicted
    :param ref_frame: reference frame
    :param blocksize: block side length in pixels
    :param search_range: maximum search displacement in +-pixels
    :param distance_metric: 'ssd' or 'sad'
    :return: pred_frame      predicted frame
             positions       total candidate positions evaluated
    """
    mvs, positions = motion_estimation_three_step(
        cur_frame, ref_frame, blocksize, search_range, distance_metric)
    pred = motion_compensation(ref_frame, blocksize, mvs)
    return pred, positions


def interpolate_half_pixel(frame):
    """
    Bilinear half-pixel interpolation.
    Integer-pixel samples are kept in place; half-pixel positions are
    filled by averaging their integer-pixel neighbours.

    :param frame: original frame
    :return:      half-pixel grid
                  even rows/cols = original samples
                  odd  cols only = horizontal half-pixels  a = (A+B+1)>>1
                  odd  rows only = vertical   half-pixels  b = (A+C+1)>>1
                  odd  rows+cols = diagonal   half-pixels  e = (A+B+C+D+2)>>2
    """
    f = frame.astype(np.int32)
    h, w = f.shape
    out = np.empty((2*h - 1, 2*w - 1), dtype=np.float32)

    out[::2,  ::2]  = f
    out[::2,  1::2] = (f[:, :-1] + f[:, 1:]  + 1) >> 1
    out[1::2, ::2]  = (f[:-1, :] + f[1:, :]  + 1) >> 1
    out[1::2, 1::2] = (f[:-1, :-1] + f[:-1, 1:]
                       + f[1:, :-1] + f[1:, 1:] + 2) >> 2
    return out


def motion_estimation_half_pixel(cur_frame, ref_frame, blocksize, search_range,
                                 use_three_step=True, distance_metric='ssd'):
    """
    Two-stage half-pixel motion estimation.
    Stage 1: integer-pixel search (three-step or full search).
    Stage 2: tests the 8 half-pixel neighbours
             around the integer best match.

    :param cur_frame: current frame
    :param ref_frame: reference frame
    :param blocksize: block side length in pixels
    :param search_range: integer-pixel search range in +-pixels
    :param use_three_step: True = TSS for Stage 1, False = full search
    :param distance_metric: 'ssd' or 'sad'
    :return: mvs  motion vectors in pixel units
                  (multiples of 0.5; e.g. 1.5 means one-and-a-half pixels)
    """
    import math
    height, width = cur_frame.shape
    B  = blocksize
    sr = search_range

    num_blocks_h = height // B
    num_blocks_w = width  // B
    mvs          = np.zeros((num_blocks_h, num_blocks_w, 2), dtype=np.float32)

    ref_interp = interpolate_half_pixel(ref_frame)   # computed once, reused per block
    dist_fn    = compute_ssd if distance_metric == 'ssd' else compute_sad

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            y, x      = i * B, j * B
            cur_block = cur_frame[y:y+B, x:x+B]

            # Stage 1: integer-pixel search
            if use_three_step:
                init_step = 1 << (int(math.log2(sr)) if sr >= 1 else 0)
                cy, cx    = 0, 0
                best_dist = dist_fn(cur_block, ref_frame[y:y+B, x:x+B])
                step      = init_step
                while step >= 1:
                    best_dn, best_dm = cy, cx
                    for dn in (-step, 0, step):
                        for dm in (-step, 0, step):
                            if dn == 0 and dm == 0:
                                continue
                            ty, tx = cy + dn, cx + dm
                            if abs(ty) > sr or abs(tx) > sr:
                                continue
                            ry, rx = y + ty, x + tx
                            if ry < 0 or ry+B > height or rx < 0 or rx+B > width:
                                continue
                            d = dist_fn(cur_block, ref_frame[ry:ry+B, rx:rx+B])
                            if d < best_dist:
                                best_dist        = d
                                best_dn, best_dm = ty, tx
                    cy, cx = best_dn, best_dm
                    step >>= 1
                int_dn, int_dm = cy, cx
            else:
                tmp    = motion_estimation(cur_frame, ref_frame, B, sr,
                                           distance_metric)
                int_dm = int(tmp[i, j, 0])
                int_dn = int(tmp[i, j, 1])
                best_dist = dist_fn(cur_block,
                                    ref_frame[y+int_dn:y+int_dn+B,
                                              x+int_dm:x+int_dm+B])

            # Stage 2: half-pixel refinement
            best_mv_y, best_mv_x = float(int_dn), float(int_dm)
            for dy_h in (-0.5, 0.0, 0.5):
                for dx_h in (-0.5, 0.0, 0.5):
                    if dy_h == 0.0 and dx_h == 0.0:
                        continue
                    ty_f, tx_f = int_dn + dy_h, int_dm + dx_h
                    # map to half-pixel grid coordinates (factor-of-2 grid)
                    iy = int((y + ty_f) * 2)
                    ix = int((x + tx_f) * 2)
                    if (iy < 0 or iy + B*2 > ref_interp.shape[0] or
                            ix < 0 or ix + B*2 > ref_interp.shape[1]):
                        continue
                    cand = ref_interp[iy:iy+B*2:2, ix:ix+B*2:2].astype(np.uint8)
                    if cand.shape != (B, B):
                        continue
                    d = dist_fn(cur_block, cand)
                    if d < best_dist:
                        best_dist            = d
                        best_mv_y, best_mv_x = ty_f, tx_f

            mvs[i, j, 0] = best_mv_x   # dm (horizontal)
            mvs[i, j, 1] = best_mv_y   # dn (vertical)

    return mvs


def motion_compensation_half_pixel(ref_frame, blocksize, mvs):
    """
    Motion compensation with half-pixel accuracy.

    :param ref_frame:  reference frame
    :param blocksize:  block side length in pixels
    :param mvs:        half-pixel motion vectors [dm, dn]
    :return:           predicted frame
    """
    height, width          = ref_frame.shape
    B                      = blocksize
    num_blocks_h, num_blocks_w = mvs.shape[:2]
    pred_frame  = np.zeros((height, width), dtype=np.uint8)
    ref_interp  = interpolate_half_pixel(ref_frame)

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            y, x   = i * B, j * B
            mv_x   = float(mvs[i, j, 0])   # dm – horizontal displacement
            mv_y   = float(mvs[i, j, 1])   # dn – vertical   displacement
            iy     = int((y + mv_y) * 2)
            ix     = int((x + mv_x) * 2)
            pred_frame[y:y+B, x:x+B] = ref_interp[iy:iy+B*2:2,
                                                    ix:ix+B*2:2].astype(np.uint8)
    return pred_frame


def predict_frame_half_pixel(cur_frame, ref_frame, blocksize, search_range,
                             use_three_step=True, distance_metric='ssd'):
    """
    Half-pixel-accurate motion-compensated prediction for one frame.

    :param cur_frame: frame to be predicted
    :param ref_frame: reference frame
    :param blocksize: block side length in pixels
    :param search_range:  integer-pixel search range in +-pixels
    :param use_three_step: True = TSS for integer search, False = full search
    :param distance_metric: 'ssd' or 'sad'
    :return:   predicted frame
    """
    mvs  = motion_estimation_half_pixel(cur_frame, ref_frame, blocksize,
                                        search_range, use_three_step,
                                        distance_metric)
    pred = motion_compensation_half_pixel(ref_frame, blocksize, mvs)
    return pred