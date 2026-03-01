import numpy as np

def compute_ssd(block1, block2):
    """
    calculate SSD (Sum of Squared Differences) between two blocks 
    
    :param block1: block of current frame
    :param block2: Candidate blocks for the reference frame
    :return: SSD value
    """
    # ensure two blocks are of equel size 
    assert block1.shape == block2.shape, "Blocks must have the same size"
    
    # convert to int32 to prevent overflow, then calculate the sum of the squares of the differences
    diff = block1.astype(np.int32) - block2.astype(np.int32)
    ssd = np.sum(diff ** 2)
    
    return ssd

def compute_sad(block1, block2):
    """
    Compute Sum of Absolute Differences (SAD)
    
    :param block1: block from current frame
    :param block2: candidate block from reference frame
    :return: SAD value
    """
    assert block1.shape == block2.shape, "Blocks must have the same size"
    diff = block1.astype(np.int32) - block2.astype(np.int32)
    sad = np.sum(np.abs(diff))
    return sad


def motion_estimation(cur_frame, ref_frame, blocksize, search_range, distance_metric='ssd'):
    """
    Maximized optimization using NumPy slicing and vectorized operations.
    Keeps original function signature and Full Search logic.
    """
    height, width = cur_frame.shape
    num_blocks_h = height // blocksize # [cite: 4]
    num_blocks_w = width // blocksize # [cite: 4]
    mvs = np.zeros((num_blocks_h, num_blocks_w, 2), dtype=np.int32)
    
    # Pre-calculate search range boundaries to avoid repeated if-statements
    # This loop iterates over blocks, which is much fewer than iterating over pixels
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            cur_y = i * blocksize # [cite: 5]
            cur_x = j * blocksize # [cite: 5]
            cur_block = cur_frame[cur_y:cur_y+blocksize, cur_x:cur_x+blocksize].astype(np.int32)
            
            # Determine the valid search boundaries for the current block
            # This ensures we don't go out of frame [cite: 7]
            win_y_min = max(0, cur_y - search_range)
            win_y_max = min(height - blocksize, cur_y + search_range)
            win_x_min = max(0, cur_x - search_range)
            win_x_max = min(width - blocksize, cur_x + search_range)
            
            best_distance = np.inf
            best_dm = 0
            best_dn = 0
            
            # --- Vectorization Core ---
            # Instead of iterating over every single (dn, dm), we can theoretically
            # extract all candidate blocks. However, for memory safety, we still 
            # loop through offsets but use highly optimized NumPy calls.
            
            for dn in range(win_y_min - cur_y, win_y_max - cur_y + 1):
                # Extract the entire row of candidate blocks for this vertical offset
                ref_y = cur_y + dn
                
                for dm in range(win_x_min - cur_x, win_x_max - cur_x + 1):
                    ref_x = cur_x + dm
                    
                    # Get candidate block from reference frame [cite: 8]
                    ref_block = ref_frame[ref_y:ref_y+blocksize, ref_x:ref_x+blocksize].astype(np.int32)
                    
                    # Calculate distance based on metric [cite: 9]
                    if distance_metric == 'ssd':
                        # Manual inline SSD is faster than calling compute_ssd function 
                        diff = cur_block - ref_block
                        distance = np.sum(diff * diff)
                    else:
                        # SAD calculation [cite: 3]
                        distance = np.sum(np.abs(cur_block - ref_block))
                    
                    # Update best match [cite: 10, 11]
                    if distance < best_distance:
                        best_distance = distance
                        best_dm = dm
                        best_dn = dn
            
            mvs[i, j, 0] = best_dm
            mvs[i, j, 1] = best_dn
            
    return mvs

def motion_compensation(ref_frame, blocksize, mvs):
    # TODO: implement motion compensation

    """
    Motion compensation: generate predicted frame from reference frame using motion vectors
    
    :param ref_frame: reference frame
    :param blocksize: block size
    :param mvs: motion vector array
    :return: pred_frame - predicted frame
    """
    height, width = ref_frame.shape
    num_blocks_h, num_blocks_w = mvs.shape[:2]
    
    # Initialize predicted frame
    pred_frame = np.zeros((height, width), dtype=np.uint8)
    
    # Perform compensation for each block
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            # Position of current block in predicted frame
            cur_y = i * blocksize
            cur_x = j * blocksize
            
            # Get motion vector for this block
            dm = mvs[i, j, 0]
            dn = mvs[i, j, 1]
            
            # Calculate position of corresponding block in reference frame
            ref_y = cur_y + dn
            ref_x = cur_x + dm

            # Copy corresponding block from reference frame to predicted frame
            pred_frame[cur_y:cur_y+blocksize, cur_x:cur_x+blocksize] = \
                ref_frame[ref_y:ref_y+blocksize, ref_x:ref_x+blocksize]
            
    return pred_frame

def predict_frame(cur_frame, ref_frame, blocksize, search_range, distance_metric='ssd'):
    """
    Main function for prediction
    
    :param cur_frame: current frame to be predicted
    :param ref_frame: reference frame (previous frame)
    :param blocksize: block size
    :param search_range: search range
    :param distance_metric: 'ssd' or 'sad' (default: 'ssd')
    :return: predicted frame
    """
    mvs = motion_estimation(cur_frame, ref_frame, blocksize, search_range, distance_metric)
    pred_frame = motion_compensation(ref_frame, blocksize, mvs)
    return pred_frame

def motion_estimation_three_step(cur_frame, ref_frame, blocksize, search_range, distance_metric='ssd'):
    """
    Three Step Search motion estimation algorithm
    
    :param cur_frame: current frame to be predicted
    :param ref_frame: reference frame (previous frame)
    :param blocksize: block size (e.g., 4x4)
    :param search_range: maximum search range (±search_range pixels)
    :param distance_metric: 'ssd' or 'sad' (default: 'ssd')
    :return: mvs - motion vector array, search_positions - number of positions searched
    """
    height, width = cur_frame.shape
    num_blocks_h = height // blocksize
    num_blocks_w = width // blocksize
    mvs = np.zeros((num_blocks_h, num_blocks_w, 2), dtype=np.int32)
    
    # Select distance function
    if distance_metric == 'ssd':
        distance_func = compute_ssd
    elif distance_metric == 'sad':
        distance_func = compute_sad
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")
    
    # Count total search positions
    total_search_positions = 0
    
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            cur_y = i * blocksize
            cur_x = j * blocksize
            cur_block = cur_frame[cur_y:cur_y+blocksize, cur_x:cur_x+blocksize]
            
            # Three Step Search Algorithm
            # Step size starts at search_range//2 and halves each iteration
            step_size = max(search_range // 2, 1)
            
            # Start at center position (0, 0)
            center_dm = 0
            center_dn = 0
            
            # Initialize best match at center
            ref_y = cur_y + center_dn
            ref_x = cur_x + center_dm
            
            if (ref_y >= 0 and ref_y + blocksize <= height and
                ref_x >= 0 and ref_x + blocksize <= width):
                ref_block = ref_frame[ref_y:ref_y+blocksize, ref_x:ref_x+blocksize]
                best_distance = distance_func(cur_block, ref_block)
            else:
                best_distance = np.inf
            
            best_dm = center_dm
            best_dn = center_dn
            total_search_positions += 1
            
            # Three step search iterations
            while step_size >= 1:
                # Test 8 neighboring points around current center
                found_better = False
                
                for dn_offset in [-step_size, 0, step_size]:
                    for dm_offset in [-step_size, 0, step_size]:
                        # Skip center point (already checked)
                        if dn_offset == 0 and dm_offset == 0:
                            continue
                        
                        test_dn = center_dn + dn_offset
                        test_dm = center_dm + dm_offset
                        
                        # Check if within search range
                        if abs(test_dn) > search_range or abs(test_dm) > search_range:
                            continue
                        
                        ref_y = cur_y + test_dn
                        ref_x = cur_x + test_dm
                        
                        # Boundary check
                        if (ref_y >= 0 and ref_y + blocksize <= height and
                            ref_x >= 0 and ref_x + blocksize <= width):
                            
                            ref_block = ref_frame[ref_y:ref_y+blocksize, 
                                                 ref_x:ref_x+blocksize]
                            distance = distance_func(cur_block, ref_block)
                            total_search_positions += 1
                            
                            if distance < best_distance:
                                best_distance = distance
                                best_dm = test_dm
                                best_dn = test_dn
                                found_better = True
                
                # Move center to best position found
                center_dm = best_dm
                center_dn = best_dn
                
                # Reduce step size
                step_size = step_size // 2
            
            # Save best motion vector
            mvs[i, j, 0] = best_dm
            mvs[i, j, 1] = best_dn
    
    return mvs, total_search_positions


def predict_frame_three_step(cur_frame, ref_frame, blocksize, search_range, distance_metric='ssd'):
    """
    Prediction using Three Step Search
    
    :param cur_frame: current frame to be predicted
    :param ref_frame: reference frame
    :param blocksize: block size
    :param search_range: search range
    :param distance_metric: 'ssd' or 'sad'
    :return: predicted frame, number of search positions
    """
    mvs, search_positions = motion_estimation_three_step(cur_frame, ref_frame, 
                                                         blocksize, search_range, 
                                                         distance_metric)
    pred_frame = motion_compensation(ref_frame, blocksize, mvs)
    return pred_frame, search_positions

def interpolate_half_pixel(frame):
    """
    Interpolate frame to half-pixel positions using bilinear interpolation
    Creates a frame with 2x resolution (half-pixel grid)
    
    :param frame: original frame (height x width)
    :return: interpolated frame with half-pixel positions (2*height-1 x 2*width-1)
    """
    height, width = frame.shape
    
    # Create interpolated frame (twice the size minus 1 in each dimension)
    interp_height = 2 * height - 1
    interp_width = 2 * width - 1
    interp_frame = np.zeros((interp_height, interp_width), dtype=np.float32)
    
    # Copy original pixels (at even positions)
    interp_frame[::2, ::2] = frame
    
    # Horizontal interpolation (half pixels between integer pixels horizontally)
    # Position 'a' in lecture notes: (A + B + 1) >> 1
    for i in range(0, interp_height, 2):
        for j in range(1, interp_width, 2):
            orig_i = i // 2
            orig_j_left = (j - 1) // 2
            orig_j_right = (j + 1) // 2
            
            if orig_j_right < width:
                A = frame[orig_i, orig_j_left]
                B = frame[orig_i, orig_j_right]
                interp_frame[i, j] = (int(A) + int(B) + 1) >> 1
    
    # Vertical interpolation (half pixels between integer pixels vertically)
    # Position 'b' in lecture notes: (A + C + 1) >> 1
    for i in range(1, interp_height, 2):
        for j in range(0, interp_width, 2):
            orig_i_top = (i - 1) // 2
            orig_i_bottom = (i + 1) // 2
            orig_j = j // 2
            
            if orig_i_bottom < height:
                A = frame[orig_i_top, orig_j]
                C = frame[orig_i_bottom, orig_j]
                interp_frame[i, j] = (int(A) + int(C) + 1) >> 1
    
    # Diagonal interpolation (center of four integer pixels)
    # Position 'e' in lecture notes: (A + B + C + D + 2) >> 2
    for i in range(1, interp_height, 2):
        for j in range(1, interp_width, 2):
            orig_i_top = (i - 1) // 2
            orig_i_bottom = (i + 1) // 2
            orig_j_left = (j - 1) // 2
            orig_j_right = (j + 1) // 2
            
            if orig_i_bottom < height and orig_j_right < width:
                A = frame[orig_i_top, orig_j_left]
                B = frame[orig_i_top, orig_j_right]
                C = frame[orig_i_bottom, orig_j_left]
                D = frame[orig_i_bottom, orig_j_right]
                interp_frame[i, j] = (int(A) + int(B) + int(C) + int(D) + 2) >> 2
    
    return interp_frame


def motion_estimation_half_pixel(cur_frame, ref_frame, blocksize, search_range, 
                                  use_three_step=True, distance_metric='ssd'):
    """
    Motion estimation with half-pixel accuracy
    
    :param cur_frame: current frame
    :param ref_frame: reference frame
    :param blocksize: block size
    :param search_range: search range (in integer pixels)
    :param use_three_step: use three-step search (True) or full search (False)
    :param distance_metric: 'ssd' or 'sad'
    :return: mvs - motion vectors with half-pixel accuracy (in half-pixel units)
    """
    height, width = cur_frame.shape
    num_blocks_h = height // blocksize
    num_blocks_w = width // blocksize
    
    # Motion vectors in half-pixel units (multiply by 2 for half-pixel grid)
    mvs = np.zeros((num_blocks_h, num_blocks_w, 2), dtype=np.float32)
    
    # Interpolate reference frame to half-pixel positions
    ref_frame_interp = interpolate_half_pixel(ref_frame)
    
    # Select distance function
    if distance_metric == 'ssd':
        distance_func = compute_ssd
    elif distance_metric == 'sad':
        distance_func = compute_sad
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")
    
    # Process each block
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            cur_y = i * blocksize
            cur_x = j * blocksize
            cur_block = cur_frame[cur_y:cur_y+blocksize, cur_x:cur_x+blocksize]
            
            best_distance = np.inf
            best_mv_y = 0.0
            best_mv_x = 0.0
            
            if use_three_step:
                # Step 1: Integer-pixel search using three-step search
                step_size = max(search_range // 2, 1)
                center_y = 0
                center_x = 0
                
                # Test center first
                ref_y = cur_y + center_y
                ref_x = cur_x + center_x
                if (ref_y >= 0 and ref_y + blocksize <= height and
                    ref_x >= 0 and ref_x + blocksize <= width):
                    ref_block = ref_frame[ref_y:ref_y+blocksize, ref_x:ref_x+blocksize]
                    best_distance = distance_func(cur_block, ref_block)
                    best_mv_y = center_y
                    best_mv_x = center_x
                
                # Three-step search at integer positions
                while step_size >= 1:
                    for dy in [-step_size, 0, step_size]:
                        for dx in [-step_size, 0, step_size]:
                            if dy == 0 and dx == 0:
                                continue
                            
                            test_y = center_y + dy
                            test_x = center_x + dx
                            
                            if abs(test_y) > search_range or abs(test_x) > search_range:
                                continue
                            
                            ref_y = cur_y + test_y
                            ref_x = cur_x + test_x
                            
                            if (ref_y >= 0 and ref_y + blocksize <= height and
                                ref_x >= 0 and ref_x + blocksize <= width):
                                ref_block = ref_frame[ref_y:ref_y+blocksize, 
                                                     ref_x:ref_x+blocksize]
                                distance = distance_func(cur_block, ref_block)
                                
                                if distance < best_distance:
                                    best_distance = distance
                                    best_mv_y = test_y
                                    best_mv_x = test_x
                    
                    center_y = best_mv_y
                    center_x = best_mv_x
                    step_size = step_size // 2
                
                # Step 2: Half-pixel refinement around best integer position
                # Search 8 half-pixel positions around best integer match
                integer_mv_y = int(best_mv_y)
                integer_mv_x = int(best_mv_x)
                
                for dy_half in [-0.5, 0.0, 0.5]:
                    for dx_half in [-0.5, 0.0, 0.5]:
                        if dy_half == 0.0 and dx_half == 0.0:
                            continue  # Already tested
                        
                        test_mv_y = integer_mv_y + dy_half
                        test_mv_x = integer_mv_x + dx_half
                        
                        # Position in interpolated frame (half-pixel grid)
                        ref_y_interp = int((cur_y + test_mv_y) * 2)
                        ref_x_interp = int((cur_x + test_mv_x) * 2)
                        
                        # Check bounds
                        if (ref_y_interp >= 0 and ref_y_interp + blocksize * 2 <= ref_frame_interp.shape[0] and
                            ref_x_interp >= 0 and ref_x_interp + blocksize * 2 <= ref_frame_interp.shape[1]):
                            
                            # Extract block from interpolated frame (every 2nd pixel)
                            ref_block = ref_frame_interp[ref_y_interp:ref_y_interp+blocksize*2:2,
                                                        ref_x_interp:ref_x_interp+blocksize*2:2]
                            
                            if ref_block.shape == cur_block.shape:
                                distance = distance_func(cur_block, ref_block.astype(np.uint8))
                                
                                if distance < best_distance:
                                    best_distance = distance
                                    best_mv_y = test_mv_y
                                    best_mv_x = test_mv_x
            
            else:
                # Full search at integer and half-pixel positions
                for dy in range(-search_range, search_range + 1):
                    for dx in range(-search_range, search_range + 1):
                        for dy_half in [0.0, 0.5]:
                            for dx_half in [0.0, 0.5]:
                                test_mv_y = dy + dy_half
                                test_mv_x = dx + dx_half
                                
                                ref_y_interp = int((cur_y + test_mv_y) * 2)
                                ref_x_interp = int((cur_x + test_mv_x) * 2)
                                
                                if (ref_y_interp >= 0 and ref_y_interp + blocksize * 2 <= ref_frame_interp.shape[0] and
                                    ref_x_interp >= 0 and ref_x_interp + blocksize * 2 <= ref_frame_interp.shape[1]):
                                    
                                    ref_block = ref_frame_interp[ref_y_interp:ref_y_interp+blocksize*2:2,
                                                                ref_x_interp:ref_x_interp+blocksize*2:2]
                                    
                                    if ref_block.shape == cur_block.shape:
                                        distance = distance_func(cur_block, ref_block.astype(np.uint8))
                                        
                                        if distance < best_distance:
                                            best_distance = distance
                                            best_mv_y = test_mv_y
                                            best_mv_x = test_mv_x
            
            mvs[i, j, 0] = best_mv_x
            mvs[i, j, 1] = best_mv_y
    
    return mvs


def motion_compensation_half_pixel(ref_frame, blocksize, mvs):
    """
    Motion compensation with half-pixel accuracy
    
    :param ref_frame: reference frame
    :param blocksize: block size
    :param mvs: motion vectors with half-pixel accuracy
    :return: predicted frame
    """
    height, width = ref_frame.shape
    num_blocks_h, num_blocks_w = mvs.shape[:2]
    pred_frame = np.zeros((height, width), dtype=np.uint8)
    
    # Interpolate reference frame
    ref_frame_interp = interpolate_half_pixel(ref_frame)
    
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            cur_y = i * blocksize
            cur_x = j * blocksize
            
            mv_x = mvs[i, j, 0]
            mv_y = mvs[i, j, 1]
            
            # Position in interpolated frame
            ref_y_interp = int((cur_y + mv_y) * 2)
            ref_x_interp = int((cur_x + mv_x) * 2)
            
            # Extract block from interpolated frame
            ref_block = ref_frame_interp[ref_y_interp:ref_y_interp+blocksize*2:2,
                                        ref_x_interp:ref_x_interp+blocksize*2:2]
            
            pred_frame[cur_y:cur_y+blocksize, cur_x:cur_x+blocksize] = ref_block.astype(np.uint8)
    
    return pred_frame


def predict_frame_half_pixel(cur_frame, ref_frame, blocksize, search_range,
                             use_three_step=True, distance_metric='ssd'):
    """
    Prediction with half-pixel accuracy
    """
    mvs = motion_estimation_half_pixel(cur_frame, ref_frame, blocksize, search_range,
                                       use_three_step, distance_metric)
    pred_frame = motion_compensation_half_pixel(ref_frame, blocksize, mvs)
    return pred_frame