
import os
import datetime
import numpy
import pandas as pd
from nrc_spifpy.input.binary_file import BinaryFile
from nrc_spifpy.images import Images

from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
from tqdm import tqdm
import time
import gc


try:
    from .fast_2ds_decoder import decode_frame
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

if os.environ.get('FORCE_PYTHON_F2DS') == '1':
    HAS_CYTHON = False

if HAS_CYTHON:
    print("Using cythonized decoder")
else:
    print("Using pure python decoder")

class Fast2DSFile(BinaryFile):
    """
    Class for reading and processing SPEC Fast 2D-S (Type 48) probe data.
    
    Reference: SPEC_OAP_Data_File_Formats_July_2022_Rev_D
    
    File Structure:
        - Base file (.F2DS): Contains image packets (2S) and NULL packets (NL)
        - HK file (.F2DSHK): Contains housekeeping (HK) and mask (MK) packets
    
    Data Block Format (4114 bytes):
        - Timestamp: 16 bytes (8 x uint16: year, month, dow, day, hour, min, sec, ms)
        - Raw Data: 4096 bytes (2048 x uint16 words containing packets)
        - Checksum: 2 bytes (discarded)
    
    Image Packet Format:
        - Word 1: Header "2S" (0x3253 = 12883 decimal)
        - Word 2: NHraw (bits 11-0: word count, bit 12: multi-packet, bit 15: overflow)
        - Word 3: NVraw (same format as NHraw)
        - Word 4: Particle ID (16-bit counter)
        - Word 5: Slice count (number of 128-pixel slices)
        - Words 6 to 5+N-3: Compressed/uncompressed image data
        - Words 5+N-2 to 5+N: 48-bit timing (only if bit 12 = 0)
    
    Image Data Encoding:
        - 0x4000: Fully shaded slice (128 shaded pixels)
        - 0x7FFF: Next 8 words are uncompressed (8 x 16 bits = 128 pixels)
        - Otherwise: RLE word (bit 14: start slice, bits 13-7: shaded, bits 6-0: clear)
    
    Attributes:
        diodes (int): Number of diodes (128 for 2D-S)
        aux_channels (list): Auxiliary data channels from HK file
    """
    
    # -------------------------------------------------------------------------
    # Format Constants
    # -------------------------------------------------------------------------
    SYNC_2S = 12883           # 0x3253: Image packet header
    SYNC_NL = 0x4C4E          # NULL packet header
    RLE_FULL_SHADED = 0x4000  # Fully shaded slice (128 pixels)
    RLE_UNCOMPRESSED = 0x7FFF # Next 8 words are raw bitmap
    # Note: Timing uses frame timestamp interpolation (not fixed clock tick)

    # =========================================================================
    # Phase 1: Initialization
    # =========================================================================

    def __init__(self, filename, inst_name, resolution):
        super().__init__(filename, inst_name, resolution)
        self.diodes = 128  # 2D-S has 128 photodiodes
        
        # Data block dtype: 4114 bytes = 16 (timestamp) + 4096 (data) + 2 (checksum)
        self.file_dtype = numpy.dtype([
            ('year', 'u2'),      # Timestamp word 1
            ('month', 'u2'),     # Timestamp word 2
            ('weekday', 'u2'),   # Timestamp word 3 (0=Sun, 6=Sat)
            ('day', 'u2'),       # Timestamp word 4
            ('hour', 'u2'),      # Timestamp word 5
            ('minute', 'u2'),    # Timestamp word 6
            ('second', 'u2'),    # Timestamp word 7
            ('ms', 'u2'),        # Timestamp word 8 (milliseconds)
            ('data', '(2048,)u2'),  # Raw data frame (4096 bytes)
            ('discard', 'u2')    # Checksum (not used)
        ])
        
        # Auxiliary channels interpolated from external HK file
        self.aux_channels = ['tas', 'user_temp', 'ps_temp']

    # =========================================================================
    # Phase 2: File Reading
    # =========================================================================

    def read(self):
        """ Read the base file and the separate HK file. """
        super().read() # Parent reads the base file into self.data
        
        # Child reads external HK file
        # For .F2DS files, HK is .F2DSHK (filename + 'HK')
        hk_filename = str(self.filename) + 'HK'
        
        if os.path.exists(hk_filename):
            self.hk_data = self._read_external_hk(hk_filename)
        else:
            print(f"Warning: HK file {hk_filename} not found.")
            self.hk_data = None
            
        if self.hk_data is not None:
             self._align_hk_to_frames() # Child aligns TAS to image frames
        else:
             print("Warning: Initializing empty TAS (HK missing/failed).")
             # We need to know length of datetimes to init TAS, but datetimes are calc'd in read().
             # read() calls super().read() first. So datetimes should exist?
             # super().read() -> calc_buffer_datetimes().
             # Yes.
             self.tas = numpy.zeros(len(self.datetimes))

    def _read_external_hk(self, filename):
        """
        Reads the Type 48 Fast 2DS Housekeeping file.
        File structure: 72-byte Mask Pack + N x 182-byte HouseKeeping Packet records.
        Each 72-byte Mask Pack record: Timestamp (16 bytes) + Raw (54 bytes) + Checksum (2 bytes)
        Each 182-byte HouseKeeping Packet record: Timestamp (16 bytes) + Raw (164 bytes) + Checksum (2 bytes).
        """
        hk_dtype = numpy.dtype([('ts_year', '<u2'), ('ts_month', '<u2'), ('ts_dow', '<u2'), ('ts_day', '<u2'),
                                ('ts_hour', '<u2'), ('ts_min', '<u2'), ('ts_sec', '<u2'), ('ts_ms', '<u2'),
                                ('data', '(82,)<u2'),  # 164 bytes = 82 words
                                ('checksum', '<u2')])
        
        try:
            # Detect header size based on file size
            file_size = os.path.getsize(filename)
            remainder = file_size % 182
            
            offset = 0
            if remainder == 72:
                offset = 72
            elif remainder == 0:
                offset = 0
            else:
                print(f"Warning: HK file size {file_size} has remainder {remainder} (not 0 or 72). Assuming 72 and truncating.")
                offset = 72

            # Skip header, then read 182-byte records
            with open(filename, 'rb') as f:
                if offset > 0:
                    f.seek(offset)
                raw = f.read()
            
            # Truncate to multiple of 182 bytes
            read_len = len(raw)
            trunc_rem = read_len % 182
            if trunc_rem != 0:
                 print(f"Warning: HK read content (size {read_len}) not aligned to 182 bytes. Truncating {trunc_rem} bytes.")
                 raw = raw[:-trunc_rem]
            
            raw_hk = numpy.frombuffer(raw, dtype=hk_dtype)
            return raw_hk
        except Exception as e:
            print(f"Error reading HK file: {e}")
            return None


    def calc_buffer_datetimes(self):
        """ Calculates datetimes from buffers read in from file.
        Override to include milliseconds using vectorized pandas/numpy.
        """
        # Ensure start_date is set
        if not hasattr(self, 'start_date') or self.start_date is None:
             self.get_start_date()

        # Build datetime64 array directly from buffer fields
        df = pd.DataFrame({
            'year': self.data['year'],
            'month': self.data['month'],
            'day': self.data['day'],
            'hour': self.data['hour'],
            'minute': self.data['minute'],
            'second': self.data['second'],
            'microsecond': self.data['ms'].astype(numpy.int64) * 1000  # ms -> us
        })
        self.datetimes = pd.to_datetime(df).values  # numpy datetime64[ns] array


    def _align_hk_to_frames(self):
        """
        Interpolates HK data (TAS, Temps) to the timestamps of the image frames.
        """
        if self.hk_data is None or self.datetimes is None:
            return

        # 1. Convert HK timestamps to unix timestamps (Vectorized)
        data = self.hk_data
        
        # Extract components safely
        years = data['ts_year'].astype(numpy.int32)
        months = data['ts_month'].astype(numpy.int32)
        days = data['ts_day'].astype(numpy.int32)
        hours = data['ts_hour'].astype(numpy.int32)
        minutes = data['ts_min'].astype(numpy.int32)
        seconds = data['ts_sec'].astype(numpy.int32)
        ms = data['ts_ms'].astype(numpy.int32)

        # Validity mask to filter bad records
        valid_mask = (
            (months >= 1) & (months <= 12) &
            (days >= 1) & (days <= 31) &
            (hours >= 0) & (hours <= 23) &
            (minutes >= 0) & (minutes <= 59) &
            (seconds >= 0) & (seconds <= 59) &
            (years >= 2000) & (years <= 2100)
        )
        
        valid_indices = numpy.nonzero(valid_mask)[0]
        
        if len(valid_indices) == 0:
            print("Warning: No valid HK timestamps found.")
            self.tas = numpy.zeros(len(self.datetimes))
            self.user_temp = numpy.zeros(len(self.datetimes))
            self.ps_temp = numpy.zeros(len(self.datetimes))
            return

        # Convert to unix timestamps
        ts_df = pd.DataFrame({
            'year': years[valid_indices],
            'month': months[valid_indices],
            'day': days[valid_indices],
            'hour': hours[valid_indices],
            'minute': minutes[valid_indices],
            'second': seconds[valid_indices],
            'microsecond': ms[valid_indices] * 1000  # ms -> us
        })

        # Convert to nanoseconds (float64)
        hk_ts = pd.to_datetime(ts_df).values.astype(numpy.float64)

        if len(hk_ts) == 0:
            print("Warning: No valid HK timestamps found after conversion.")
            self.tas = numpy.zeros(len(self.datetimes))
            self.user_temp = numpy.zeros(len(self.datetimes))
            self.ps_temp = numpy.zeros(len(self.datetimes))
            return
            
        # 2. Convert Frame timestamps to nanoseconds (float64)
        frame_ts = self.datetimes.astype(numpy.float64)
        
        # 3. Extract and Interpolate Channels (use only valid HK records)
        
        # --- TAS (True Air Speed) ---
        # TAS is at words 75,76 in the 82-word data section (empirically verified)
        # Word 75 is MSW (17194), Word 76 is LSW (0) -> (w75 << 16) | w76 = 170.00 m/s
        
        tas_w75 = self.hk_data['data'][valid_indices, 75]
        tas_w76 = self.hk_data['data'][valid_indices, 76]
        
        # Combine to 32-bit int
        tas_int = (tas_w75.astype(numpy.uint32) << 16) | tas_w76.astype(numpy.uint32)
        hk_tas = numpy.frombuffer(tas_int.tobytes(), dtype=numpy.float32)
        
        # Filter out invalid TAS values and interpolate
        valid_tas_mask = (hk_tas > 0) & (hk_tas < 500)
        
        if numpy.any(valid_tas_mask):
            self.tas = numpy.interp(frame_ts, hk_ts[valid_tas_mask], hk_tas[valid_tas_mask])
        else:
            print("Warning: No valid TAS values found in HK data.")
            self.tas = numpy.zeros(len(self.datetimes))
        
        # --- Temperatures ---
        # DSP Board Temp (Word 17) as 'user_temp' and Power Supply (Word 22) as 'ps_temp'
        # Conversion: C0=-64.8, C1=0.07323
        
        raw_dsp = self.hk_data['data'][valid_indices, 16]
        dsp_temp = raw_dsp * 0.07323 - 64.8
        self.user_temp = numpy.interp(frame_ts, hk_ts, dsp_temp)
        
        # Power Supply Word 22 (Index 21)
        raw_ps = self.hk_data['data'][valid_indices, 21]
        ps_temp = raw_ps * 0.07323 - 64.8
        self.ps_temp = numpy.interp(frame_ts, hk_ts, ps_temp)


    # =========================================================================
    # Phase 3: Parallel Processing (Main Entry Point)
    # =========================================================================

    def process_file(self, spiffile, processors=None):
        """
        Orchestrates parallel processing of the file.
        Uses per-chunk state for cross-frame particle handling within each parallel worker.
        """
        if processors is None:
            processors = os.cpu_count() - 1 if os.cpu_count() > 1 else 1

        spiffile.set_start_date(self.start_date.strftime('%Y-%m-%d %H:%M:%S %z'))
        
        # Create Output Groups
        spiffile.create_inst_group(self.name + '-H')
        spiffile.create_inst_group(self.name + '-V')
        spiffile.set_filenames_attr(self.name + '-H', self.filename)
        spiffile.set_filenames_attr(self.name + '-V', self.filename)
        
        # Write buffer info
        spiffile.write_buffer_info(self.start_date, self.datetimes)
        
        # Setup parallel processing
        process_until = len(self.data)
        chunksize = 500
        data_chunk = range(0, process_until)
        
        pbar1 = tqdm(desc='Processing frames', total=process_until, unit='frame')
        pbar2 = tqdm(desc='Writing frames', total=process_until, unit='frame')
        
        futures = []
        max_write_queue = 8
        images_remaining = True
        i = 0
        
        tot_h = 0
        tot_v = 0
        t00 = time.time()
        
        with ProcessPoolExecutor(max_workers=processors) as executor:
            while True:
                while len(futures) <= max_write_queue and images_remaining:
                    # Submit chunk
                    futures.append(executor.submit(self.process_frames, data_chunk[i: i + chunksize]))
                    i += chunksize
                    if i >= process_until:
                         images_remaining = False
                    pbar1.update(chunksize)
                
                # Collect Results
                done, running = wait(futures, return_when=FIRST_COMPLETED)
                for f in done:
                    indx = futures.index(f)
                    if indx == 0:
                        pbar2.update(chunksize)
                        h_imgs, v_imgs = f.result()
                        
                        tot_h += len(h_imgs)
                        tot_v += len(v_imgs)
                        
                        # Write Images (must call conv_to_array first to flatten)
                        if len(h_imgs) > 0:
                            h_imgs.conv_to_array(self.diodes)
                            spiffile.write_images(self.name + '-H', h_imgs)
                        if len(v_imgs) > 0:
                            v_imgs.conv_to_array(self.diodes)
                            spiffile.write_images(self.name + '-V', v_imgs)
                            
                        futures.pop(indx)
                        gc.collect()
                        
                if not images_remaining and len(futures) == 0:
                    break
        
        pbar1.close()
        pbar2.close()
        
        print(f'Finished. {tot_h}-H, {tot_v}-V images processed in {time.time()-t00:.2f}s')


    def process_frames(self, chunk):
        """
        Process a chunk of frames indices.
        State is initialized per-chunk for parallel processing compatibility.
        """
        h_accum = []
        v_accum = []
        
        # Initialize per-chunk state (not shared with other chunks)
        # We rely on next_start_idx to handle frame boundaries logic locally
        chunk_state = {
            'pending_h': {},
            'pending_v': {},
        }
        
        next_start_idx = 0
        for frame_idx in chunk:
            result = self.process_frame_with_state(frame_idx, chunk_state, start_idx=next_start_idx)
            h_accum.extend(result['h'])
            v_accum.extend(result['v'])
            next_start_idx = result['next_idx']
            
        buffer = {'h': h_accum, 'v': v_accum}
        return self.extract_images(buffer)

    # =========================================================================
    # Phase 4: Frame Decoding
    # =========================================================================

    def process_frame_with_state(self, frame, chunk_state, start_idx=0):
        """
        Decodes a single 4096-byte frame (stored as 2048 'u2' words).
        Handles multi-packet particles and cross-frame spanning using Look-Ahead.
        
        Parameters
        ----------
        frame : int
            Index of the current frame in self.data.
        chunk_state : dict
            Dictionary holding pending multi-packet particle states.
        start_idx : int, optional
            Index to start processing from within the frame (skipping data consumed by previous frame).
            
        Returns
        -------
        dict
            {'h': h_images, 'v': v_images, 'next_idx': next_start_idx}
        """
        record = self.data[frame]['data']
        
        # Look ahead availability
        try:
            record_next = self.data[frame+1]['data']
            next_record_exists = True
        except (IndexError, KeyError):
            record_next = numpy.empty(0, dtype=numpy.uint16)
            next_record_exists = False
        
        if HAS_CYTHON:
             # Delegate to Cython implementation
             return decode_frame(record, record_next, next_record_exists, chunk_state, frame, start_idx)

        h_images = []
        v_images = []
        
        idx = start_idx
        limit = len(record) # Limit for processing loop
        reach_record_end = False
        next_start_idx = 0
        
        # Main particle scanning loop
        while idx < limit:
            word = record[idx]
            
            # Check for '2S' image packet sync word (0x3253 = 12883)
            if word == self.SYNC_2S:
                # Parse 5-word header: [2S, NHraw, NVraw, PID, Slices]
                if idx + 5 <= limit:
                    nh_temp = record[idx+1] & 0x0FFF
                    nv_temp = record[idx+2] & 0x0FFF
                    n_temp = nh_temp if nh_temp > 0 else nv_temp
                    
                    if idx + 5 + n_temp > limit:
                        reach_record_end = True
                else:
                    reach_record_end = True

                if reach_record_end:
                    if next_record_exists:
                        # Extend record with next frame
                        record = numpy.concatenate((record, record_next))
                        # Note: 'limit' still refers to original frame boundary
                    else:
                        break  # EOF 

                # Decode Header
                nh_raw = record[idx+1]
                nv_raw = record[idx+2]
                pid = record[idx+3]
                slices = record[idx+4]
                
                # Extract word count from bits 11-0 (mask 0x0FFF = 4095)
                nh = nh_raw & 0x0FFF
                nv = nv_raw & 0x0FFF
                n_words = nh if nh > 0 else nv  # One of NH/NV is always 0
                
                if n_words == 0:
                    idx += 1
                    continue
                
                is_horiz = (nh > 0)
                # Bit 12 (0x1000): multi-packet flag (1 = more packets follow, no timing words)
                is_multi_packet = ((nh_raw if is_horiz else nv_raw) & 0x1000) >> 12
                
                data_start = idx + 5
                data_end = data_start + n_words
                
                # Packet data
                full_packet_data = record[data_start:data_end]
                
                # --- Decoding Logic ---
                
                pending = chunk_state['pending_h'] if is_horiz else chunk_state['pending_v']
                if pid not in pending:
                    pending[pid] = {'img_decomp': [], 'slice_decomp': [], 'non_compressed': 0}
                state = pending[pid]
                
                # Process Data
                # Per spec: if bit 12 set (multi-packet), NO timing words at end
                # Otherwise: last 3 words are 48-bit timing
                if is_multi_packet:
                    # Multi-packet: all N words are image data, no timing
                    timing = 0
                    payload_data = full_packet_data
                elif len(full_packet_data) >= 3:
                    # Single/final packet: extract 48-bit timing from last 3 words
                    timing = ((int(full_packet_data[-1]) << 32) | 
                              (int(full_packet_data[-2]) << 16) | 
                              (int(full_packet_data[-3])))
                    payload_data = full_packet_data[:-3]
                else:
                    timing = 0  # Shouldn't happen but handle gracefully
                    payload_data = full_packet_data
                
                # Decode image data (RLE compressed or raw bitmap)
                for val in payload_data:
                    val = int(val)
                    
                    if state['non_compressed'] > 0:
                        # Raw bitmap mode: convert 16-bit word to 16 pixels
                        # Pixel format: 1=clear, 0=shaded (inverted at final output)
                        bin_line = [-1 * (int(n) - 1) for n in bin(val)[2:].zfill(16)[::-1]]
                        state['slice_decomp'].extend(bin_line)
                        state['non_compressed'] -= 1
                        if state['non_compressed'] == 0 and len(state['slice_decomp']) > 0:
                            # Pad slice to 128 pixels and finalize
                            if len(state['slice_decomp']) % 128 > 0:
                                state['slice_decomp'].extend([0] * (128 - (len(state['slice_decomp']) % 128)))
                            state['img_decomp'].extend(state['slice_decomp'])
                            state['slice_decomp'] = []
                            
                    elif val == self.RLE_UNCOMPRESSED:  # 0x7FFF
                        # Start of uncompressed slice: next 8 words are raw 128-bit bitmap
                        if len(state['slice_decomp']) > 0:
                            if len(state['slice_decomp']) % 128 > 0:
                                state['slice_decomp'].extend([0] * (128 - (len(state['slice_decomp']) % 128)))
                            state['img_decomp'].extend(state['slice_decomp'])
                            state['slice_decomp'] = []
                        state['non_compressed'] = 8  # Next 8 words are raw bitmap
                        
                    elif val == self.RLE_FULL_SHADED:  # 0x4000
                        # Fully shaded slice: 128 shaded pixels
                        if len(state['slice_decomp']) > 0:
                            if len(state['slice_decomp']) % 128 > 0:
                                state['slice_decomp'].extend([0] * (128 - (len(state['slice_decomp']) % 128)))
                            state['img_decomp'].extend(state['slice_decomp'])
                            state['slice_decomp'] = []
                        state['img_decomp'].extend([1] * 128)  # All shaded
                        
                    else:
                        # RLE encoded word:
                        #   bit 15: always 0 for RLE
                        #   bit 14: start of new slice (1 = first word of slice)
                        #   bits 13-7: number of shaded pixels (0-127)
                        #   bits 6-0: number of clear pixels (0-127)
                        startslice = (val >> 14) & 1
                        num_shaded = (val >> 7) & 0x7F   # bits 13-7
                        num_clear = val & 0x7F           # bits 6-0
                        
                        if startslice == 1 and len(state['slice_decomp']) > 0:
                            # Start of new slice - finalize previous
                                if len(state['slice_decomp']) % 128 > 0:
                                    state['slice_decomp'].extend([0] * (128 - (len(state['slice_decomp']) % 128)))
                                state['img_decomp'].extend(state['slice_decomp'])
                                state['slice_decomp'] = []
                        # Append clear pixels then shaded pixels
                        state['slice_decomp'].extend([0] * num_clear)
                        state['slice_decomp'].extend([1] * num_shaded)
                

                # Finalize Image
                if not is_multi_packet:
                    if len(state['slice_decomp']) > 0:
                        if len(state['slice_decomp']) % 128 > 0:
                            state['slice_decomp'].extend([0] * (128 - (len(state['slice_decomp']) % 128)))
                        state['img_decomp'].extend(state['slice_decomp'])
                    
                    final_data = [1 - x for x in state['img_decomp']]
                    
                    img_result = {'id': pid, 'slices': slices, 'time': timing, 'data': final_data, 'buffer_index': frame}
                    if is_horiz:
                        h_images.append(img_result)
                    else:
                        v_images.append(img_result)
                    del pending[pid]

                # Exit loop after spanning particle (it was the last one in this frame)
                if reach_record_end:
                    next_start_idx = data_end - limit
                    break
                    
                idx = data_end
            
            elif word == self.SYNC_NL:  # 0x4C4E: NULL packet (FIFO flush/padding)
                idx += 1
            else:
                # Unknown word - skip
                idx += 1
                
        return {'h': h_images, 'v': v_images, 'next_idx': next_start_idx}


    # =========================================================================
    # Phase 5: Image Conversion
    # =========================================================================


    def _vectorized_interpolate(self, timings, frame_timing_table):
        """
        Helper for vectorized interpolation using pre-calculated timing table.
        """
        if not frame_timing_table or len(timings) == 0:
            return numpy.array([], dtype=numpy.int64), numpy.array([], dtype=numpy.int64)
        
        # Convert table to arrays (cached if possible, but fast enough here)
        if not hasattr(self, '_ft_timings') or self._ft_timings is None or len(self._ft_timings) != len(frame_timing_table):
            self._ft_timings = numpy.array([x['timing'] for x in frame_timing_table], dtype=numpy.int64)
            self._ft_timestamps = numpy.array([x['timestamp'] for x in frame_timing_table], dtype=numpy.float64)
        
        ft_timings = self._ft_timings
        ft_timestamps = self._ft_timestamps
        n_frames = len(ft_timings)
        
        if n_frames < 2:
            return numpy.array([], dtype=numpy.int64), numpy.array([], dtype=numpy.int64)

        # 1. Find indices in the table
        idxs = numpy.searchsorted(ft_timings, timings, side='right')
        
        # 2. Handle extrapolation cases
        segment_idxs = idxs.copy()
        segment_idxs[segment_idxs == 0] = 1 
        segment_idxs[segment_idxs >= n_frames] = n_frames - 1 
        
        # 3. Gather T1, T2, t1, t2
        right_indices = segment_idxs
        left_indices = segment_idxs - 1
        
        T1 = ft_timestamps[left_indices]
        T2 = ft_timestamps[right_indices]
        t1 = ft_timings[left_indices]
        t2 = ft_timings[right_indices]
        
        # 4. Interpolate
        denom = (t2 - t1).astype(numpy.float64)
        mask = denom == 0
        denom[mask] = 1.0 
        
        fraction = (timings - t1) / denom
        particle_ts = T1 + fraction * (T2 - T1)
        particle_ts[mask] = T1[mask]
        
        # 5. Convert to sec/ns
        sec_arr = particle_ts.astype(numpy.int64)
        ns_arr = ((particle_ts - sec_arr) * 1e9).astype(numpy.int64)
        
        return sec_arr, ns_arr


    def extract_images(self, buffer, frame_timing_table=None):
        """
        Implementation of the abstract extract_images method.
        Slices the buffer into individual Image objects.
        Returns separate Images objects for H and V channels to match SPECFile structure.
        
        Uses frame timestamp interpolation with VECTORIZED numpy operations for speed:
        T_particle = T_frame1 + (T_frame2 - T_frame1) * (timing - timing1) / (timing2 - timing1)
        """
        h_imgs = Images(self.aux_channels)
        v_imgs = Images(self.aux_channels)
        
        # Build frame timing table if not provided
        if frame_timing_table is None:
            frame_timing_table = getattr(self, '_frame_timing_table', None)
            if frame_timing_table is None:
                # OPTIMIZATION: Build timing table directly from decoded images
                # This avoids re-scanning the file and ensures consistency
                temp_timing_map = {} 
                
                # Check both channels for valid frame timings
                for ch in ['h', 'v']:
                    for img in buffer.get(ch, []):
                        f_idx = img.get('buffer_index')
                        t_val = img.get('time')
                        # Use first valid timing found for each frame
                        if f_idx is not None and t_val is not None and t_val > 0:
                            if f_idx not in temp_timing_map:
                                temp_timing_map[f_idx] = t_val
                
                # Convert to sorted list for interpolation
                frame_timing_table = []
                sorted_idxs = sorted(temp_timing_map.keys())
                
                for f_idx in sorted_idxs:
                    timing_word = temp_timing_map[f_idx]
                    # Calculate timestamp (relative seconds)
                    frame_sec = (self.datetimes[f_idx] - numpy.datetime64(self.start_date)) / numpy.timedelta64(1, 's')
                    frame_timing_table.append({
                        'frame_idx': f_idx,
                        'timestamp': frame_sec,
                        'timing': timing_word
                    })
                
                self._frame_timing_table = frame_timing_table
        
        # Process each channel with vectorization
        for channel_name, imgs_obj in [('h', h_imgs), ('v', v_imgs)]:
            img_list = buffer.get(channel_name, [])
            if not img_list:
                continue
                
            # 1. First pass: Collect all data into lists
            raw_data_list = []
            timings_list = []
            buf_indices_list = []
            
            # Filter valid images and extract data
            for img_dict in img_list:
                if 'data' in img_dict and len(img_dict['data']) > 0:
                     raw_data_list.append(img_dict)
                     timings_list.append(img_dict.get('time', 0))
                     buf_indices_list.append(img_dict.get('buffer_index', 0))
            
            if not raw_data_list:
                continue

            # 2. Calculate precise timing using interpolated particle times
            # Use the 48-bit timing words collected in timings_list
            timings_arr = numpy.array(timings_list, dtype=numpy.int64)
            
            # Use vectorized interpolation
            sec_array, ns_array = self._vectorized_interpolate(timings_arr, frame_timing_table)
            
            # Fallback if interpolation returns empty (shouldn't happen if len > 0)
            if len(sec_array) != len(timings_arr):
                # 2. Calculate precise timing from buffer timestamps directly
                # This is simpler and more reliable than 48-bit timing word interpolation
                buf_indices_arr = numpy.array(buf_indices_list, dtype=numpy.int64)
                
                # Get buffer_sec for each particle (relative seconds from start_date)
                sec_array = numpy.zeros(len(buf_indices_arr), dtype=numpy.int64)
                ns_array = numpy.zeros(len(buf_indices_arr), dtype=numpy.int64)
                
                for i, buf_idx in enumerate(buf_indices_arr):
                    # For numpy datetime64: subtract and convert to float seconds
                    dt64 = self.datetimes[buf_idx]
                    delta = (dt64 - numpy.datetime64(self.start_date)) / numpy.timedelta64(1, 's')
                    sec_array[i] = int(delta)
                    ns_array[i] = int((delta - sec_array[i]) * 1e9)
            
            # 3. Populate Images object
            for i, img_dict in enumerate(raw_data_list):
                try:
                    image_data = numpy.array(img_dict['data'], dtype=numpy.uint8)
                    
                    imgs_obj.image.append(image_data)
                    imgs_obj.ns.append(int(ns_array[i]))
                    imgs_obj.sec.append(int(sec_array[i]))
                    imgs_obj.length.append(len(image_data) // 128)
                    
                    buf_idx = buf_indices_list[i]
                    imgs_obj.buffer_index.append(buf_idx)
                    
                    if len(self.tas) > buf_idx:
                        imgs_obj.tas.append(self.tas[buf_idx])
                    else:
                        imgs_obj.tas.append(0.0)
                        
                except (ValueError, TypeError):
                    pass
        
        return h_imgs, v_imgs
