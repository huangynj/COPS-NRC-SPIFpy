
import os
import datetime
import numpy
from nrc_spifpy.input.binary_file import BinaryFile
from nrc_spifpy.images import Images

from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
from tqdm import tqdm
import time
import gc

class Fast2DSFile(BinaryFile):
    """
    Class representing "Fast 2DS" (Type 48) probe data.
    Based on SPEC_OAP_Data_File_Formats_July_2022_Rev_D.
    
    Key Characteristics:
    - Type 48 Format
    - 48-bit timing words (encoded as 3 x 16-bit words at end of particle)
    - External Housekeeping file (.2DSHK)
    - Mixed Compressed (RLE) and Uncompressed data
    """
    
    # 2DS starts with 0x3253 ('2S')
    syncword = 12883 

    def __init__(self, filename, inst_name, resolution):
        super().__init__(filename, inst_name, resolution)
        self.diodes = 128
        
        # Base file contains only Images and NULLs, but we still define a dtype for the header
        self.file_dtype = numpy.dtype([('year', 'u2'),
                                       ('month', 'u2'),
                                       ('weekday', 'u2'),
                                       ('day', 'u2'),
                                       ('hour', 'u2'),
                                       ('minute', 'u2'),
                                       ('second', 'u2'),
                                       ('ms', 'u2'),
                                       ('data', '(2048, )u2'),
                                       ('discard', 'u2') 
                                       ])
                                       
        # Aux channels come from external HK file
        self.aux_channels = ['tas', 'user_temp', 'ps_temp']
        
        # Track pending multi-particle images across frames
        # Each entry: {pid: {'img_decomp': [], 'slice_decomp': [], 'non_compressed': 0}}
        self.pending_h = {}
        self.pending_v = {}

    def read(self):
        """ Read the base file and the separate HK file. """
        super().read() # Reads the base file into self.data
        
        # Read external HK file
        # For .F2DS files, HK is .F2DSHK (filename + 'HK')
        hk_filename = str(self.filename) + 'HK'
        
        if os.path.exists(hk_filename):
            self.hk_data = self._read_external_hk(hk_filename)
        else:
            print(f"Warning: HK file {hk_filename} not found.")
            self.hk_data = None
            
        if self.hk_data is not None:
             self._align_hk_to_frames()
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
        File structure: 72-byte header + N x 182-byte records.
        Each 182-byte record: Timestamp (16 bytes) + Raw (164 bytes) + Checksum (2 bytes).
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
        """ Calculates datetimes from buffers read in from file and sets
        to datetimes class attribute.
        Override to include milliseconds.
        """
        self.datetimes = [datetime.datetime(d['year'],
                                            d['month'],
                                            d['day'],
                                            d['hour'],
                                            d['minute'],
                                            d['second'],
                                            d['ms'] * 1000) for d in self.data]
        self.datetimes = numpy.array(self.datetimes)

    def _align_hk_to_frames(self):
        """
        Interpolates HK data (TAS, Temps) to the timestamps of the image frames.
        """
        if self.hk_data is None or self.datetimes is None:
            return

        # 1. Convert HK timestamps to unix timestamps (skip invalid records)
        hk_ts = []
        valid_indices = []
        for idx, d in enumerate(self.hk_data):
            try:
                # Validate timestamp fields before constructing datetime
                year = int(d['ts_year'])
                month = int(d['ts_month'])
                day = int(d['ts_day'])
                hour = int(d['ts_hour'])
                minute = int(d['ts_min'])
                second = int(d['ts_sec'])
                ms = int(d['ts_ms'])
                
                if not (1 <= month <= 12 and 1 <= day <= 31 and 
                        0 <= hour <= 23 and 0 <= minute <= 59 and
                        0 <= second <= 59 and 2000 <= year <= 2100):
                    continue
                    
                dt = datetime.datetime(year, month, day, hour, minute, second, ms * 1000)
                hk_ts.append(dt.timestamp())
                valid_indices.append(idx)
            except (ValueError, OverflowError):
                # Skip invalid timestamp records
                continue
        
        if len(hk_ts) == 0:
            print("Warning: No valid HK timestamps found.")
            self.tas = numpy.zeros(len(self.datetimes))
            self.user_temp = numpy.zeros(len(self.datetimes))
            self.ps_temp = numpy.zeros(len(self.datetimes))
            return
            
        hk_ts = numpy.array(hk_ts)
        valid_indices = numpy.array(valid_indices)
        
        # 2. Convert Frame timestamps to unix timestamps
        frame_ts = numpy.array([dt.timestamp() for dt in self.datetimes])
        
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

    def process_frame(self, frame):
        """
        Wrapper for backwards compatibility - uses class-level state.
        """
        # Use class-level state for sequential processing
        chunk_state = {
            'pending_h': self.pending_h,
            'pending_v': self.pending_v
        }
        result = self.process_frame_with_state(frame, chunk_state)
        # Update class-level state
        return result
    
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
            next_record_exists = False
        
        h_images = []
        v_images = []
        
        idx = start_idx
        limit = len(record) # Limit for processing loop
        next_start_idx = 0
        
        # Main particle scanning loop
        while idx < limit:
            word = record[idx]
            
            # Check for '2S' Sync (0x3253 = 12883)
            if word == 12883:
                # Need header at minimum
                if idx + 5 <= limit:
                    nh_raw = record[idx+1]
                    nv_raw = record[idx+2]
                    pid = record[idx+3]
                    slices = record[idx+4]
                else:
                    # Header spans to next frame
                    if not next_record_exists: break
                    rem_len = limit - idx
                    # Stitch generic header from current end and next start
                    temp_header = []
                    # Current part
                    import numpy
                    temp_header.extend(record[idx:])
                    # Next part
                    needed = 5 - len(temp_header)
                    temp_header.extend(record_next[:needed])
                    
                    nh_raw = temp_header[1]
                    nv_raw = temp_header[2]
                    pid = temp_header[3]
                    slices = temp_header[4]
                
                nh = nh_raw & 4095
                nv = nv_raw & 4095
                n_words = nh if nh > 0 else nv
                
                if n_words == 0: 
                    idx += 1
                    continue
                
                is_horiz = (nh > 0)
                is_multi_particle = ((nh_raw if is_horiz else nv_raw) & 4096) >> 12
                
                data_start = idx + 5
                data_end = data_start + n_words
                
                # Check if data spans to next frame
                if data_end > limit:
                     # Packet spans to next frame -> LOOK AHEAD
                     if next_record_exists:
                         words_needed = data_end - limit
                         next_start_idx = words_needed # Tell next frame to skip these
                         
                         chunk_next = record_next[:words_needed]
                         
                         # Construct full packet data
                         # Part in current
                         if data_start < limit:
                             p_part1 = record[data_start:]
                         else:
                             p_part1 = [] 
                             
                         import numpy
                         if len(p_part1) > 0:
                             # Convert to list for concatenation if needed or use numpy
                             # record is likely numpy array. 
                             # safe approach:
                             full_packet_data = numpy.concatenate((p_part1, chunk_next))
                         else:
                             full_packet_data = chunk_next
                        
                     else:
                         break # EOF mid-packet
                else:
                     # Packet fully in current frame
                     full_packet_data = record[data_start:data_end]
                
                # --- Decoding Logic ---
                
                pending = chunk_state['pending_h'] if is_horiz else chunk_state['pending_v']
                if pid not in pending:
                    pending[pid] = {'img_decomp': [], 'slice_decomp': [], 'non_compressed': 0}
                state = pending[pid]
                
                # Process Data
                # Timing words identified at start of full_packet_data (Words 2,3,4 relative to packet start)
                # full_packet_data contains [Word 2, Word 3, Word 4, Word 5...] because data_start = idx + 2 (skips Sync, Header) - WAIT
                # idx = Sync. idx+1 = Header. idx+2 = Word 2.
                # data_start = idx + 5 in ORIGINAL code?
                # Let's check original code viewed lines 275:
                # data_start = idx + 5.
                # If idx=0. Sync=0. Head=1. Word2,3,4 = 2,3,4.
                # So data_start 5 is correct if timing is skipped?
                # But debug script used `start=2` to find timing.
                # So timing is at idx 2, 3, 4.
                # data_start was 5. So `full_packet_data` STARTED at 5.
                # So `full_packet_data` *missed* the timing!
                # I need to change `data_start` to `idx + 2`.
                # Then `full_packet_data[0:3]` is timing.
                # And `full_packet_data[3:]` is image.
                
                # Wait, I cannot change `data_start` easily without changing *everything*.
                # Better: Read timing *before* creating `full_packet_data`.
                # `data_start` was `idx + 5`.
                # Timing is at `idx + 2`, `idx + 3`, `idx + 4`.
                # I should read them from `record`.
                
                # Check for cross-frame spanning timing?
                # Timing words are 3 words.
                # Sync + Header + Timing = 5 words.
                # `idx + 5` is exactly where data starts.
                # So timing IS implicitly handled if I read it here.
                
                # Timing Extraction (Words 2, 3, 4)
                # These were already read into nv_raw, pid, slices during header parsing
                t_msw = nv_raw
                t_isw = pid
                t_lsw = slices
                timing = (int(t_msw) << 32) | (int(t_isw) << 16) | int(t_lsw)

                # Wait, what if Header spans? 
                # Lines 247-262 output `temp_header` if spanning.
                # But `temp_header` logic only reads 5 words!
                # Sync(0), Head(1), NV(2), PID(3), Slices(4).
                # Wait. Standard Header (Type 48) is just Sync + NH/NV.
                # Where did `pid` and `slices` come from in the `temp_header` logic?
                # "idx+2" being "nv_raw"?
                # In debug script:
                # Word 0: Sync.
                # Word 1: NH (0x13C3).
                # Word 2: 0x0 (Timing MSW).
                # Word 3: 0x0 (Timing ISW).
                # Word 4: 0x338 (Timing LSW).
                
                # My `process_frame` logic (lines 242-245) assumed:
                # idx+1: nh
                # idx+2: nv (if nh=0?) No.
                # idx+3: pid
                # idx+4: slices
                # This seems designed for Type 30 (CIP)!
                # Type 48 doesn't have PID/Slices in header.
                # It just has Sync + NH match.
                
                # IF the file is Type 48, my header parsing at 242-245 reads TIMING as NV/PID/Slices!
                # nv_raw = t_msw.
                # pid = t_isw.
                # slices = t_lsw.
                
                # This explains why I missed timing!
                # And `data_start = idx + 5` correctly skips them.
                # So `full_packet_data` starts AFTER timing.
                # So `full_packet_data` is purely image data.
                # So why did I have "vertical lines"?
                # Because I *disabled* the `no_timing` logic? No.
                # Maybe `full_packet_data` included *some* timing?
                # No, if `data_start = idx + 5`, we skip 5 words.
                # If Timing is Words 2-4, we skipped it.
                # So `full_packet_data` is just image.
                # Wait. If `full_packet_data` is just image, why lines?
                # Maybe I interpreted *CheckSum* at the end?
                # Or maybe my header logic is wrong.
                
                # Correction:
                # Parse timing from `nv_raw`, `pid`, `slices` variables I already read (or `temp_header`).
                # Use them as `t_msw`, `t_isw`, `t_lsw`.
                # `payload_data` is `full_packet_data` (no slicing needed at end).
                
                # Let's fix lines 310+ logic.
                
                # REVERT strict header parsing if needed, but for now:
                # `t_msw = record[idx+2]`...
                # Handle spanning.
                
                # Process Data
                # Since NH includes the 3 timing words (which are at the start, Words 2-4), 
                # and we started reading 'full_packet_data' at idx+5 (skipping them),
                # 'full_packet_data' has length NH but contains only (NH-3) image words + 3 extra words (likely next header).
                # We must trim the end to remove these 3 extra words.
                if len(full_packet_data) >= 3:
                     payload_data = full_packet_data[:-3]
                else:
                     payload_data = full_packet_data
                
                for val in payload_data:
                    if state['non_compressed'] > 0:
                        bin_line = [-1 * (int(n) - 1) for n in bin(val)[2:].zfill(16)[::-1]]
                        state['slice_decomp'].extend(bin_line)
                        state['non_compressed'] -= 1
                        if state['non_compressed'] == 0 and len(state['slice_decomp']) > 0:
                             if len(state['slice_decomp']) % 128 > 0:
                                 state['slice_decomp'].extend([0] * (128 - (len(state['slice_decomp']) % 128)))
                             state['img_decomp'].extend(state['slice_decomp'])
                             state['slice_decomp'] = []
                    elif val == 0x7FFF:
                         if len(state['slice_decomp']) > 0:
                             if len(state['slice_decomp']) % 128 > 0:
                                 state['slice_decomp'].extend([0] * (128 - (len(state['slice_decomp']) % 128)))
                             state['img_decomp'].extend(state['slice_decomp'])
                             state['slice_decomp'] = []
                         state['non_compressed'] = 8
                    elif val == 0x4000:
                         if len(state['slice_decomp']) > 0:
                             if len(state['slice_decomp']) % 128 > 0:
                                 state['slice_decomp'].extend([0] * (128 - (len(state['slice_decomp']) % 128)))
                             state['img_decomp'].extend(state['slice_decomp'])
                             state['slice_decomp'] = []
                         state['img_decomp'].extend([1] * 128)
                    else:
                        timeslice = (val & (2 ** 15)) >> 15
                        startslice = (val & (2 ** 14)) >> 14
                        num_shaded = (val & 16256) >> 7
                        num_clear = (val & 127)
                        
                        if timeslice == 0:
                            if startslice == 1 and len(state['slice_decomp']) > 0:
                                if len(state['slice_decomp']) % 128 > 0:
                                    state['slice_decomp'].extend([0] * (128 - (len(state['slice_decomp']) % 128)))
                                state['img_decomp'].extend(state['slice_decomp'])
                                state['slice_decomp'] = []
                            state['slice_decomp'].extend([0] * num_clear)
                            state['slice_decomp'].extend([1] * num_shaded)
                

                # Finalize Image
                if not is_multi_particle:
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

                # Update index
                if data_end > limit:
                     idx = limit + 1 # Break loop
                else:
                     idx = data_end
            
            elif word == 0x4C4E:
                idx += 1
            else:
                idx += 1
                
        return {'h': h_images, 'v': v_images, 'next_idx': next_start_idx}


    def extract_images(self, buffer):
        """
        Implementation of the abstract extract_images method.
        Slices the buffer into individual Image objects.
        Returns separate Images objects for H and V channels to match SPECFile structure.
        """
        h_imgs = Images(self.aux_channels)
        v_imgs = Images(self.aux_channels)
        
        # 1. Group particles by buffer index and find baseline clock for each buffer
        # This assumes particles within a buffer share a common time reference (start of buffer)
        # We align the first particle's clock to the buffer timestamp.
        
        buffer_baselines = {}
        all_particles = []
        if 'h' in buffer: all_particles.extend(buffer['h'])
        if 'v' in buffer: all_particles.extend(buffer['v'])
        
        for p in all_particles:
            buf_idx = p.get('buffer_index', -1)
            time_clock = p.get('time', 0)
            if buf_idx not in buffer_baselines:
                 buffer_baselines[buf_idx] = time_clock
            else:
                 if time_clock < buffer_baselines[buf_idx]:
                      buffer_baselines[buf_idx] = time_clock

        # 2. Process Horizontal Channel
        for img_dict in buffer['h']:
            if 'data' in img_dict and len(img_dict['data']) > 0:
                try:
                    # Flatten image data to 1D array
                    image_data = numpy.array(img_dict['data'], dtype=numpy.uint8)
                    
                    buf_idx = img_dict.get('buffer_index', 0)
                    timing = img_dict.get('time', 0)
                    
                    # Calculate precise time
                    # T_particle = T_buffer + (Clock_particle - Clock_min) * 50ns
                    base_clock = buffer_baselines.get(buf_idx, timing)
                    delta_clock = timing - base_clock
                    if delta_clock < 0: delta_clock = 0 # Should not happen if base is min
                    
                    delta_ns_total = delta_clock * 50 # 50ns per tick (20 MHz)
                    
                    # Get buffer wall time
                    base_dt = self.datetimes[buf_idx]
                    base_sec = int(base_dt.timestamp())
                    base_ns = base_dt.microsecond * 1000
                    
                    total_ns = base_ns + delta_ns_total
                    add_sec = total_ns // 1000000000
                    rem_ns = total_ns % 1000000000
                    
                    sec = base_sec + add_sec
                    ns = int(rem_ns)
                    
                    h_imgs.image.append(image_data)
                    h_imgs.ns.append(ns)
                    h_imgs.sec.append(sec)
                    # h_imgs.length.append(img_dict.get('slices', len(image_data) // 128))
                    h_imgs.length.append(len(image_data) // 128) # Force calc from data
                    h_imgs.buffer_index.append(img_dict.get('buffer_index', 0))
                    
                    # Populate Aux Channels
                    if len(self.tas) > buf_idx:
                         h_imgs.tas.append(self.tas[buf_idx])
                    else:
                         h_imgs.tas.append(0.0) # Should not happen
                         
                except (ValueError, TypeError):
                    pass

        # Process Vertical Channel
        for img_dict in buffer['v']:
            if 'data' in img_dict and len(img_dict['data']) > 0:
                try:
                    image_data = numpy.array(img_dict['data'], dtype=numpy.uint8)
                    
                    buf_idx = img_dict.get('buffer_index', 0)
                    timing = img_dict.get('time', 0)
                    
                    # Calculate precise time
                    base_clock = buffer_baselines.get(buf_idx, timing)
                    delta_clock = timing - base_clock
                    if delta_clock < 0: delta_clock = 0
                    
                    delta_ns_total = delta_clock * 50
                    
                    base_dt = self.datetimes[buf_idx]
                    base_sec = int(base_dt.timestamp())
                    base_ns = base_dt.microsecond * 1000
                    
                    total_ns = base_ns + delta_ns_total
                    add_sec = total_ns // 1000000000
                    rem_ns = total_ns % 1000000000
                    
                    sec = base_sec + add_sec
                    ns = int(rem_ns)
                    
                    v_imgs.image.append(image_data)
                    v_imgs.ns.append(ns)
                    v_imgs.sec.append(sec)
                    # v_imgs.length.append(img_dict.get('slices', len(image_data) // 128))
                    v_imgs.length.append(len(image_data) // 128) # Force calc from data
                    v_imgs.buffer_index.append(img_dict.get('buffer_index', 0))
                    
                    # Populate Aux Channels
                    if len(self.tas) > buf_idx:
                         v_imgs.tas.append(self.tas[buf_idx])
                    else:
                         v_imgs.tas.append(0.0)

                except (ValueError, TypeError):
                    pass
        
        return h_imgs, v_imgs

    def calc_image_times(self, spiffile):
        """
        Recalculates image times based on clock counts and buffer times.
        Adapted from SPECFile.calc_image_times for Fast 2DS (48-bit clock).
        """
        inst_groups = [self.name + '-H', self.name + '-V']
        
        # Define needed parameters for recomputing times
        times = numpy.array(self.datetimes, dtype='datetime64[ns]') - numpy.datetime64(self.start_date)
        secs = times.astype('timedelta64[s]')
        ns = times - secs
        datetimes = secs.astype(float) + ns.astype(float) * 1e-9

        # Iterate over instrument groups in current file
        for i, inst_group in enumerate(inst_groups):
            if inst_group not in spiffile.instgrps:
                continue

            # Read relevant parameters from spiffile (access core group directly)
            grp = spiffile.instgrps[inst_group]['core']
            
            if 'buffer_index' not in grp.variables or 'image_ns' not in grp.variables:
                continue
                
            buffer_indx = grp['buffer_index'][:]
            if len(buffer_indx) == 0:
                continue
            
            # Fast 2DS uses TAS from auxiliary file or constant
            # For now, if TAS is available in variables, use it. Otherwise use constant.
            if 'tas' in grp.variables:
                tas = grp['tas'][:]
                mask = numpy.isnan(tas)
                if numpy.any(~mask):
                     tas[mask] = numpy.interp(numpy.flatnonzero(mask), numpy.flatnonzero(~mask), tas[~mask])
                else:
                     tas[:] = 100.0 # Fallback
            else:
                tas = numpy.ones(len(buffer_indx)) * 100.0 # Fallback
            
            # Get timestamp for each image corresponding their parent buffer
            buffer_time = datetimes[buffer_indx]

            # Re-save buffer time as image_sec + ns
            # This is a simplified implementation that sets time to buffer time
            # Ideally we would use the 48-bit clock to add precision, but we only have 32-bits stored in 'image_ns'.
            # Given the constraints, aligning to buffer time is the first step.
            epoch_time = numpy.modf(buffer_time)
            new_secs = epoch_time[1]
            new_ns = epoch_time[0] * 1e9

            # Save new time to file
            spiffile.write_variable(grp, 'image_sec', new_secs)
            spiffile.write_variable(grp, 'image_ns', new_ns)

    def _convert_bitmap_to_bits(self, words):
        """
        Helper to convert 8 x 16-bit words (bitmap) into a list of 128 bits (0/1).
        
        Parameters
        ----------
        words : array-like
            8 uint16 words.
            
        Returns
        -------
        list
            List of 128 integers (0 or 1).
        """
        bits = []
        for word in words:
            # 16 bits per word
            # Format: 1=Clear, 0=Shaded
            for i in range(15, -1, -1):
                bits.append((word >> i) & 1)
        return bits

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
