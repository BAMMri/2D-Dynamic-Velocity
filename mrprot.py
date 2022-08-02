#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# original version by Damien Nguyen
# Modified by Francesco Santini

import itertools
import io
import numpy as np
import re
import pydicom

# ==============================================================================

def postprocess_dict(d):
    """Replace None values in dictionary with 0 or empty arrays
    
    Arguments:
    - `d`: input dictionnary
    """
    for key in list(d.keys()):
        if isinstance(d[key], dict):
            postprocess_dict(d[key])
        elif isinstance(d[key], np.ndarray):
            for idx in range(0, len(d[key])):
                if isinstance(d[key][idx], dict):
                    postprocess_dict(d[key][idx])

                # remove None values since scipy.io.savemat doesn't like them
                elif (key.startswith('u') \
                or key.startswith('ad') \
                or key.startswith('af') \
                or key.startswith('al') \
                or key.startswith('an')) \
                and d[key][idx] == None:
                    d[key][idx] = 0.0

                # NOTE: this only handles some remaning complex data, since
                #       the major part is actually handled by the previous case,
                #       because numpy is nice and auto-converts 0.0 to a complex
                #       value
                elif key == 'ComplexData' and d[key][idx] == None:
                    d[key][idx] = np.complex(0.0)

                elif d[key][idx] == None:
                    d[key][idx] = []
            
        elif d[key] == None:
            d[key] = 0.0

        # ------------------------------

        if isinstance(d[key], np.ndarray) and len(d[key]) > 0:
            try:
                if key.startswith('u'):
                    d[key] = d[key].astype(np.uint32)
                elif key.startswith('al'):
                    d[key] = d[key].astype(np.int32)
                else:
                    c = [np.iscomplex(f) for f in d[key]]
                    if any(c):
                        d[key] = d[key].astype(np.complex)
                    else:
                        d[key] = d[key].astype(np.float)
            except ValueError:
                pass
            except TypeError:
                pass

        # ------------------------------

        if isinstance(d[key], dict):
            key_list = list(d[key].keys())
            if 'Size1' in key_list and 'Size2' in key_list \
               and len(key_list) == 3:
                # extract the name of the last key
                key_list = [k for k in key_list if k not in ['Size1', 'Size2']]
                data = d[key][key_list[0]]
                d[key] = np.reshape(data,
                                    (d[key]['Size1'], d[key]['Size2']),
                                    order='C')

# ==============================================================================

class MrProt_Base(object):
    """Base class for MrProt variaants
    """
    def __init__(self):
        """
        """
        self.main_dict = dict()
        self.cur_dict  = self.main_dict
        self.prev_dict = None
        self.prev_key  = None
    
    def __getitem__(self, key):
        """
        
        Arguments:
        - `key`: Name of field in MrProt
        """
        return self.main_dict[key]

    def is_valid(self):
        """Check whether the current instance contains valid data
        """
        return self.main_dict != dict()

    def reset_cur_dict(self):
        self.cur_dict = self.main_dict
        self.prev_key = None
        self.prev_dict = None

    def get_dict(self):
        if self.main_dict == dict():
            return []
        else:
            return self.main_dict

    def set_value(self, key, val):
        if key in ['dIm', 'dRe']:
            real = 0.0
            imag = 0.0
            if isinstance(self.cur_dict, dict):
                if 'dRe' in list(self.cur_dict.keys()):
                    real = self.cur_dict['dRe']
                if 'dIm' in list(self.cur_dict.keys()):
                    real = self.cur_dict['dIm']
            elif isinstance(self.cur_dict, complex):
                real = self.cur_dict.real
                imag = self.cur_dict.imag

            if key == 'dRe':
                real = val
            else:
                imag = val

            if isinstance(self.prev_key, tuple):
                self.prev_dict[self.prev_key[0]][self.prev_key[1]] = np.complex(real, imag)
            else:
                self.prev_dict[self.prev_key] = np.complex(real, imag)
        else:
            if key.startswith('d'):
                val = float(val)
            elif key.startswith('u'):
                try:
                    val = np.uint32(val)
                except ValueError:
                    val = np.uint32(int(val.replace('#', '').replace('\'', ''), 0))
            elif key.startswith('l'):
                try:
                    val = np.int32(val)
                except ValueError:
                    pass

            self.cur_dict[key] = val

    def do_post_processing(self):
        postprocess_dict(self.main_dict)

# ------------------------------------------------------------------------------
        
class MrProt_VD(MrProt_Base):
    """Class representing an MrProt from a DICOM image  (VDxx line)
    """
    
    def __init__(self):
        super(MrProt_VD, self).__init__()

    def create_array(self, key, size):
        # print 'creating array for {} of size {}'.format(key, size)
        self.cur_dict[key] = np.array([None] * size)

    def cd(self, key):
        # print 'cd to {}'.format(key)
        m = re.search('\[([0-9]+)\]', key)
        if m:
            idx = int(m.group(1))
            real_key = key.replace(m.group(0), '')
            if not isinstance(self.cur_dict[real_key], np.ndarray):
                raise RuntimeError('ERROR: tried to access index {0} of {1}, but {1} is not an array'.format(idx, key))
            else:
                if self.cur_dict[real_key][idx] == None:
                    self.cur_dict[real_key][idx] = dict()
                self.prev_dict = self.cur_dict
                self.prev_key = (real_key, idx)
                self.cur_dict = self.cur_dict[real_key][idx]
        else:
            if key not in list(self.cur_dict.keys()):
                self.cur_dict[key] = dict()
            self.prev_key = key
            self.prev_dict = self.cur_dict
            self.cur_dict = self.cur_dict[key]

    def set_value_in_array(self, key, val):
        m = re.search('\[([0-9]+)\]', key)
        if m:
            real_key = key.replace(m.group(0), '')
            idx = int(m.group(1))
            if idx == 0 and real_key not in self.cur_dict:
                self.cur_dict[real_key] = np.array([val])
            elif isinstance(self.cur_dict[real_key], np.ndarray):
                self.cur_dict[real_key][idx] = val
            else:
                raise RuntimeError('ERROR: key = {} does not lead to an array!'.format(key))
        else:
            raise RuntimeError('ERROR: cannot call set_value_in_array with key = {} (cannot determine index)'.format(key))

# ------------------------------------------------------------------------------

class MrProt_VB(MrProt_Base):
    """Class representing an MrProt from a DICOM image (VBxx line)
    """
    
    def __init__(self):
        """
        """
        super(MrProt_VB, self).__init__()

    def cd(self, key):
        # print 'cd to {}'.format(key)
        m = re.search('\[([0-9]+)\]', key)
        if m:
            idx = int(m.group(1))
            real_key = key.replace(m.group(0), '')
            if real_key not in list(self.cur_dict.keys()) and idx == 0:
                # we are creating a new array
                self.cur_dict[real_key] = np.array([None])
            elif not isinstance(self.cur_dict[real_key], np.ndarray):
                raise RuntimeError('ERROR: tried to access index {0} of {1}, but {1} is not an array'.format(idx, key))
            elif len(self.cur_dict[real_key]) == idx:
                # we need to add an element to the array
                self.cur_dict[real_key] = np.append(self.cur_dict[real_key], None)
            elif len(self.cur_dict[real_key]) < idx:
                raise RuntimeError("ERROR: code does not handle cases where array ' \
                +'index does not increase strictly monotonially ' \
                +'(current length is {}, index is {}".format(len(self.cur_dict[real_key]), idx))

            if self.cur_dict[real_key][idx] == None:
                self.cur_dict[real_key][idx] = dict()
                self.prev_dict = self.cur_dict
                self.prev_key = (real_key, idx)
                self.cur_dict = self.cur_dict[real_key][idx]
        else:
            if key not in list(self.cur_dict.keys()):
                self.cur_dict[key] = dict()
            self.prev_key = key
            self.prev_dict = self.cur_dict
            self.cur_dict = self.cur_dict[key]

    def set_value_in_array(self, key, val):
        m = re.search('\[([0-9]+)\]', key)
        if m:
            real_key = key.replace(m.group(0), b'')
            idx = int(m.group(1))
            if real_key not in self.cur_dict:
                if idx == 0:
                    self.cur_dict[real_key] = np.array([val])
                else:
                    self.cur_dict[real_key] = np.array([None] * (idx+1))
                    self.cur_dict[real_key][idx] = val
            elif isinstance(self.cur_dict[real_key], np.ndarray):
                if idx < len(self.cur_dict[real_key]):
                    self.cur_dict[real_key][idx] = val
                else:
                    # VBxx does not export size of array, so we need to expand array as we go along...
                    Nmissing_elements = idx - len(self.cur_dict[real_key]) +1
                    self.cur_dict[real_key] = np.concatenate((self.cur_dict[real_key], np.array([None] * Nmissing_elements)))
                    try:
                        self.cur_dict[real_key][idx] = val
                    except IndexError as e:
                        print(e)
                        print(real_key, self.cur_dict[real_key], idx, key, val, Nmissing_elements)
                        raise e
        else:
            raise RuntimeError('ERROR: cannot call set_value_in_array with key = {} (cannot determine index)'.format(key))

    def do_post_processing(self):
        postprocess_dict(self.main_dict)

# ==============================================================================

class ProtStreamManager():
    def __init__(self, file_or_header):
        self.file_or_header = file_or_header
        self.file = None
    
    def __enter__(self):
        if type(self.file_or_header) == pydicom.dataset.FileDataset:
            self.file = io.BytesIO(self.file_or_header[0x29,0x1020][:])
        elif type(self.file_or_header) == bytes:
            self.file = io.BytesIO(self.file_or_header)
        else:
            self.file = open(self.file_or_header, 'rb')  
        return self.file
    
    def __exit__(self, exc_type, exc_value, exc_traceback): 
        self.file.close() 

def read_siemens_prot(file_or_header):

    import sys
    # First we need to figure out the baseline
    # Note: not the most efficient way of going about it, but works for now...
    

    baseline_version = None
    # need 'rb' for Windows, doesn't change anything for Unix
    ignore = True
    with ProtStreamManager(file_or_header) as f:
        for line in f:
            if ignore and line.strip().startswith(b'### ASCCONV BEGIN'):
                ignore = False
                continue
            elif ignore:
                continue
    
            line = line.decode('utf-8')
            
            try:
                if 'N4_VB' in line.split('=')[1]:
                    baseline_version = 'VB'
                    break
                elif 'N4_VD' in line.split('=')[1]:
                    baseline_version = 'VD'
                    break
                elif 'N4_VE' in line.split('=')[1]:
                    baseline_version = 'VE'
                    break
            except IndexError:
                pass
            
    # extract the MrProt from the DICOM image
    output = io.BytesIO()
    if baseline_version in  ['VD', 'VE']:
        # start_tag = '### ASCCONV BEGIN object'
        start_tag = b'### ASCCONV BEGIN object=MrProtDataImpl@MrProtocolData'
    else:
        start_tag = b'### ASCCONV BEGIN'
    end_tag   = b'### ASCCONV END'
    
    with ProtStreamManager(file_or_header) as f:
        # need 'rb' for Windows, doesn't change anything for Unix
        it = itertools.dropwhile(
            lambda line: not line.strip().startswith(start_tag), f)
        it = itertools.islice(it, 1, None)

        try:
            next(it)
        except StopIteration as e:
            # No MrProt found... return empty
            return None

        if not all(ord(chr(c)) < 128 for c in next(it)):
            # some DICOM files have a ### ASCCONV BEGIN line with binary
            # data following them, so if we find some non-ASCII char, 
            # we therefore need to find the next occurence of the start_tag
            # (which lies betweent the first ### ASCCONV BEGIN and
            # ### ASCCONV END by the way...)
            it = itertools.dropwhile(
                lambda line: not line.strip().startswith(start_tag), it)
            it = itertools.islice(it, 1, None)

        it = itertools.takewhile(
            lambda line: not line.strip().startswith(end_tag), it)
        output.writelines(it)

    # --------------------------------------------------------------------------

    if baseline_version == 'VB':
        mrprot = MrProt_VB()
    elif baseline_version in  ['VD', 'VE']:
        mrprot = MrProt_VD()
    else:
        raise RuntimeError("Unknown baseline version! (got {0})".format(
            baseline_version))
    output.seek(0)
    count = 0
    for line in output.readlines():
        line = line.decode('utf-8')
        count += 1
        try:
            key,val = [a.strip() for a in line.split('=')]
        except ValueError as e:
            print('')
            print('ERROR:', line)
            print('ERROR:', key, val)
            raise e

        # ----------------------------------------------------------------------
        # Convert val to integer / float if possible

        try:
            val = float(val)
            if val.is_integer():
                val = int(val)
        except ValueError:
            try:
                val = int(val, 0) # for hexadecimal numbers
            except ValueError:
                val = val.replace('""', '')

        # ----------------------------------------------------------------------
        # now split the key to get the name of all the intermediate structures

        subkey = key.split('.')
        process_last_key = True
        # parse the list of keys and create dictionaries/arrays as needed
        for idx in range(0, len(subkey)-1):
            k = subkey[idx]

            # fix case issue with VBxx
            if k.lower() == 'swipmemblock':
                k = 'sWipMemBlock'

            k_next = subkey[idx + 1]

            # Note that the following only works for VD...
            # for VB everything happens during the call to mrprot.cd(...)
            if k_next == '__attribute__':
                # in this case, the next key in the list is 'size', which
                # is totally uninteresting, so we skip it
                mrprot.create_array(k, val)
                process_last_key = False
                break;

            mrprot.cd(k)

        # ------------------------------

        if process_last_key:
            mykey = subkey[-1]

            if '[' in mykey:
                # -> element of an array
                mrprot.set_value_in_array(mykey, val)
            else:
                # -> simple element
                mrprot.set_value(mykey, val)

        # ------------------------------

        mrprot.reset_cur_dict()

        # ----------------------------------------------------------------------

    mrprot.do_post_processing()

    return mrprot

# ==============================================================================

