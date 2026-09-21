"""Fail-closed NVIDIA device-memory observations; never an allocation quota.

The watchdog counts the whole device, including other applications, rather than
under-counting WDDM process usage. A sampled limit can be crossed between polls.
"""
from __future__ import annotations
import ctypes as C
import os
from pathlib import Path


class DeviceMemory:
    def __init__(self):
        if os.name != 'nt':
            raise NotImplementedError('reviewed GPU worker monitoring currently requires Windows')
        self.api = C.WinDLL(str(Path(os.environ['SystemRoot']) / 'System32' / 'nvml.dll'))
        self.open = False
        class Memory(C.Structure):
            _fields_ = [('total', C.c_ulonglong), ('free', C.c_ulonglong), ('used', C.c_ulonglong)]
        self.Memory = Memory
        signatures = {'nvmlInit_v2': [], 'nvmlShutdown': [],
            'nvmlDeviceGetCount_v2': [C.POINTER(C.c_uint)],
            'nvmlDeviceGetHandleByIndex_v2': [C.c_uint, C.POINTER(C.c_void_p)],
            'nvmlDeviceGetMemoryInfo': [C.c_void_p, C.POINTER(Memory)],
            'nvmlDeviceGetUUID': [C.c_void_p, C.c_char_p, C.c_uint]}
        for name, args in signatures.items():
            call = getattr(self.api, name)
            call.argtypes, call.restype = args, C.c_int
        self._check(self.api.nvmlInit_v2())
        self.open = True
        try:
            count = C.c_uint()
            self._check(self.api.nvmlDeviceGetCount_v2(C.byref(count)))
            if count.value != 1:
                raise ValueError('reviewed GPU worker requires exactly one visible physical NVIDIA GPU')
            self.device = C.c_void_p()
            self._check(self.api.nvmlDeviceGetHandleByIndex_v2(0, C.byref(self.device)))
            buffer = C.create_string_buffer(96)
            self._check(self.api.nvmlDeviceGetUUID(self.device, buffer, len(buffer)))
            self.uuid = buffer.value.decode('ascii')
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _check(code):
        if code:
            raise RuntimeError('NVIDIA device memory monitor unavailable: NVML status ' + str(code))

    def used(self):
        value = self.Memory()
        self._check(self.api.nvmlDeviceGetMemoryInfo(self.device, C.byref(value)))
        if not 0 <= value.used <= value.total or not value.total:
            raise ValueError('invalid NVIDIA memory observation')
        return value.used

    def close(self):
        if self.open:
            self._check(self.api.nvmlShutdown())
            self.open = False
