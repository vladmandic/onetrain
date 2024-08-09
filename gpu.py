class GPU():
    use_torch = False
    use_nvml = False
    available: bool = False
    version = {}
    devices: list = []
    nvml: list = []

    def __init__(self, use_torch: bool = True, use_nvml: bool = True) -> None:
        self.use_torch = use_torch
        if self.use_torch:
            try:
                import torch
                self.available = torch.cuda.is_available()
                self.init_torch()
                self.update_torch()
            except Exception:
                self.available = False
                self.use_torch = False
        self.use_nvml = use_nvml
        if self.use_nvml:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.version['cuda-driver'] = pynvml.nvmlSystemGetCudaDriverVersion()
                self.version['driver'] = pynvml.nvmlSystemGetDriverVersion()
                self.init_nvml()
                self.update_nvml()
            except Exception:
                self.use_nvml = False

    @staticmethod
    def mb(val: float):
        return round(val / 1024 / 1024, 2)

    def init_torch(self):
        import torch
        if self.available:
            self.version['torch'] = torch.__version__
            self.version['cuda'] = torch.version.cuda
            self.version['cudnn'] = torch.backends.cudnn.version()

    def update_torch(self):
        if not self.available:
            return
        import torch
        self.devices.clear()
        for gpu in [torch.cuda.device(i) for i in range(torch.cuda.device_count())]:
            mem = torch.cuda.mem_get_info(gpu)
            mem = torch.cuda.mem_get_info(gpu)
            s = dict(torch.cuda.memory_stats(gpu))
            device = {
                'name': torch.cuda.get_device_name(gpu),
                'capabilities': torch.cuda.get_device_capability(gpu),
                'memory': { 'free': self.mb(mem[0]), 'used': self.mb(mem[1] - mem[0]), 'total': self.mb(mem[1]) },
                'active': { 'current': self.mb(s['active_bytes.all.current']), 'peak': self.mb(s['active_bytes.all.peak']) },
                'allocated': { 'current': self.mb(s['allocated_bytes.all.current']), 'peak': self.mb(s['allocated_bytes.all.peak']) },
                'reserved': { 'current': self.mb(s['reserved_bytes.all.current']), 'peak': self.mb(s['reserved_bytes.all.peak']) },
                'inactive': { 'current': self.mb(s['inactive_split_bytes.all.current']), 'peak': self.mb(s['inactive_split_bytes.all.peak']) },
                'events': { 'retries': s['num_alloc_retries'], 'oom': s['num_ooms'] },
                'utilization': torch.cuda.utilization(),
            }
            self.devices.append(device)

    def reset(self):
        import torch
        for gpu in [torch.cuda.device(i) for i in range(torch.cuda.device_count())]:
            with gpu:
                torch.cuda.reset_peak_memory_stats()

    def gc(self):
        import torch
        for gpu in [torch.cuda.device(i) for i in range(torch.cuda.device_count())]:
            with gpu:
                torch.cuda.empty_cache() # cuda gc
                torch.cuda.ipc_collect()

    def get_reason(self, val):
        throttle = {
            1: 'gpu idle',
            2: 'applications clocks setting',
            4: 'sw power cap',
            8: 'hw slowdown',
            16: 'sync boost',
            32: 'sw thermal slowdown',
            64: 'hw thermal slowdown',
            128: 'hw power brake slowdown',
            256: 'display clock setting',
        }
        reason = ', '.join([throttle[i] for i in throttle if i & val])
        return reason if len(reason) > 0 else 'ok'

    def init_nvml(self):
        import pynvml
        self.version['cuda-driver'] = pynvml.nvmlSystemGetCudaDriverVersion()
        self.version['driver'] = pynvml.nvmlSystemGetDriverVersion()
        self.nvml.clear()
        for i in range(pynvml.nvmlDeviceGetCount()):
            dev = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                name = pynvml.nvmlDeviceGetName(dev)
            except Exception:
                import torch
                name = torch.cuda.get_device_name(i)
            try:
                rom = pynvml.nvmlDeviceGetInforomImageVersion(dev)
            except:
                rom = None
            device = {
                'name': name,
                'version': {
                    'vbios': pynvml.nvmlDeviceGetVbiosVersion(dev),
                    'rom': rom,
                    'capabilities': pynvml.nvmlDeviceGetCudaComputeCapability(dev),
                },
                'pci': {
                    'link': pynvml.nvmlDeviceGetCurrPcieLinkGeneration(dev),
                    'width': pynvml.nvmlDeviceGetCurrPcieLinkWidth(dev),
                    'busid': pynvml.nvmlDeviceGetPciInfo(dev).busId,
                    'deviceid': pynvml.nvmlDeviceGetPciInfo(dev).pciDeviceId,
                },
            }
            self.nvml.append(device)

    def update_nvml(self):
        import pynvml
        for i in range(pynvml.nvmlDeviceGetCount()):
            dev = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                power = { 'current': round(pynvml.nvmlDeviceGetPowerUsage(dev)/1000, 2), 'max': round(pynvml.nvmlDeviceGetEnforcedPowerLimit(dev)/1000, 2)}
            except:
                power = None
            try:
                fan = pynvml.nvmlDeviceGetFanSpeed(dev)
            except:
                fan = None
            mem = pynvml.nvmlDeviceGetMemoryInfo(dev)
            device = {
                'memory': {
                    'total': self.mb(mem.total),
                    'free': self.mb(mem.free),
                    'used': self.mb(mem.used),
                },
                'clock': { # gpu, sm, memory
                    'gpu': { 'current': pynvml.nvmlDeviceGetClockInfo(dev, 0), 'max': pynvml.nvmlDeviceGetMaxClockInfo(dev, 0)},
                    'sm': { 'current': pynvml.nvmlDeviceGetClockInfo(dev, 1), 'max': pynvml.nvmlDeviceGetMaxClockInfo(dev, 1)},
                    'memory': { 'current': pynvml.nvmlDeviceGetClockInfo(dev, 2), 'max': pynvml.nvmlDeviceGetMaxClockInfo(dev, 2)},
                },
                'load': {
                    'gpu': round(pynvml.nvmlDeviceGetUtilizationRates(dev).gpu),
                    'memory': round(pynvml.nvmlDeviceGetUtilizationRates(dev).memory),
                    'temp': pynvml.nvmlDeviceGetTemperature(dev, 0),
                    'fan': fan,
                },
                'power': power,
                'state': self.get_reason(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(dev)),
            }
            self.nvml[i].update(device)

    def update(self):
        if self.use_torch:
            self.update_torch()
        if self.use_nvml:
            self.update_nvml()
