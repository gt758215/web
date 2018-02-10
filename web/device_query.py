import ctypes


class c_cudaDeviceProp(ctypes.Structure):
    """
    Passed to cudart.cudaGetDeviceProperties()
    """
    _fields_ = [
        ('name', ctypes.c_char * 256),
        ('totalGlobalMem', ctypes.c_size_t),
        ('sharedMemPerBlock', ctypes.c_size_t),
        ('regsPerBlock', ctypes.c_int),
        ('warpSize', ctypes.c_int),
        ('memPitch', ctypes.c_size_t),
        ('maxThreadsPerBlock', ctypes.c_int),
        ('maxThreadsDim', ctypes.c_int * 3),
        ('maxGridSize', ctypes.c_int * 3),
        ('clockRate', ctypes.c_int),
        ('totalConstMem', ctypes.c_size_t),
        ('major', ctypes.c_int),
        ('minor', ctypes.c_int),
        ('textureAlignment', ctypes.c_size_t),
        ('texturePitchAlignment', ctypes.c_size_t),
        ('deviceOverlap', ctypes.c_int),
        ('multiProcessorCount', ctypes.c_int),
        ('kernelExecTimeoutEnabled', ctypes.c_int),
        ('integrated', ctypes.c_int),
        ('canMapHostMemory', ctypes.c_int),
        ('computeMode', ctypes.c_int),
        ('maxTexture1D', ctypes.c_int),
        ('maxTexture1DMipmap', ctypes.c_int),
        ('maxTexture1DLinear', ctypes.c_int),
        ('maxTexture2D', ctypes.c_int * 2),
        ('maxTexture2DMipmap', ctypes.c_int * 2),
        ('maxTexture2DLinear', ctypes.c_int * 3),
        ('maxTexture2DGather', ctypes.c_int * 2),
        ('maxTexture3D', ctypes.c_int * 3),
        ('maxTexture3DAlt', ctypes.c_int * 3),
        ('maxTextureCubemap', ctypes.c_int),
        ('maxTexture1DLayered', ctypes.c_int * 2),
        ('maxTexture2DLayered', ctypes.c_int * 3),
        ('maxTextureCubemapLayered', ctypes.c_int * 2),
        ('maxSurface1D', ctypes.c_int),
        ('maxSurface2D', ctypes.c_int * 2),
        ('maxSurface3D', ctypes.c_int * 3),
        ('maxSurface1DLayered', ctypes.c_int * 2),
        ('maxSurface2DLayered', ctypes.c_int * 3),
        ('maxSurfaceCubemap', ctypes.c_int),
        ('maxSurfaceCubemapLayered', ctypes.c_int * 2),
        ('surfaceAlignment', ctypes.c_size_t),
        ('concurrentKernels', ctypes.c_int),
        ('ECCEnabled', ctypes.c_int),
        ('pciBusID', ctypes.c_int),
        ('pciDeviceID', ctypes.c_int),
        ('pciDomainID', ctypes.c_int),
        ('tccDriver', ctypes.c_int),
        ('asyncEngineCount', ctypes.c_int),
        ('unifiedAddressing', ctypes.c_int),
        ('memoryClockRate', ctypes.c_int),
        ('memoryBusWidth', ctypes.c_int),
        ('l2CacheSize', ctypes.c_int),
        ('maxThreadsPerMultiProcessor', ctypes.c_int),
        ('streamPrioritiesSupported', ctypes.c_int),
        ('globalL1CacheSupported', ctypes.c_int),
        ('localL1CacheSupported', ctypes.c_int),
        ('sharedMemPerMultiprocessor', ctypes.c_size_t),
        ('regsPerMultiprocessor', ctypes.c_int),
        ('managedMemSupported', ctypes.c_int),
        ('isMultiGpuBoard', ctypes.c_int),
        ('multiGpuBoardGroupID', ctypes.c_int),
        # Extra space for new fields in future toolkits
        ('__future_buffer', ctypes.c_int * 128),
        # added later with cudart.cudaDeviceGetPCIBusId
        # (needed by NVML)
        ('pciBusID_str', ctypes.c_char * 16),
    ]


def get_library(name):
    """
    Returns a ctypes.CDLL or None
    """
    try:
        return ctypes.cdll.LoadLibrary(name)
    except OSError:
        pass
    return None


def get_cudart():
    for major in xrange(9, 5, -1):
        for minor in (5, 0):
            cudart = get_library('libcudart.so.%d.%d' % (major, minor))
            if cudart is not None:
                return cudart
    return get_library('libcudart.so')

devices = None


def get_devices(force_reload=False):
    """
    Returns a list of c_cudaDeviceProp's
    Prints an error and returns None if something goes wrong
    Keyword arguments:
    force_reload -- if False, return the previously loaded list of devices
    """
    global devices
    if not force_reload and devices is not None:
        # Only query CUDA once
        return devices
    devices = []

    cudart = get_cudart()
    if cudart is None:
        return []

    # check CUDA version
    cuda_version = ctypes.c_int()
    rc = cudart.cudaRuntimeGetVersion(ctypes.byref(cuda_version))
    if rc != 0:
        print 'cudaRuntimeGetVersion() failed with error #%s' % rc
        return []
    if cuda_version.value < 6050:
        print 'ERROR: Cuda version must be >= 6.5, not "%s"' % cuda_version.value
        return []

    # get number of devices
    num_devices = ctypes.c_int()
    rc = cudart.cudaGetDeviceCount(ctypes.byref(num_devices))
    if rc != 0:
        print 'cudaGetDeviceCount() failed with error #%s' % rc
        return []

    # query devices
    for x in xrange(num_devices.value):
        properties = c_cudaDeviceProp()
        rc = cudart.cudaGetDeviceProperties(ctypes.byref(properties), x)
        if rc == 0:
            pciBusID_str = ' ' * 16
            # also save the string representation of the PCI bus ID
            rc = cudart.cudaDeviceGetPCIBusId(ctypes.c_char_p(pciBusID_str), 16, x)
            if rc == 0:
                properties.pciBusID_str = pciBusID_str
            devices.append(properties)
        else:
            print 'cudaGetDeviceProperties() failed with error #%s' % rc
        del properties
    return devices