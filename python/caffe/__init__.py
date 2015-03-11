from .pycaffe import Net, SGDSolver
from ._caffe import (
    set_mode_cpu, set_mode_gpu, set_device, Layer, get_solver,
    get_device,
    check_mode_cpu, check_mode_gpu,
    get_cuda_num_threads, get_blocks,
)
from .proto.caffe_pb2 import TRAIN, TEST
from .classifier import Classifier
from .detector import Detector
import io
