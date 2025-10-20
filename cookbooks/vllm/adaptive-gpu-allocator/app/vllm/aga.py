import os
import time

import pynvml
import torch
import vllm
from adaptive_gpu_allocator.pytorch_ddp import PyTorchDDPAdaptiveGPUAllocator
from torch.utils.weak import WeakIdKeyDictionary
from vllm.logger import init_logger
from vllm.platforms.cuda import with_nvml_context

logger = init_logger(__name__)


@with_nvml_context
def device_id_to_physical_device_id(device_id: int) -> int:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        device_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        physical_device_uuid = device_ids[device_id]

        handle = pynvml.nvmlDeviceGetHandleByUUID(physical_device_uuid)
        physical_device_id = pynvml.nvmlDeviceGetIndex(handle)
        return physical_device_id
    else:
        return device_id


@with_nvml_context
def convert_to_idxs(device_uuids_str: str) -> str:
    if device_uuids_str == "":
        return ""

    device_uuids = device_uuids_str.split(",")
    device_idxs = []
    for device_uuid in device_uuids:
        handle = pynvml.nvmlDeviceGetHandleByUUID(device_uuid)
        device_idx = pynvml.nvmlDeviceGetIndex(handle)
        device_idxs.append(device_idx)

    return ",".join(map(str, device_idxs))


# override method
vllm.platforms.cuda.device_id_to_physical_device_id = device_id_to_physical_device_id
vllm.envs.environment_variables["CUDA_VISIBLE_DEVICES"] = lambda: convert_to_idxs(
    os.environ.get("CUDA_VISIBLE_DEVICES", None)
)


class LLMInferenceGPUAllocator(PyTorchDDPAdaptiveGPUAllocator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cpu_tensors = WeakIdKeyDictionary()

    def _move_models_to_device(self, device: torch.device) -> None:
        assert len(self.optimizers) == 0
        assert len(self.schedulers) == 0

        for model in self.models:
            self._move_model_to_device(model, device)

    def _move_model_to_device(self, model, device):
        assert not model.training

        if device.type == "cpu":
            # to CPU
            for p in model.parameters():
                if p.device == device:
                    continue

                cpu_tensor = self.cpu_tensors.get(p)
                if cpu_tensor is None:
                    cpu_tensor = p.to(device, non_blocking=True)
                    assert cpu_tensor.is_pinned() or cpu_tensor.numel() == 0
                    self.cpu_tensors[p] = cpu_tensor

                p.data = cpu_tensor
        else:
            # to non CPU device
            for p in model.parameters():
                if p.device == device:
                    continue

                cpu_tensor = self.cpu_tensors.get(p)
                if cpu_tensor is None:
                    cpu_tensor = p.pin_memory().data
                    self.cpu_tensors[p] = cpu_tensor

                p.data = cpu_tensor.to(device, non_blocking=True)


def set_storage(t, storage):
    new_t = torch.as_strided(
        torch.tensor(storage, dtype=t.dtype, device=storage.device),
        t.size(),
        t.stride(),
        storage_offset=t.storage_offset(),
    )
    t.data = new_t


class AGAManager:
    aga = None
    model = None
    kv_cache = None
    device = None

    @classmethod
    def init(cls, world_size, world_rank, group):
        if cls.aga is not None:
            return

        logger.info("AGA: init")
        cls.aga = LLMInferenceGPUAllocator(world_size=world_size, rank=world_rank)

    @classmethod
    def on_device_begin(cls):
        if cls.aga.on_device:
            return

        logger.info("AGA: on_device_begin")
        t_b = time.time()
        cls.aga.on_device_begin(device="cuda")
        cls.device = cls.aga.get_device()
        logger.info(f"AGA: on_device_begin end ({time.time() - t_b:.1f} s)")
        logger.info(f"AGA: device={cls.aga.get_device()} on rank={cls.aga.rank}")
        cls.alloc_kv_cache_gpu(cls.aga.get_device())

    @classmethod
    def on_device_end(cls):
        if not cls.aga.on_device:
            return

        cls.free_kv_cache_gpu()
        logger.info("AGA: on_device_end")
        t_b = time.time()
        cls.aga.on_device_end()
        cls.device = None
        logger.info(f"AGA: on_device_end end ({time.time() - t_b:.1f} s)")

    @classmethod
    def register_model(cls, model):
        if cls.model is not None:
            logger.warning("AGA: model is already registered")
            return

        cls.model = model
        cls.aga.models.append(model)

    @classmethod
    def register_kv_cache(cls, kv_cache):
        if cls.kv_cache is not None:
            logger.warning("AGA: kv_cache is already registered")
            return

        cls.kv_cache = kv_cache

    @classmethod
    def alloc_kv_cache_gpu(cls, device):
        if cls.kv_cache is None:
            return

        logger.info("AGA: alloc_kv_cache")
        t_b = time.time()
        cache_engines = cls.kv_cache
        if not isinstance(cache_engines, list):
            cache_engines = [cache_engines]

        for cache_engine in cache_engines:
            for t in cache_engine.gpu_cache:
                new_storage = torch.UntypedStorage(size=t.untyped_storage().size(), device=device)
                set_storage(t, new_storage)
                if t._base is not None:
                    set_storage(t._base, new_storage)

        logger.info(f"AGA: alloc_kv_cache ({time.time() - t_b:.1f} s)")

    @classmethod
    def free_kv_cache_gpu(cls):
        if cls.kv_cache is None:
            return

        logger.info("AGA: free_kv_cache")
        t_b = time.time()
        cache_engines = cls.kv_cache
        if not isinstance(cache_engines, list):
            cache_engines = [cache_engines]

        cpu_storage = torch.UntypedStorage(size=cache_engines[0].gpu_cache[0].untyped_storage().size(), device="cpu")
        for cache_engine in cache_engines:
            for t in cache_engine.gpu_cache:
                set_storage(t, cpu_storage)
                if t._base is not None:
                    set_storage(t._base, cpu_storage)

        logger.info(f"AGA: free_kv_cache ({time.time() - t_b:.1f} s)")
