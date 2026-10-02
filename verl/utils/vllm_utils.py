# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

from msgspec import field
from vllm.logger import init_logger
from vllm.lora.lora_model import LoRAModel
from vllm.lora.request import LoRARequest
from vllm.lora.utils import is_in_target_modules, is_supported_lora_module
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager


logger = init_logger(__name__)


class TensorLoRARequest(LoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


class VLLMHijack:
    _original_load_adapter = None
    _is_hijacked = False

    @staticmethod
    def hijack():
        if VLLMHijack._is_hijacked:
            return

        VLLMHijack._original_load_adapter = LRUCacheWorkerLoRAManager._load_adapter

        def hijack__load_adapter(self, lora_request: TensorLoRARequest) -> LoRAModel:
            """
            Extend vLLM's _load_adapter to support loading LoRA adapters directly
            from in-memory tensors. Non-tensor requests are delegated to the
            original vLLM implementation.
            """
            if not isinstance(lora_request, TensorLoRARequest):
                return VLLMHijack._original_load_adapter(self, lora_request)

            supported_lora_modules = self._adapter_manager.supported_lora_modules
            packed_modules_mapping = self._adapter_manager.packed_modules_mapping
            expected_lora_lst: List[str] = []
            for module in supported_lora_modules:
                if module in packed_modules_mapping:
                    expected_lora_lst.extend(packed_modules_mapping[module])
                else:
                    expected_lora_lst.append(module)
                if module == "experts":
                    expected_lora_lst.append(module)
            expected_lora_modules = set(expected_lora_lst)

            from vllm.lora.peft_helper import PEFTHelper

            peft_helper = PEFTHelper.from_dict(lora_request.peft_config)

            peft_helper.validate_legal(self.lora_config)

            model = self._adapter_manager.model
            hf_to_vllm_mapper = getattr(model, "hf_to_vllm_mapper", None)
            lora_skip_prefixes = getattr(model, "lora_skip_prefixes", None)

            lora = self._lora_model_cls.from_lora_tensors(
                lora_model_id=lora_request.lora_int_id,
                tensors=lora_request.lora_tensors,
                peft_helper=peft_helper,
                device="cpu",
                dtype=self.lora_config.lora_dtype,
                model_vocab_size=self.vocab_size,
                weights_mapper=hf_to_vllm_mapper,
                skip_prefixes=lora_skip_prefixes,
            )

            target_modules = self.lora_config.target_modules
            expected_lora_modules_lst = list(expected_lora_modules)
            for module_name in lora.loras:
                if not is_supported_lora_module(module_name, expected_lora_modules_lst):
                    logger.warning_once(
                        "LoRA module '%s' in adapter '%s' is not in the "
                        "model's supported LoRA target modules [%s]. "
                        "These parameters will be ignored, which may "
                        "cause abnormal model behavior.",
                        module_name,
                        lora_request.lora_path,
                        ", ".join(sorted(expected_lora_modules_lst)),
                    )
                elif not is_in_target_modules(module_name, target_modules):
                    logger.warning_once(
                        "LoRA module '%s' in adapter '%s' is not in the "
                        "deployment-time target_modules restriction [%s]."
                        " These parameters will be ignored.",
                        module_name,
                        lora_request.lora_path,
                        ", ".join(sorted(target_modules)),
                    )

            return lora

        setattr(LRUCacheWorkerLoRAManager, "_load_adapter", hijack__load_adapter)
        VLLMHijack._is_hijacked = True
