Python 3.10.12
---
Name: torch
Version: 2.5.1+cu121
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
Home-page: https://pytorch.org/
Author: PyTorch Team
Author-email: packages@pytorch.org
License: BSD-3-Clause
Location: /workspaces/AI_Bias/venv_bias/lib/python3.10/site-packages
Requires: filelock, fsspec, jinja2, networkx, nvidia-cublas-cu12, nvidia-cuda-cupti-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-runtime-cu12, nvidia-cudnn-cu12, nvidia-cufft-cu12, nvidia-curand-cu12, nvidia-cusolver-cu12, nvidia-cusparse-cu12, nvidia-nccl-cu12, nvidia-nvtx-cu12, sympy, triton, typing-extensions
Required-by: accelerate, peft, torchaudio, torchvision
---
Name: transformers
Version: 4.43.3
Summary: State-of-the-art Machine Learning for JAX, PyTorch and TensorFlow
Home-page: https://github.com/huggingface/transformers
Author: The Hugging Face team (past and future) with the help of all our contributors (https://github.com/huggingface/transformers/graphs/contributors)
Author-email: transformers@huggingface.co
License: Apache 2.0 License
Location: /workspaces/AI_Bias/venv_bias/lib/python3.10/site-packages
Requires: filelock, huggingface-hub, numpy, packaging, pyyaml, regex, requests, safetensors, tokenizers, tqdm
Required-by: peft
---
Name: accelerate
Version: 1.12.0
Summary: Accelerate
Home-page: https://github.com/huggingface/accelerate
Author: The HuggingFace team
Author-email: zach.mueller@huggingface.co
License: Apache
Location: /workspaces/AI_Bias/venv_bias/lib/python3.10/site-packages
Requires: huggingface_hub, numpy, packaging, psutil, pyyaml, safetensors, torch
Required-by: peft
---
Name: peft
Version: 0.18.1
Summary: Parameter-Efficient Fine-Tuning (PEFT)
Home-page: https://github.com/huggingface/peft
Author: The HuggingFace team
Author-email: benjamin@huggingface.co
License: Apache
Location: /workspaces/AI_Bias/venv_bias/lib/python3.10/site-packages
Requires: accelerate, huggingface_hub, numpy, packaging, psutil, pyyaml, safetensors, torch, tqdm, transformers
Required-by: 
---
Name: datasets
Version: 4.6.1
Summary: HuggingFace community-driven open-source library of datasets
Home-page: https://github.com/huggingface/datasets
Author: HuggingFace Inc.
Author-email: thomas@huggingface.co
License: Apache 2.0
Location: /workspaces/AI_Bias/venv_bias/lib/python3.10/site-packages
Requires: dill, filelock, fsspec, httpx, huggingface-hub, multiprocess, numpy, packaging, pandas, pyarrow, pyyaml, requests, tqdm, xxhash
Required-by: 
---
CHECKLIST!
[ ] accelerate config/test Done.
[ ] torch/transformers/accelerate/peft Recorded. 
[ ] repo current commit recorded.
[ ] Baseline A luanched and evaluated.
[ ] Baseline B luanched and evaluated.
[ ] Both Baselines' log file existed.
[ ] Both Baselines' save dir existed.
