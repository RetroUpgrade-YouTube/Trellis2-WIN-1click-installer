# 🌀 ComfyUI Wrapper for [https://github.com/microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
---
<img width="883" height="566" alt="{09272892-57D6-4EB8-B27B-6B875916982A}" src="https://github.com/user-attachments/assets/a7788f13-141c-4072-9143-b8b1ee1ead2a" />
---
<img width="980" height="579" alt="{F6FE6B7B-94B7-44C6-8C89-02E7C81EBF7E}" src="https://github.com/user-attachments/assets/ad27111c-beb8-48ef-8613-c533a3a5cacd" />
---

**DEVELOPMENT IN PROGRESS - PLEASE WAIT**

---

**REQUIREMENTS**

You need to have access to facebook dinov3 models in order to use Trellis.2 [https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)

Save the model in ComfyUI models folder under "facebook/dinov3-vitl16-pretrain-lvd1689m"

So in ComfyUI/models/facebook/dinov3-vitl16-pretrain-lvd1689m

---

## ⚙️ Installation Guide

> Tested on **Windows 11** with **Python 3.11** with **Torch = 2.7.0 + cu128**.

### 1. Install Wheel

#### For a standard python environment:

```bash
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/nvdiffrast-0.4.0-cp311-cp311-win_amd64.whl
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/nvdiffrec_render-0.0.0-cp311-cp311-win_amd64.whl
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/flex_gemm-0.0.1-cp311-cp311-win_amd64.whl
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/o_voxel-0.0.1-cp311-cp311-win_amd64.whl
```

**If you use Torch v2.7.0:**
```bash
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/Cumesh/Torch270/cumesh-0.0.1-cp311-cp311-win_amd64.whl
```

**If you use Torch v2.8.0:**
```bash
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/Cumesh/Torch280/cumesh-0.0.1-cp311-cp311-win_amd64.whl
```

---

#### For ComfyUI Portable:

```bash
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\nvdiffrast-0.4.0-cp311-cp311-win_amd64.whl
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\nvdiffrec_render-0.0.0-cp311-cp311-win_amd64.whl
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\flex_gemm-0.0.1-cp311-cp311-win_amd64.whl
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\o_voxel-0.0.1-cp311-cp311-win_amd64.whl
```

**If you use Torch v2.7.0:**
```bash
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\Cumesh\Torch270\cumesh-0.0.1-cp311-cp311-win_amd64.whl
```

**If you use Torch v2.8.0:**
```bash
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\Cumesh\Torch280\cumesh-0.0.1-cp311-cp311-win_amd64.whl
```


