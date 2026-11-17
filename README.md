# GVHMR: World-Grounded Human Motion Recovery via Gravity-View Coordinates
### [Project Page](https://zju3dv.github.io/gvhmr) | [Paper](https://arxiv.org/abs/2409.06662)

> World-Grounded Human Motion Recovery via Gravity-View Coordinates  
> [Zehong Shen](https://zehongs.github.io/)<sup>\*</sup>,
[Huaijin Pi](https://phj128.github.io/)<sup>\*</sup>,
[Yan Xia](https://isshikihugh.github.io/scholar),
[Zhi Cen](https://scholar.google.com/citations?user=Xyy-uFMAAAAJ),
[Sida Peng](https://pengsida.net/)<sup>†</sup>,
[Zechen Hu](https://zju3dv.github.io/gvhmr),
[Hujun Bao](http://www.cad.zju.edu.cn/home/bao/),
[Ruizhen Hu](https://csse.szu.edu.cn/staff/ruizhenhu/),
[Xiaowei Zhou](https://xzhou.me/)  
> SIGGRAPH Asia 2024

<p align="center">
    <img src=docs/example_video/project_teaser.gif alt="animated" />
</p>

## News 🔥

- [2025-03-08] By default not using DPVO. We implemented a SimpleVO, which is more efficient and compatible with GVHMR.
- [2025-03-08] We added a new option `f_mm` to specify the focal length of the fullframe camera in mm.

## Setup

Please see [installation](docs/INSTALL.md) for details.

## Quick Start

### [<img src="https://i.imgur.com/QCojoJk.png" width="30"> Google Colab demo for GVHMR](https://colab.research.google.com/drive/1N9WSchizHv2bfQqkE9Wuiegw_OT7mtGj?usp=sharing)

### [<img src="https://s2.loli.net/2024/09/15/aw3rElfQAsOkNCn.png" width="20"> HuggingFace demo for GVHMR](https://huggingface.co/spaces/LittleFrog/GVHMR)

### Demo
Demo entries are provided in `tools/demo`. Use `-s` to skip visual odometry if you know the camera is static, otherwise the camera will be estimated by DPVO.
We also provide a script `demo_folder.py` to inference a entire folder.
```shell
python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s
python tools/demo/demo_folder.py -f inputs/demo/folder_in -d outputs/demo/folder_out -s
```

### Desktop viewer

After completing [installation](docs/INSTALL.md), install the additional GUI and optional transcription dependencies with
`pip install -r tools/demo/requirements-gui.txt`. Install `ffmpeg` and `ffprobe` on your PATH for audio and embedded subtitle extraction; install `tesseract` for OCR. Launch with:

```shell
python tools/demo/gui.py
```

Select a video and run GVHMR to view the input, person tracking, 2D keypoints, in-camera, global, and combined renders. Runs are saved in separate folders under the selected output directory, including the original `hmr4d_results.pt` predictions and intermediate files. The model requires a compatible CUDA GPU and the checkpoints and body models described in the installation guide.

The **Explore outputs** tab opens automatically generated predictions, or a previous `hmr4d_results.pt` via **Open predictions**. Select global or in-camera coordinates to view the reconstructed SMPL-X body: drag to rotate, use the toolbar to zoom, and move the frame slider or press Play. The output dropdown includes position and rotation plots for the head and each shoulder, elbow, and wrist. Positions show X/Y/Z in meters; rotations show the three local axis-angle components in radians, relative to the parent joint, not Euler angles. The dashed line follows the selected frame. Position includes overall body translation and, in camera coordinates, camera movement. Select any numerical output and component to plot its values over frames; static arrays are plotted by element instead. The mesh and joint positions are reconstructed on CPU using the neutral SMPL-X body model in the selected GVHMR checkout. The demo converts input video to 30 fps, which the explorer timeline assumes. Values are estimates, not guaranteed measurements; individual finger and facial movements are not predicted.

The **Extract audio and text** button independently saves the first audio stream as WAV and converts embedded text subtitle streams to SRT. Optional speech transcription uses faster-whisper's `small` model on CPU (it downloads model weights on first use); OCR uses Tesseract on one frame every two seconds. Results, missing streams, and errors appear in the Audio and text tab and are saved with a manifest. Image-based subtitle streams may not be convertible to text; use OCR for visible captions. Extracted audio opens in the system's media player.

These are local sidecar features: **GVHMR uses only the video, not the extracted text or audio**. Extraction may be incomplete due to speech recognition, OCR, sampling, or subtitle format limitations.

### Reproduce
1. **Test**:
To reproduce the 3DPW, RICH, and EMDB results in a single run, use the following command:
    ```shell
    python tools/train.py global/task=gvhmr/test_3dpw_emdb_rich exp=gvhmr/mixed/mixed ckpt_path=inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt
    ```
    To test individual datasets, change `global/task` to `gvhmr/test_3dpw`, `gvhmr/test_rich`, or `gvhmr/test_emdb`.

2. **Train**:
To train the model, use the following command:
    ```shell
    # The gvhmr_siga24_release.ckpt is trained with 2x4090 for 420 epochs, note that different GPU settings may lead to different results.
    python tools/train.py exp=gvhmr/mixed/mixed
    ```
    During training, note that we do not employ post-processing as in the test script, so the global metrics results will differ (but should still be good for comparison with baseline methods).

# Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{shen2024gvhmr,
  title={World-Grounded Human Motion Recovery via Gravity-View Coordinates},
  author={Shen, Zehong and Pi, Huaijin and Xia, Yan and Cen, Zhi and Peng, Sida and Hu, Zechen and Bao, Hujun and Hu, Ruizhen and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia Conference Proceedings},
  year={2024}
}
```

# Acknowledgement

We thank the authors of
[WHAM](https://github.com/yohanshin/WHAM),
[4D-Humans](https://github.com/shubham-goel/4D-Humans),
and [ViTPose-Pytorch](https://github.com/gpastal24/ViTPose-Pytorch) for their great works, without which our project/code would not be possible.
