# RefGlitch-Bench: A Benchmark for Reference-based Gameplay Glitch Detection with Vision-Language Models

This repository contains the code for the paper [RefGlitch-Bench: A Benchmark for Reference-based Gameplay Glitch Detection with Vision-Language Models](https://arxiv.org/abs/2604.11082).

## Dataset

The **RefGlitch-Bench** dataset can be downloaded from [Hugging Face](https://huggingface.co/datasets/asgaardlab/RefGlitch).

Please download the dataset separately and update the local file paths in the scripts if needed.

## Environment Setup

We provide a `requirements.txt` file for the Python environment. `ffmpeg` is also required.

- The `Inference_qwen3vl_*.py` scripts are intended for a GPU environment because they load `Qwen3-VL-8B` model.

## Repository Structure

- `Inference_godot_gpt.py`: GPT-based reference-guided inference used for **RQ1**.
- `Inference_godot_gpt_noref.py`: GPT-based no-reference baseline corresponding to the RQ1 setup.
- `Inference_qwen3vl_lastclean_godot.py`: RESP-style sequential prompting with the last clean frame as the reference, used for **RQ2**.
- `Inference_qwen3vl_previous_godot.py`: Qwen3-VL sequential prompting baseline using the immediately previous frame, used for **RQ2**.
- `Inference_qwen3vl_random_godot.py`: Qwen3-VL sequential prompting baseline using a random earlier frame, used for **RQ2**.
- `extract_keyframes.py`: Utility script for extracting keyframes or uniformly sampled frames from videos.
- `godot_video_folders.txt`: List of RefGlitch video folders.
- `glitch_with_ref_godot`: Prompt template for reference-guided inference.
- `glitch_without_ref_godot`: Prompt template for single-frame inference without a reference image.
- `glitch_with_ref_buggy_godot`: Prompt template for reference-guided inference when the selected reference is itself buggy.

## Running the Code

### Examine the role of reference frames where reference frames are manually selected (RQ1)

`Inference_godot_gpt.py` is the main script; replacing GPT with other models should be straightforward. You can run it with:

```bash
python Inference_godot_gpt.py
```

The no-reference baseline can be run with:

```bash
python Inference_godot_gpt_noref.py
```

### Examine the role of reference frames where reference frames are automatically selected (RQ2)

The `Inference_qwen3vl_*.py` scripts are used for three different reference selection strategies. You can run them with:

```bash
python Inference_qwen3vl_lastclean_godot.py
python Inference_qwen3vl_previous_godot.py
python Inference_qwen3vl_random_godot.py
```

## Notes

- Some file paths in the scripts are currently hard-coded for the original experiment setup. You may need to update them to match your local dataset layout.

## Citation

If you use this code or the dataset, please cite:

```bibtex
@misc{yu2026refglitch-bench,
  title={RefGlitch-Bench: A Benchmark for Reference-based Gameplay Glitch Detection with Vision-Language Models},
  author={Yakun Yu and Ashley Wiens and Adrian Barahona-Rios and Benedict Wilkins and Saman Zadtootaghaj and Nabajeet Barman and Cor-Paul Bezemer},
  year={2026},
  eprint={2604.11082},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
