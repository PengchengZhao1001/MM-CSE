# Multimodal Class-aware Semantic Enhancement Network for Audio-Visual Video Parsing
Pengcheng Zhao*, Jinxing Zhou*, Yang Zhao, Dan Guo^, Yanxiang Chen^

Official code for AAAI 2025 paper: [Multimodal Class-aware Semantic Enhancement Network for Audio-Visual Video Parsing](https://arxiv.org/abs/2412.11248)
***

Before running, reader needs to dowlond the features of LLP dataset from [VALOR](https://github.com/Franklin905/VALOR),  and places them in folder **data**.

Folder **data** contains the annotations for the LLP dataset.

Folder **code** contains the code for audio-visual video parsing, and our trained model weights.

## Evaluation

For code/main_avvp.py

``--mode test --test_weights ./checkpoint_best.pth``

## Citation

```
@article{zhao2024multimodal,
  title={Multimodal Class-aware Semantic Enhancement Network for Audio-Visual Video Parsing},
  author={Zhao, Pengcheng and Zhou, Jinxing and Guo, Dan and Zhao, Yang and Chen, Yanxiang},
  journal={arXiv preprint arXiv:2412.11248},
  year={2024}
}
```

## Acknowledgement
The core of our code is built upon on [VALOR](https://github.com/Franklin905/VALOR).  Thanks for their excellent work.
