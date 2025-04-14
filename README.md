# Multimodal Class-aware Semantic Enhancement Network for Audio-Visual Video Parsing
Pengcheng Zhao*, Jinxing Zhou*, Yang Zhao, Dan Guo^, Yanxiang Chen^

Official code for AAAI 2025 paper: [Multimodal Class-aware Semantic Enhancement Network for Audio-Visual Video Parsing](https://doi.org/10.1609/aaai.v39i10.33134)
***

Before running, reader needs to dowlond the features of LLP dataset from [VALOR](https://github.com/Franklin905/VALOR),  and places them in folder **data**.

Folder **data** contains the annotations for the LLP dataset.

Folder **code** contains the code for audio-visual video parsing, and our trained model weights.

## Evaluation

For code/main_avvp.py

``--mode test --test_weights ./checkpoint_best.pth``

## Citation

```
@inproceedings{zhao2025multimodal,
  title={Multimodal class-aware semantic enhancement network for audio-visual video parsing},
  author={Zhao, Pengcheng and Zhou, Jinxing and Zhao, Yang and Guo, Dan and Chen, Yanxiang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={10},
  pages={10448--10456},
  year={2025}
}

@misc{zhao2024multimodalclassawaresemanticenhancement,
  title={Multimodal Class-aware Semantic Enhancement Network for Audio-Visual Video Parsing}, 
  author={Pengcheng Zhao and Jinxing Zhou and Yang Zhao and Dan Guo and Yanxiang Chen},
  year={2024},
  eprint={2412.11248},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2412.11248}, 
}
```

## Acknowledgement
The core of our code is built upon on [VALOR](https://github.com/Franklin905/VALOR).  Thanks for their excellent work.
