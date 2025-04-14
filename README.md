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
@article{Zhao_Zhou_Zhao_Guo_Chen_2025, 
  title={Multimodal Class-aware Semantic Enhancement Network for Audio-Visual Video Parsing},
  author={Zhao, Pengcheng and Zhou, Jinxing and Zhao, Yang and Guo, Dan and Chen, Yanxiang},
  volume={39}, 
  url={https://ojs.aaai.org/index.php/AAAI/article/view/33134}, 
  DOI={10.1609/aaai.v39i10.33134}, 
  number={10}, 
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  year={2025}, 
  month={Apr.}, 
  pages={10448-10456}
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
