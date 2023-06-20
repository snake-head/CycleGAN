# CycleGAN
Implement mutual conversion between virtual endoscope and real endoscope images based on cyclgan. Realize defogging of endoscopic videos.

## Depth Estimation

![image](https://github.com/snake-head/CycleGAN/assets/62976678/45fd74c4-2f96-4679-823c-5b718add1883)

**Table 1. Performance on synthetic colonoscopy datasets of DenseDepth**
|     Method     | 𝐿1 (cm) | 𝐿𝑅𝑀𝑆𝐸 (cm) | 𝐿𝑟𝑒𝑙 (%) |
| :------------: | :-----: | :----------: | :---------: |
|  DenseDepth    |  0.038  |    0.066     |    1.22     |

## Pose Estimation

![image](https://github.com/snake-head/CycleGAN/assets/62976678/184dc680-556a-44bc-bb14-7541b2c0e3c7)

**Table 2. Performance on synthetic colonoscopy datasets of SC-SfMLearner**
|       Trajectory Scale       | ATE (cm) | RTE (mm) | ROT (°) |
| :-------------------------: | :------: | :------: | :-----: |
|        Frames_S9             |   1.03   |   3.34   |  0.12   |
|        Frames_B9             |   1.04   |   2.72   |  0.14   |
|        Frames_S4             |   1.01   |   6.16   |  0.15   |
|        Frames_B4             |   1.05   |   5.10   |  0.15   |
|        Frames_S14            |   1.00   |   7.99   |  0.15   |
|        Frames_B14            |   1.05   |   3.37   |  0.16   |
|          Average             |   4.78   |  0.145   |  0.18   |


## Style Migration

![image](https://github.com/snake-head/CycleGAN/assets/62976678/5c0a6e66-4203-435d-8a92-462282c1808b)

**Results**

![image](https://github.com/snake-head/CycleGAN/assets/62976678/7f5e04f5-19eb-4ea0-a4b9-04eab1946940)
