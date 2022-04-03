<p align="center">
<img width="150px" src="https://user-images.githubusercontent.com/18649508/141604792-b256023d-c751-46d8-bab5-29a207d714ba.png"/>
</p>

<hr/>

<p align="center">
<a href="https://persiaml-tutorials.pages.dev" rel="nofollow"><img src="https://camo.githubusercontent.com/5f2d7c7e08b25fa4f95ce11be89982f25bb81bdef3e15f90b765aa91db371ff6/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f7475746f7269616c732d70617373696e672d677265656e" alt="tutorials" data-canonical-src="https://img.shields.io/badge/tutorials-passing-green" style="max-width: 100%;"></a>
<a href="https://persiaml.pages.dev" rel="nofollow"><img src="https://camo.githubusercontent.com/bf535e4ed96252a7731419446fe108bb36ce4b91e4960630ceccf31558329193/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f646f63756d656e746174696f6e2d70617373696e672d677265656e" alt="Documentation Status" data-canonical-src="https://img.shields.io/badge/documentation-passing-green" style="max-width: 100%;"></a>
<a href="https://badge.fury.io/py/persia" rel="nofollow"><img src="https://camo.githubusercontent.com/d6f9832aba04a67bbb6643eb81bd6be2173e12e51150553f30f16b629f40dc73/68747470733a2f2f62616467652e667572792e696f2f70792f7065727369612e737667" alt="PyPI version" data-canonical-src="https://badge.fury.io/py/persia.svg" style="max-width: 100%;"></a>
<a href="https://pypi.org/project/persia/"><img src="https://pepy.tech/badge/persia" alt="PyPI downloads"></a>
<a href="https://hub.docker.com/u/persiaml"><img alt="Docker Pulls" src="https://img.shields.io/docker/pulls/persiaml/persia-cuda-runtime"></a>
<a href="https://github.com/PersiaML/Persia/blob/main/LICENSE" rel="nofollow"><img src="https://img.shields.io/github/license/PersiaML/Persia" alt="license" style="max-width: 100%;"></a>
</p>

<div align="center">
<a href="https://github.com/PersiaML/Persia/stargazers"><img src="https://reporoster.com/stars/PersiaML/Persia" /><a/>
</div>


*WARNING: THIS PROJECT IS CURRENTLY IN MAINTENANCE MODE, DUE TO COMPANY REORGANIZATION.*
  
**PERSIA** (**P**arallel r**E**commendation t**R**aining **S**ystem with hybr**I**d **A**cceleration)  is developed by [AI platform@Kuaishou Technology](https://www.kuaishou.com/en), collaborating with ETH. It is a PyTorch-based (the first public one to our best knowledge) system for training large scale deep learning recommendation models on commodity hardwares. It is capable of training recommendation models with up to 100 trillion parameters. To the best of our knowledge, this is the largest model size in recommendation systems so far. Empirical study on public datasets indicate PERSIA's significant advantage over several other existing training systems in recommendation [1]. Its efficiency and robustness have also been validated by multiple applications with 100 million level DAU at Kuaishou. 

*Disclaimer: The program is usable and has served several important businesses. However, the official English documentation and tutorials are still under heavy construction and they are a bit raw now. We encourage adventurers to try out PERSIA and contribute!*

## News

* [参数量卷到一百万亿！华人团队开源史上最大的推荐训练系统 PERSIA](https://archive.md/FbocB) (In Chinese. Title: PERSIA, the Largest Recommended Training System in the History of Open Source by Far)
* AI Engines in the "Short-video" Era: Eating 100 Trillion Parameters, Invited talk, Facebook, 2021.
* 单机训练速度提升 640 倍！独家解读快手商业广告模型 GPU 训练平台 PERSIA (In Chinese. Title: 640x Faster GPU Based Learning System for Ad Recommendation)
   * [[AI Front]](https://archive.is/2ii2L) [[中国日报]](https://archive.is/N8fK2) [[InfoQ]](https://archive.is/JESDU) [[CSDN]](https://archive.is/tpvkN) [[Tencent Cloud News]](https://archive.is/kLuaT) [[AcFun]](https://archive.md/vuPmb)
* 创新、平衡与大格局：快手商业化的慢与快 (In Chinese. Title: Innovation, Balance, and Big Picture: The Speed of Kwai Commercialization)
   * [[TechSir]](https://archive.is/EOQ18) [[China Daily]](https://archive.is/L2VJE) [[Sohu]](https://archive.is/aY66U)

## Links

* [GitHub Repository](https://github.com/PersiaML/PERSIA)
* [Tutorials](https://persiaml-tutorials.pages.dev/)
* [API documentation](https://persiaml.pages.dev/) (Under Construction)

## Discussion

Feel free to join our [Telegram Group](https://t.me/joinchat/fLlD66VX8PQxMmJh) for discussion!  

## References

1. Xiangru Lian, Binhang Yuan, Xuefeng Zhu, Yulong Wang, Yongjun He, Honghuan Wu, Lei Sun, Haodong Lyu, Chengjun Liu, Xing Dong, Yiqiao Liao, Mingnan Luo, Congfei Zhang, Jingru Xie, Haonan Li, Lei Chen, Renjie Huang, Jianying Lin, Chengchun Shu, Xuezhong Qiu, Zhishan Liu, Dongying Kong, Lei Yuan, Hai Yu, Sen Yang, Ce Zhang, & Ji Liu. (2021). [Persia: A Hybrid System Scaling Deep Learning Based Recommenders up to 100 Trillion Parameters.](https://arxiv.org/abs/2111.05897)

2. Ji Liu & Ce Zhang. (2021). [Distributed Learning Systems with First-order Methods](https://arxiv.org/pdf/2104.05245).

## License

This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.
