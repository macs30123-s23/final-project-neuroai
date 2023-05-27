# final-project-neuroai

This is the GitHub repository for the final project of MACS 30123 Large Scale Computing.

## Background

The progress made in Deep Neural Networks (DNN) has brought about significant transformations in the field of artificial intelligence. Moreover, for scientists who are dedicated to understanding and replicating the intricate workings of natural intelligence, DNNs have emerged as the most effective model for simulating the primate visual system along the ventral pathway([Yamins and DiCarlo, 2016](https://www.nature.com/articles/nn.4244)).  Systematic comparision between DNNs and primate visual system suggests that while not constrained by neural data, the two systems converges at the representation of objects (Figure 1). 

![Fig.1](https://p.ipic.vip/p9op9m.png)

**Figure 1.** Demostration of comparision between the brain and DNNs. Adapeted from [(Yamins and DiCarlo, 2016)](https://www.nature.com/articles/nn.4244).

Most of the comparisions focus on the high-level representation of object, although excpetions do exist [(Pospisil et al., 2018)](https://elifesciences.org/articles/38242). In a study utilizing intrincic optical imaging, it was discovered that simple stimuli, which were previously believed to be solely processed in the lower-level cortex, are also represented in the IT cortex, the final destination of the ventral visual pathway. Importantly, such representation becomes more scarce as the hierachy goes up. 

While the sparce coding property has been probed before [(Liu et al., 2020)](https://www.frontiersin.org/articles/10.3389/fncom.2020.578158/full), the stimuli the authors focused was still the high-level representation of object. 

The question remains as whether the representaion of simiple geometry would be represented in deeper layers of DNN, and more critically, what is the trend of propotion of selective units along the hiearchy. 

## Model 

I used CORnet [(Kubilius et al., 2018)](https://www.biorxiv.org/content/10.1101/408385v1.abstract) as the target model for the comparision. CORnets are designed to directly micic ventral pathway of the primate visual system and come with both DNN and RNN. In particular, CORnet-Z is the simple feedforward version, with each layer has a direct compariable cortex. All of the CORnets are optimized for ImageNet classification task. 

<img src="https://p.ipic.vip/tfunhu.png" alt="Screenshot 2023-05-26 at 19.20.41" style="zoom:50%;" />

**Figure 2.** Architecture of CORnet-Z. Adapted from [(Kubilius et al., 2018)](https://www.biorxiv.org/content/10.1101/408385v1.abstract).

## Stimuli

In the *'Artiphysiology'* experiment, I followed the same stimuli with compariable parameters that were used by intrincic optical imaging. Note for the *in vivo* experiements, another feafure that is probed is *motion direction*. However, as the reference model is the feedforward CORnet-Z, images should be static. To compensate the fact that stimuli are static and to offer as many stimuli as possible, I exhaustively explored all possible combinations of stimuli.  

Two types of selectivity are considered in the experiment:

- orientation selectivity
- shape selectivity

### Orientation

Four conditions of stimuli are generated and treated as two groups:

- Horizontal VS Vertival
- Tilted by 45 and Tilted by 134

For the generation code, please refer to 

<img src="https://p.ipic.vip/kijxas.png" alt="example" style="zoom:24%;" />

**Figure 3.** Examples of Orientation stimuli





## References

Yamins, D. L., & DiCarlo, J. J. (2016). Using goal-driven deep learning models to understand sensory cortex. *Nature neuroscience*, *19*(3), 356-365.

Pospisil, D. A., Pasupathy, A., & Bair, W. (2018). 'Artiphysiology'reveals V4-like shape tuning in a deep network trained for image classification. *Elife*, *7*, e38242.

Liu, X., Zhen, Z., & Liu, J. (2020). Hierarchical sparse coding of objects in deep convolutional neural networks. *Frontiers in computational neuroscience*, *14*, 578158.

Kubilius, J., Schrimpf, M., Nayebi, A., Bear, D., Yamins, D. L., & DiCarlo, J. J. (2018). Cornet: Modeling the neural mechanisms of core object recognition. *BioRxiv*, 408385.



