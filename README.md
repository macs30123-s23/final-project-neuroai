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

![model_parameter](https://p.ipic.vip/q3mh6a.jpg)

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

For the generation code, please refer to [1.0_gen_grating_HV.py](https://github.com/macs30123-s23/final-project-neuroai/blob/main/1.0_gen_grating_HV.py) and [1.1_gen_grating_tilt.py](https://github.com/macs30123-s23/final-project-neuroai/blob/main/1.1_gen_grating_tilt.py).

The policy for generating such images is: 

1. Generate a large background image 
2. Move the slicing window across the image: window size = 224x224
3. Moving step: 1

<img src="https://p.ipic.vip/kijxas.png" alt="example" width="400"  />

**Figure 3.** Examples of Orientation stimuli. The horizontal and vertical stimuli each has 39 examplars, and the tilted version each has 28 exemplars. 

### Shape

Two levels of shape stimuli are generated: Circle and Triangle. Note for tirangle, I introduced some randomness of orientation. 

For the generation code, please refer to [1.2_gen_shape.py](https://github.com/macs30123-s23/final-project-neuroai/blob/main/1.2_gen_shape.py).

The basic policy for generating such image is:

1. Generate base image (one circle or triangle)
2. Exapnd the base image to the background image
3. Move the slicing window across the image: window size = 224x224
4. Moving step: 1

<img src="https://p.ipic.vip/zjcmc4.png" alt="shape_example" width="400" />

**Figure 4.** Examples of Shape stimuli. Each shape has 529 examplars. However, in this study, 40 examplars are randomly drawn from the stimuli pool. 

## Computation Challenges

To probe the selectivity of each unit at each layer, I have to feed one image to the neural network, truncate the model at the layer of interest, record the activation of the unit, store the actications across stimuli, and ultimately perform independent sample t-test to determin the selectivity of the unit. The whole process will ended in a nested for loop.

```py
laysers = ['V1', 'V2', 'V4', 'IT']
jobs = ['grating_hv', 'grating_tilt', 'shape']
for layer in layers: # primary loop, select the layer of interest
  for job in jobs: # secondary loop, selsect the type of stimuli
    for im in im_pool: # tertiary loop, feed one image at one time
      for unit in units: # quaternary loop, probe the selectivity of the unit
        p = get_p_value_wiht_ind_t-test()
        if p < 0.05:
          #True
```

- number of units across layers
  - 'V1': 200704 
  - 'V2': 100352
  -  'V4': 50176
  - 'IT': 25088
- 4 types of jobs 
- each job average 40 stimuli

The total number of operation is 376320 * 4 * 40 (around **60 million**).

I implemented the nested_loop version, please refer to [2.0_test_nn.py](https://github.com/macs30123-s23/final-project-neuroai/blob/main/2.0_test_nn.py). Given the fact that the examplars are not large, the nested loop version would work at small scale, but certainly not appliable if the experiment scales up. 

To tackle the nested loop problem, I offer two solutions:

### Use Lambda Function

Given the fact that the fundamental operation unit is to hold the activation of one unit across mulitple stimuli, and then compare the activation through statistical inference, the lambda function includes:

1. Target for one job
2. Target for one layer
3. Target for one unit
4. Records the activations under multiple stimuli
5. Perfrom statistical inference 
6. Return a boolean value
7. Store to S3 bucket

### Use MPI

Parallel solution to the nested loop. 

To implement the nested loop in parallel, the logic is bottom up:

1. quaternary loop, parallel across units
   1. As the turncated model returns an array of units, it is much slower if iterate through every unit, especially when one layer contains millions of units. This process could be parallelized by vectrize the actications as a matrix, with each row representing the observations (number of pictures), each columns as the unit, and returns a boolean array for each layer in each job
2. tertiary loop, parallel across stimuli
   1. pytorch supports feeding a batch of images and returns a tensor with the row as observations and columns as features
3. sencondary and primary loop, parallel across job and layer 
   1. Using MPI, the 12 combinations of jobs and layers are parallelized and treated independently. The parallel execution allows for efficient processing of each combination, and the results are then combined or aggregated to obtain the final summary.

For the immplementation of MPI, refer to [3.0_test_nn_mpi.py](https://github.com/macs30123-s23/final-project-neuroai/blob/main/3.0_test_nn_mpi.py).

## Results

The selectivity of units are defined by the different activation across stimuli. To define a unit to be orientation selective, it should be either significant in Horizontal VS Vertical or two tilted versions (effectively a logic_or of the two boolean maps.) To define a unit to be shape selective, it should be significant between two shape stimuli. For the conjunction unit, it should be significant in both selectivity (effectivly a logic_and of the two selective maps).

![Orientation Selective](https://p.ipic.vip/2mk256.png)

![Shape Selective](https://p.ipic.vip/hunbek.png)

![Conjunction](https://p.ipic.vip/ou1pj0.png)

The three figures shows the proportion of units that are selective in three cases. 

Notablely, when camparing the performance of V4 and IT, the trend converages to the observation in vivo （As the in vivo study is still on-going, I will not show the exact data here). 

## Limitations

Although this study presents preliminary results, it is important to acknowledge that further tests regarding the rigor of the observations need to be considered:

1. Multi-comparision Correction.
2. Expanding examplar size, with white noise
3. Effect size (Cohen's d) as discrimination measruement
4. Constructing confidence interval using bootstrapping for cross-layer comparision

## References

Yamins, D. L., & DiCarlo, J. J. (2016). Using goal-driven deep learning models to understand sensory cortex. *Nature neuroscience*, *19*(3), 356-365.

Pospisil, D. A., Pasupathy, A., & Bair, W. (2018). 'Artiphysiology'reveals V4-like shape tuning in a deep network trained for image classification. *Elife*, *7*, e38242.

Liu, X., Zhen, Z., & Liu, J. (2020). Hierarchical sparse coding of objects in deep convolutional neural networks. *Frontiers in computational neuroscience*, *14*, 578158.

Kubilius, J., Schrimpf, M., Nayebi, A., Bear, D., Yamins, D. L., & DiCarlo, J. J. (2018). Cornet: Modeling the neural mechanisms of core object recognition. *BioRxiv*, 408385.



