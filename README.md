# Fair yet Asymptotically Equal Collaborative Learning [ICML 2023]
This repo is for our paper: Fair yet Asymptotically Equal Collaborative Learning

## Environment setup
We use `conda` to manage our environment. To setup the environment, run the command in the terminal:
```
conda env create -f environment.yml
```
After that, you can activate it using:
```
conda activate Fair2Stage
```

## Code file explaination
We exploit two settings in our main experiments:
1) federated reinforcement learning
2) federated online incremental learning

For federated reinforcement learning, the corresponding script is `run_2stages_exp_fedrl.py` with parameters as follows:
```
--d: the name of the Atari game, ['Breakout', 'SpaceInvaders', 'Pong']
--b: the temperature parameter used in the softmax
--r: the proportion of nodes being selected each round
--st: the type of score used in the incentive, the default is 'proportion' ['random', 'reverse', 'proportion']
--split: the way to divide the dataset into datasets from different nodes, the default is 'uniform', ['uniform', 'classimbalance', 'powerlaw']
--T: the number of total training iterations
--T1: the number of iterations for contribution evaluation, the default is 0 which uses the hypothesis testing approach to determine the stopping iteration for contribution evaluation. Specify a positive number to stop at a specific iteration
--T2: the number of iterations for giving out incentives, do not need to specify if T is specified
--E: the number of local iterations in each node
--lr: the learning rate
--n: the name of the experiment
--dv: the data valuation method used, the default is cos_grad, ['fed_loo', 'cos_grad', 'mr']
--a: the way to vary the contributions of different nodes, the default is 'normal' which do not vary the contribution of different nodes, ['normal', 'reward_noise', 'state_noise','memory_size','exploration']
--nc: the number of nodes
--alpha: the parameter alpha (i.e., the threshold) for hypothesis testing
--samp_num: the parameter tau (i.e., the number of look-ahead iterations) for hypothesis testing
--samp_dim: the number of nodes used for hypothesis testing
```



For federated online incremental learning, the corresponding script is `run_2stages_exp_stream.py` with parameters as follows:
```
--d: the name of the dataset used, ['mnist', 'cifar10', 'hft', 'electricity', 'pathmnist']
--b: the temperature parameter used in the softmax
--r: the proportion of nodes being selected each round
--st: the type of score used in the incentive, the default is 'proportion' ['random', 'reverse', 'proportion']
--split: the way to divide the dataset into datasets from different nodes, the default is 'uniform', ['uniform', 'classimbalance', 'powerlaw']
--T: the number of total training iterations
--T1: the number of iterations for contribution evaluation, the default is 0 which uses the hypothesis testing approach to determine the stopping iteration for contribution evaluation. Specify a positive number to stop at a specific iteration
--T2: the number of iterations for giving out incentives, do not need to specify if T is specified
--E: the number of local iterations in each node
--lr: the learning rate
--n: the name of the experiment
--dv: the data valuation method used, the default is cos_grad, ['fed_loo', 'cos_grad', 'mr']
--noise: the type of noise used to vary the contributions of the nodes, the default is 'normal' which does not add any noise, ['normal', 'label_noise_different', 'feature_noise_different', 'powerlaw', 'missing_values',nonstatinary_label_noise_inc', 'nonstatinary_label_noise_dec', 'nonstatinary_label_noise_both']
--nc: the number of nodes
--max_noise: the maximum magnitude of the noise added to the dataset
--alpha: the parameter alpha (i.e., the threshold) for hypothesis testing
--samp_num: the parameter tau (i.e., the number of look-ahead iterations) for hypothesis testing
--samp_dim: the number of nodes used for hypothesis testing
```

## Dataset preparation
For MNIST and CIFAR-10, we integrate the downloading of these two datasets in our code. Therefore, no futher effort is needed to run the following code.

For high frequency trading (HFT) dataset, the dataset is avaliable at https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649. You can download it and unzip the `Benchmark` folder to `./datasets`.

For electricity loads time series prediction (ELECTRICITY), the dataset is avaliable at https://data.open-power-system-data.org/time_series/2020-10-06. You can directly run the jupyter note book in the `./datasets` folder to download and pre-process it.

For PathMNIST(PATH), the homepage of the dataset is at https://medmnist.com/. You can download the `pathmnist.npz` file from https://zenodo.org/record/6496656#.YoXgofNBz0o and put it under `./datasets`.

## Examples
For each script, we provide an example to run it.

### Federated reinforcement learning
`python run_2stages_exp_fedrl.py -d Breakout -T 450 -nc 10 -alpha 0.95 -a state_noise -n state`

### Federated online incremental learning
`python run_2stages_exp_stream.py -d mnist -T 150 -noise label_noise_different -max_noise 0.2 -E 1 -b 150 -alpha 0.7`