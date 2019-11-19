## Overview

This branch host the minimal implementation of simulations that are presented in the document 
*"on recovering latent factors from sampling and firing graph"*
 on which relies the deyep algorithm. The document is accessible at [Arxiv web site](https://arxiv.org/abs/1909.09493) 

## Running simulations (instruction for Ubuntu)

 * Clone git firing_graph repository
    
    ```
    cd $HOME
    mkdir lib 
    cd lib/
    git clone https://github.com/pierreGouedard/firing_graph.git
    cd firing_graph
    git checkout publi_1
   ```

 * Run firing_graph's  unit tests
    
    ```
    cd $HOME/lib/firing_graph
    conda env create -f environment.yml
    conda activate fg-env
    python -m unittest discover tests
    conda deactivate
   ```


* Clone git deyep repository
    
    ```
    cd $HOME
    git clone https://github.com/pierreGouedard/deyep.git
   ```
  
 
 * Create conda env and set PYTHONPATH environment variable
  
    ``` 
    cd $HOME/deyep
    conda env create -f environment.yml 
    export PYTHONPATH="$HOME/lib:$HOME/deyep:$PYTHONPATH"
   ```

 * Run simulation

    ``` python simulations/signal_plus_noise_1.py &&
        python simulations/signal_plus_noise_2.py &&
        python simulations/signal_plus_noise_3.py &&
        python simulations/sparse_identification.py &&
        python simulations/sparse_identification_2
    ``` 

## Additional tips
Due to uncomplete conda package texlive-core, the matplotlib plots may not be able to display if install in conda env. A solution is to install texlive-full package outside env

```sudo apt-get install texlive-full```

To uninstall run 

```sudo apt-get purge texlive-*```

