## Overview

The simulations presented in the paper  *"On recovering latent factors from sampling and firing graph"*, 
accessible at [Arxiv web site](https://arxiv.org/abs/1909.09493), are now avalable in the git project "firing graph". In 
ordrer to access and run simulations, please follow intrusctions below:

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

Follow instruction in the readme.md and run the simulations ! 

## Additional tips
Due to uncomplete conda package texlive-core, the matplotlib plots may not be able to display correctly. A solution is to install texlive-full package outside env

```sudo apt-get install texlive-full```

To uninstall run 

```sudo apt-get purge texlive-*```

