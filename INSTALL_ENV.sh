
#python = 3.11
conda install -y pip

#stl
conda install -y pandas
conda install -y numpy
conda install -y matplotlib
conda install -y seaborn
conda install -y tqdm
conda install -y ipykernel
conda install -y jupyter

#data processing
conda install -y scikit-learn
conda install -y networkx
conda install -y scipy

#kernel
python -m ipykernel install --user --name="pyg_CUDA"

#ai stack (torch = 2.2, cuda = 11.8, pyg = 2.5.0) --> compatible with cuda 12.2
conda install pytorch==2.2.1 torchvision=0.17.1 torchaudio=2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia

#graph stack, pyg buggy with conda
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install missingno