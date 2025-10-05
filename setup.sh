apt-get install git-lfs -y
git lfs install
apt-get install libgl1-mesa-glx -y

cd baselines/il_lib/
python -m pip install -e .
python -m pip install latex2sympy2-extended==1.10.2
cd ../../

cd BEHAVIOR-1K
python -m pip install git+https://github.com/cnr-isti-vclab/PyMeshLab.git@v2022.2
python -m pip install -e bddl
python -m pip install git+https://github.com/huggingface/lerobot@577cd10974b84bea1f06b6472eb9e5e74e07f77a
python -m pip install -e OmniGibson[eval]
cd ..

