#git clone the 1.3.17 branch of the mmcv
git clone -b v1.3.17 https://github.com/open-mmlab/mmcv.git

# first run on the non-interactive session
# with internet connection check the requirement and let it fail
cd mmcv
module load cuda
MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -e .

# create an interactive session with gpu
# run the following command again
salloc --partition=gpu --gres=gpu:p100:1 --time=1:00:00 --mem=10GB --account=nevatia_174
module load cuda
cd mmcv
MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -e .

