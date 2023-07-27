salloc --time=48:00:00 --partition=gpu --gres=gpu:v100:2 --account=nevatia_174 --ntasks=1 --cpus-per-task=4 --mem=40GB
salloc --time=12:00:00 --partition=gpu --gres=gpu:p100:2 --account=nevatia_174 --ntasks=1 --cpus-per-task=4 --mem=40GB
salloc --time=12:00:00 --partition=gpu --gres=gpu:a100:2 --account=nevatia_174 --ntasks=1 --cpus-per-task=4 --mem=40GB
salloc --time=12:00:00 --partition=gpu --gres=gpu:a100:2 --account=nevatia_174 --mem=40GB --exclusive
salloc --time=1:00:00 --partition=gpu --gres=gpu:a40:2 --account=nevatia_174 --mem=40GB --exclusive
salloc --partition=gpu --gres=gpu:a100:2 --mem=32GB --exclusive
salloc --time=4:00:00 --partition=gpu --gres=gpu:p100:2 --account=nevatia_174 --ntasks=1 --cpus-per-task=4 --mem=80GB --exclusive