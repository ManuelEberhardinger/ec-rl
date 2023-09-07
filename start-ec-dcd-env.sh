HOME=/container/
export HOME
PATH="/usr/local/conda/bin:/container/pypy3.6-v7.3.3-linux64/bin:$PATH"
export PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/miniconda3/lib/:/opt/miniconda3/envs/my_new_env/lib/
export CONDA_AUTO_ACTIVATE_BASE=false
eval `opam config env`
cd /home/ma/e/eberhardinger/workspaces/ec/
if [ $# = 0 ] ; then exec bash --rcfile /home/ma/e/eberhardinger/.bashrc; else exec "$@"; fi
