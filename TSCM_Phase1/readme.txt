conda config --remove-key channels
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes
conda create -n tscm python=3.10 numpy scipy pandas matplotlib statsmodels -y
conda activate tscm
python -m pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ lifelines
python run_experiment.py


============================================================
TSCM Phase 1 实验结果
============================================================
对照组前250轮平均崩溃率: 0.1234
对照组后250轮平均崩溃率: 0.1201
对照组变化: -2.7%

实验组前250轮平均崩溃率: 0.1245
实验组后250轮平均崩溃率: 0.0876
实验组变化: -29.6%

实验组崩溃率 Mann-Kendall 趋势检验: tau=-0.234, p=0.0003
有效维度: 前250轮平均 28.3 → 后250轮平均 17.1

✅ 假说得到支持：崩溃率显著下降，禁忌标记有效
============================================================