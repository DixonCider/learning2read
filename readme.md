# Machine Learning Techniques, Spring 2018, Project
`learning2read/` 套件
+ code共享區
+ 盡量包成class、名稱取長一點沒關係

`report/` 文件
+ 取名方式4001,5001,6001
    + 4001-ReportName.ipynb
    + 4001-ReportName.pdf
    + 4001-ReportName.py
    + 4001-ReportName.c
    + 4001-ReportName.sh

`playground/` (local)
+ `.gitignore`裡有一行`playground/*`，本地實驗用

## Commit
+ 開工前別忘了先拉一下`git pull`
+ [Add short codes prefixes for commits to style guide](https://github.com/quantopian/zipline/issues/96)
    + SciPy/NumPy所使用加在commit message前的短語
    + 看得懂就好，不是必要XD


> Links
> + [Slack](https://learning2read.slack.com/)
> + [FinalProject說明（規則、report格式）](https://www.csie.ntu.edu.tw/~htlin/course/mltech18spring/project/)
> + [Track1-Scoreboard](https://learner.csie.ntu.edu.tw/judge/ml18spring/track1/scoreboard/)

# learning2read Installation
1. git pull這個repo
2. 在python/虛擬環境目錄下建symbolic link
    + 有點髒我知道但它很方便 :p
    + windows
        + `mklink /d learning2read "I:\Dropbox\_a5_Projects\EmptyCup106-2\MLTech-Final\learning2read"`
        + `mklink /d src dst`
    + linux
        + `ln -s /Users/qtwu/Dropbox/_a5_Projects/EmptyCup106-2/MLTech-Final/learning2read /Users/qtwu/anaconda/lib/python3.6/site-packages`
        + `ln -s dst src`

# Dependency

+ `pandas` DataFrame
+ `numpy` matrix,linalg
+ `scipy` stats
+ `sklearn` a.k.a. scikit-learn, 各種ML模型和測試工具
+ `pyecharts` 百度所支持畫圖套件Echarts的python接口
+ `path.py` 檔案路徑小工具（別懷疑，它名稱就叫path.py）
+ `lightgbm` Microsoft支持的Gradient Boost Descision Tree

## 觀望中
+ `pytorch` Facebook支持的深度學習庫
    + [PyTorch到底好用在哪里? - 知乎](https://www.zhihu.com/question/65578911)