# Machine Learning Techniques, Spring 2018, Project
> Links
> + [mltech18spring/project](https://www.csie.ntu.edu.tw/~htlin/course/mltech18spring/project/)
> + [Book-Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)
>     + 資料格式描述
>     + [CSV Dump [25.475 KB]](http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip)
>         + 尚不確定跟traning set的關係
>         + 1149780筆評分
>         + 271379本書
>         + 278858使用者
> + [learning2read.slack.com](https://learning2read.slack.com/)
> + [Add short codes prefixes for commits to style guide](https://github.com/quantopian/zipline/issues/96)
>     + SciPy/NumPy所使用加在commit message前的短語

# Machine Learning Packages/Classes (WIP)

Type|Package|Description
-|-|-
probability|sklearn.linear_model.LogisticRegression|MLE Estimator

# Installation
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

`pandas` DataFrame
`numpy` matrix,linalg
`scipy` stats
`sklearn` a.k.a. scikit-learn, 各種ML模型和測試工具
`pyecharts` plot
`path.py` 檔案路徑小工具（別懷疑，它名稱就叫path.py）
`lightgbm`

## 觀望中
`pytorch` facebook支持的深度學習庫
+ [PyTorch到底好用在哪里? - 知乎](https://www.zhihu.com/question/65578911)


## (windows)


## (osx / linux)