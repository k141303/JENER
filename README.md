# JENER
[拡張固有表現階層v9.0](http://liat-aip.sakura.ne.jp/ene/ene9/definition_jp/index.html)の名前以下195カテゴリーを対象とした日本語固有表現抽出器(JENER: Japanese Extended Named Entity Recognizer)

## インストール方法
[Python](https://www.python.org/downloads/)と[git-lfs](https://github.com/git-lfs/git-lfs?utm_source=gitlfs_site&utm_medium=installation_link&utm_campaign=gitlfs#installing)のインストールが必要です。

```bash
git clone https://github.com/k141303/JENER.git
cd JENER
git lfs pull
pip install .
```

## 使い方

### コマンドライン

```bash
python3 -m jener "木村拓哉は日本の俳優であり、SMAPの元メンバーである。"
# {'surf': '木村拓哉', 'span': (0, 4), 'ENEs': ['人名']}
# {'surf': '日本', 'span': (5, 7), 'ENEs': ['地名>ＧＰＥ>国名']}
# {'surf': '俳優', 'span': (8, 10), 'ENEs': ['プロダクト名>称号名>地位職業名']}
# {'surf': 'SMAP', 'span': (14, 18), 'ENEs': ['組織名>公演組織名']}
```

### スクリプト

```python
from jener import JENER
jener = JENER()
predicts = jener("木村拓哉は日本の俳優であり、SMAPの元メンバーである。")
print(predicts)
# [{'surf': '木村拓哉', 'span': (0, 4), 'ENEs': ['人名']}, {'surf': '日本', 'span': (5, 7), 'ENEs': ['地名>ＧＰＥ>国名']}, {'surf': '俳優', 'span': (8, 10), 'ENEs': ['プロダクト名>称号名>地位職業名']}, {'surf': 'SMAP', 'span': (14, 18), 'ENEs': ['組織名>公演組織名']}]
```
