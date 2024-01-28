# JENER
拡張固有表現階層を対象とした日本語固有表現抽出器(JENER: Japanese Extended Named Entity Recognizer)

## インストール方法
git-lfsの[インストール](https://github.com/git-lfs/git-lfs?utm_source=gitlfs_site&utm_medium=installation_link&utm_campaign=gitlfs#installing)が必要です。

```bash
git clone https://github.com/k141303/JENER.git
cd JENER
git lfs pull
pip install .
```

## 使い方

```bash
python3 -m jener "木村拓哉は日本の俳優であり、SMAPの元メンバーである。"
# 入力: 木村拓哉は日本の俳優であり、SMAPの元メンバーである。
# {'surf': '木村拓哉', 'span': (0, 4), 'ENEs': ['人名']}
# {'surf': '日本', 'span': (5, 7), 'ENEs': ['地名>ＧＰＥ>国名']}
# {'surf': '俳優', 'span': (8, 10), 'ENEs': ['プロダクト名>称号名>地位職業名']}
# {'surf': 'SMAP', 'span': (14, 18), 'ENEs': ['組織名>公演組織名']}
```
