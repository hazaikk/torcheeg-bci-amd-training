# GitHub 常用命令速查

> 本项目专用 — 适用于 `torcheeg-bci-amd-training`

---

## 📦 首次上传

```bash
# 在 cloud_amd_training_for_github/ 目录下操作
cd cloud_amd_training_for_github

git init
git add .
git commit -m "Initial: BCICIV2a TorchEEG training for AMD Cloud"
git branch -M main

# 创建远程仓库并推送
gh repo create torcheeg-bci-amd-training --public --source . --push

# 或使用已有远程仓库
git remote add origin https://github.com/hazaikk/torcheeg-bci-amd-training.git
git push -u origin main
```

---

## 🔄 日常更新

```bash
# 1) 在 cloud_amd_training_for_github/ 目录操作
cd cloud_amd_training_for_github

# 2) 重新生成最新代码 (从 cloud_amd_training 复制)
cd ../cloud_amd_training
python create_github_upload.py
cd ../cloud_amd_training_for_github

# 3) 提交并推送
git add .
git commit -m "Update: 训练参数优化 / EarlyStopping / etc"
git push origin main
```

---

## 🌿 分支管理

```bash
# 查看本地分支
git branch

# 查看远程分支
git branch -r

# 删除远程分支 (例如多余的 master)
git push origin --delete master

# 删除本地分支
git branch -d master

# 创建并切换到新分支
git checkout -b feature-xxx
```

---

## 📤 推送 / 覆盖

```bash
# 常规推送
git push origin main

# 强制覆盖 (本地完全覆盖远程)
git push origin main --force

# 设置上游分支 (首次推送后只需 git push)
git push -u origin main
```

---

## 📥 拉取 / 回退

```bash
# 拉取最新
git pull origin main

# 放弃本地修改, 强制与远程一致
git fetch origin
git reset --hard origin/main

# 查看提交记录
git log --oneline --graph --all

# 回退到上个版本 (保留工作区)
git reset --soft HEAD~1

# 回退到上个版本 (放弃工作区)
git reset --hard HEAD~1
```

---

## 🔍 状态 / 查看

```bash
# 查看工作区状态
git status

# 查看具体修改了什么
git diff

# 查看已暂存的改动
git diff --cached

# 查看提交历史
git log --oneline -10
```

---

## ⚙️ 配置

```bash
# 设置用户名邮箱 (首次使用)
git config --global user.name "Your Name"
git config --global user.email "your@email.com"

# 查看配置
git config --list

# 设置默认分支名为 main
git config --global init.defaultBranch main
```

---

## 🚀 AMD Cloud 部署

```
GitHub Repo URL: https://github.com/hazaikk/torcheeg-bci-amd-training
Notebook Path:   cloud_amd_training/BCICIV2a_TorchEEG_Training.ipynb
```
