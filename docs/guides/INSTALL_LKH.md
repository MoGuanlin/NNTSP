# LKH-3 安装指南 (LKH-3 Installation Guide)

LKH-3 是解决 TSP 问题的核心求解器。本指南将指导你在不同操作系统上安装 LKH-3。

## 1. Linux 系统 (推荐)

在 Linux 上，最稳健的方法是直接从源码编译。

### A. 安装依赖 (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install build-essential wget
```

### B. 下载与编译
```bash
# 下载最新版 3.0.13
wget http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.13.tgz
tar xvf LKH-3.0.13.tgz
cd LKH-3.0.13
make
```

### C. 加入系统路径 (可选)
编译完成后，将生成的 `LKH` 文件复制到 `/usr/local/bin`：
```bash
sudo cp LKH /usr/local/bin/
```

---

## 2. Windows 系统

### A. 直接下载执行文件
1. 访问官方下载地址：[LKH-3.0.13-WIN.zip](http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.13-WIN.zip)
2. 解压该 ZIP 包，你会发现一个 `LKH.exe`。

### B. 配置环境变量 (推荐)
1. 将 `LKH.exe` 放置在一个固定的文件夹（例如 `C:\Software\LKH\`）。
2. 将该路径添加到系统的 `Path` 环境变量中。
3. 打开命令行执行 `LKH`，若有反馈则说明成功。

---

## 3. macOS 系统

### A. 使用源码编译 (通法)
安装 Xcode 命令行工具后，流程与 Linux 一致：
```bash
# 确保安装了开发工具
xcode-select --install

# 后续流程参考 Linux 编译部分
```

---

## 4. 验证安装

在终端/命令行输入：
```bash
LKH
```

**正确反馈：**
```text
PARAMETER_FILE = 
PROBLEM_FILE = 
...
```
*(看到这些提示说明命令已识别，按 Ctrl+C 退出)*

---

## 5. 在本项目中调用

在项目 `Makefile` 中，默认的执行命令名为 `LKH`。
* 如果你已将 `LKH` 加入系统路径，直接运行：
  ```bash
  make eval_lkh
  ```
* 如果你未加入系统路径，需指定其位置：
  ```bash
  make eval_lkh LKH_EXE=/你的/存放路径/LKH
  ```

---
**官方主页 (Helsgaun 教授)**: [http://akira.ruc.dk/~keld/research/LKH-3/](http://akira.ruc.dk/~keld/research/LKH-3/)
