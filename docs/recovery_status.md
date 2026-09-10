# 仓库恢复与验证状态

日期：2026-09-11。

## 恢复基线

- 工作目录：`/Users/ethanjiang/Developer/InterviewOS`，不再使用已删除的 Documents/招聘。
- 来源：`https://github.com/eThAN02010506/InterviewOS.git`，恢复提交 `1563658`。
- 检查时 master 与 origin/master 一致；175 个受版本管理文件，无本地改动。
- `git fsck --full` 通过；新仓库 `.git` 未发现 `dataless` 占位文件。

## 环境与数据边界

- 在新目录重建 Python 虚拟环境与 npm 依赖。PyCharm 解释器改用 `.venv/bin/python`。
- Git 不包含 `.env`、数据库、录音、密钥和构建产物。代码恢复不等于历史会话恢复。
- 用户目录中的服务配置文件与 macOS 桌面数据库仍存在；本轮仅确认存在，未读取敏感内容，
  未复制、覆盖或合并这些数据，也未验证历史数据完整性。
- 浏览器服务默认使用工作目录的数据库；桌面应用默认使用系统应用数据目录。
  如需恢复旧浏览器会话，应从备份恢复数据库及录音，不应直接覆盖为桌面数据库。

## 验证范围

- 完整 pytest：604 passed，1 skipped（仅 Windows 的独占端口绑定行为），39.07 秒。
- 初次全量运行发现 13 项过时测试失败，已修复并保留原有安全断言：畸形 Unicode 使用
  显式 JSON 转义发送；服务状态在 lifespan 启动后读取；音频测试复用完整模拟客户端；
  首次配置版本与持久化版本均断言为 0。不改动生产接口逻辑或放宽密钥校验。
- Ruff 通过；mypy 检查 92 个源码文件通过；pip check 无依赖冲突。
- 所有 Web JS 和桌面脚本的 Node 语法检查通过；Rust 格式检查通过。
- `npm ci` 按 lockfile 安装成功，审计报告 0 vulnerabilities（不代表无任何安全问题）。
- 独立临时数据目录下启动真实 Uvicorn 服务，`/health`、`/`、`/app.js` 均返回 200；
  验证结束后已关闭临时服务，未启动常驻服务或触碰历史用户数据库。
- 保留一个 Starlette/AnyIO 的弃用警告，未屏蔽；原生桌面打包、麦克风和真实模型端到端
  交互本轮未验收。此前“完整测试收集受阻”的旧记录属于旧 iCloud 工作区，不是本次结果。
- 环境：Python 3.13.5；关键依赖 FastAPI 0.141.1、Starlette 1.6.0、Pydantic 2.13.5、
  pytest 9.1.1。本次按 pyproject 范围重装，Python 依赖尚未锁定所有精确版本。
