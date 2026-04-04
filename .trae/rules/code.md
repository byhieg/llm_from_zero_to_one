1. 生成的代码需要有方法和类注释。注释优先中文。符合顶级开源项目的注释规范。
2. 运行的话，可以选 source .venv/bin/activate 激活虚拟环境。然后使用 uv run python3 -m main xxx 的方式来运行。pytest 也是 一样的，都是在虚拟环境中使用 uv run python3 运行。
3. 修改完的代码，必须有 pytest 对应的测试。测试必须通过。
4. 修改完代码之后使用 ruff 格式化代码，包括 isort 调整import 顺序。
5. 测试通过之后，将改动的代码进行 git add,git commit。
