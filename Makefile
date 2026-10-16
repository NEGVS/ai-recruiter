PY := python
.PHONY: init index demo test eval api clean

init:      ## 初始化数据库与演示数据
	$(PY) scripts/init_data.py

index:     ## 向量化入库（自动选后端）
	$(PY) scripts/index_data.py

demo:      ## 运行内置示例集
	$(PY) run_demo.py --demo

test:      ## 运行测试
	$(PY) tests/test_all.py

eval:      ## 运行评测
	$(PY) scripts/run_eval.py --json eval_report.json

api:       ## 启动 API 服务
	$(PY) -m uvicorn api:app --port 8000

clean:
	rm -rf data/recruit.db data/profiles_cache.json **/__pycache__
