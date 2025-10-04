clean:
	@find ./ -type d -name ".venv" -exec rm -rf {} +
	@find ./ -type f -name "uv.lock" -exec rm -f {} +