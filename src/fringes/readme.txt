before publishing new version to pypi:

update docs

uv self update
uv python upgrade
uv python pin
uv version --bump patch/minor/major  # https://docs.astral.sh/uv/reference/cli/#uv-version
uv sync -U --all-groups
uv tree --outdated --depth=1
uv run pytest tests src --doctest-modules  # speed, speed, compile_time
uv run coverage run -m pytest  # speed, speed, compile_time
uv run coverage report
uv run coverage html

git tag
git push (with tags)

GitHub new release (by tag)

read-the-docs check docs
read-the-docs activate version (not hidden)

uv build
uv publish

pip install fringes -U  # test
