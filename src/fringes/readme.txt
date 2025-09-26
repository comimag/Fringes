before publishing new version to pypi:

update docs

uv self update
uv python upgrade
uv python pin
uv version --bump patch/minor/major  # https://docs.astral.sh/uv/reference/cli/#uv-version
uv sync -U --all-groups
uv run black tests src examples
uv run pytest  # speed, speed, compile_time
uv run pytest src/ --doctest-modules
    uv run pytest --cov  # speed, speed, compile_time
    uv run pytest --cov-report html
    uv run coverage run -m pytest src/

git tag
git push (with tags)

GitHub new release (by tag)

read-the-docs check docs
read-the-docs activate version (not hidden)

uv build
uv publish

pip install fringes -U  # test
