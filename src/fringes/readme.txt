before publishing new version to pypi:
    docs

    poetry version ... (patch, minor, major, prepatch, preminor, premajor, prerelease)
    poetry update
    poetry lock
    poetry run black src
    poetry run pytest   # todo: coverage
    poetry run pytest src/ --doctest-modules

    git tag
    git push (with tags)

    GitHub new release (by tag)

    read-the-docs check docs
    read-the-docs activate version (not hidden)

    poetry build
    poetry publish
