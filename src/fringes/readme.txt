before publishing new version to pypi:
    docs

    poetry version ... (patch, minor, major, prepatch, preminor, premajor, prerelease)
    poetry update
    poetry lock
    poetry run black src
    poetry run pytest   # todo: coverage

    git tag
    git push (with tags)

    check docs on read-the-docs

    poetry build
    poetry publish
