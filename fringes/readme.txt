before publishing new version to pypi:
    poetry version
    poetry update
    poetry lock
    pytest
    black
    citation (zenodo)
    git tag
    git push
    check docs on read-the-docs
    poetry build
    poetry publish
