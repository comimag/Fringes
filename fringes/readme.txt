before publishing new version to pypi:
    poetry version
    poetry update
    poetry lock
    pytest
    citation (zenodo)
    black
    push to github
    check docs on read-the-docs
    poetry build
    poetry publish
