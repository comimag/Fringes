https://www.youtube.com/watch?v=b4iFyrLQQh4
https://www.youtube.com/watch?v=nZttMg_n_s0

in docs within package

configure conf.py
poetry run sphinx-quickstart
poetry run sphinx-apidoc -f -o . ../fringes /*decoder*
poetry run sphinx-build . _build
