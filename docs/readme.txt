https://www.youtube.com/watch?v=b4iFyrLQQh4
https://www.youtube.com/watch?v=nZttMg_n_s0

cd docs

configure conf.py
poetry run sphinx-quickstart
# poetry run sphinx-apidoc -f -o ./api_reference ../fringes /*decoder*
poetry run sphinx-apidoc -f -o ./source/03_api ../src/fringes /*decoder*
# poetry run sphinx-apidoc -f -o source/03_api ../fringes/fringes /*decoder*  # automatically done by the 'autodoc' and 'autosummary' extension
poetry run sphinx-build source build
# poetry run sphinx-build -M latexpdf source build
