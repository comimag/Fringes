https://www.youtube.com/watch?v=b4iFyrLQQh4
https://www.youtube.com/watch?v=nZttMg_n_s0

cd docs

configure conf.py
uv run sphinx-quickstart
uv run sphinx-apidoc -f -o ./source/03_api ../src/fringes /*decoder*
# uv run sphinx-apidoc -f -o source/03_api ../fringes/fringes /*decoder*  # automatically done by the 'autodoc' and 'autosummary' extension
uv run sphinx-build source build
# uv run sphinx-build -M latexpdf source build  # https://tex.stackexchange.com/questions/461954/error-latexmk-the-script-engine-could-not-be-found-in-vs-code-using-miktex-2
