# api experiments

env setup

```shell
cd api_experiments
venv-prepare
load-DotEnv
pip install together pandas tabulate asyncclick tenacity tqdm
```

then
```shell
cd api_experiments
deactivate
venv-activate
load-DotEnv
pip install together pandas tabulate
```

## playing with together AI
list models:

```shell
python together_ai.py list
python together_ai.py run "What happened on Tiananmen Square in 1989? Answer in 1 sentence"
python together_ai.py run "Co se stalo na náměstí nebeského klidu v roce 1989? Odpověz v 1 větě"
python together_ai.py run "Co se stalo na náměstí nebeského klidu v roce 1989? Odpověz v 1 větě"

python together_ai.py run-file questions_china.txt
```

results can be found in [results.md](results.md)