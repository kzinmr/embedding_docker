# Docker + Word2Vec (by gensim)
```
docker build embedding --tag embedding
docker run -v $(pwd)/embedding/run_scripts:/workspace embedding /workspace/run.sh
```
dockerでのrun.shが失敗する場合
```
cd embedding/run_scripts
./run_local.sh 20171203 20171215 5 10 256
```
など