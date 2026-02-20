```bash
uv init
```

Project struture: 
```bash
mkdir -p configs src/{common,data,features,training,inference,artifacts,api,cli} tests scripts infra/{iam,s3,ecr,ecs} notebooks envs && \
touch \
configs/{train.yaml,inference.yaml,aws.yaml} \
src/common/{logging.py,io.py,schemas.py} \
src/data/{load.py,validate.py} \
src/features/{preprocess.py,build.py} \
src/training/{pipeline.py,train.py,tune.py,evaluate.py} \
src/inference/{pipeline.py,predictor.py} \
src/artifacts/{save.py,load.py} \
src/api/{app.py,routes.py} \
src/cli/{train.py,predict.py,retrain.py} \
tests/{test_features.py,test_training.py,test_inference.py,test_api.py} \
scripts/{build_image.sh,push_ecr.sh,run_local.sh} \
envs/{local.env,staging.env,prod.env} \
notebooks/exploration.ipynb
```

```bash
git config --global user.name "asdfaf"
```

```bash
git config --global user.email "@gmail.com"
```
```bash
git restore --staged .
```
```bash
git add .
```

```bash
git commit -m "create project structure"
```

```bash
git remote -v
```

```bash
git remote set-url origin git@github.com:Psuuuuu/ml-aws.git
```

```bash
ssh -T git@github.com
```

```bash
git push
```

### Builders should not perfrom file I/O