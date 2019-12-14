# Kaggle Days Tokyo 2019

## Quick Start

copy `docker-compose.yml` file

```bash
cp docker-compose.dev.yml docker-compose.yml
```

* Notes
  * change data dir on your local machine dir.
  * default value is `/media/disk1/dataset/atma_cup_03/`
  * add original data to `/<your-local-data-dir>/raw`

docker login to gitlab.com

```bash
docker login registry.gitlab.com
```

build and `docker-compose up -d`

```bash
docker-compose build
docker-compose up -d

# exec container 
docker exec -it kaggle-days_jupyter_1 bash
```

Good Luck 😆

### Train

run `src/train.py` 

```bash
usage: train.py [-h]
                [--molecule {benchmark,groupby_user,add_kiji_vec,target_encoding,add_time_diff}]
                [--simple]

run training

optional arguments:
  -h, --help            show this help message and exit
  --molecule {benchmark,groupby_user,add_kiji_vec,target_encoding,add_time_diff}
                        molecule name (see kaggle_days.molecules.py file)
                        (default: benchmark)
  --simple              If True, run on small models (LightGBMx3 different
                        objective function) (default: False)
```

arguments

* molecule
  * change feature set. 
* simple:
  * If add, run simple models (single lightGBM model x 3)