import copy
import datetime
import time

import pandas as pd
import ray.exceptions
from pymongo import MongoClient

from scripts.eval import eval_runner

# > Mongo connection
mongo_url = "mongodb://10.200.205.226:27017"
stats_db_name = "deplan_analysis"
stats_collection = "ex_collection_generalization"
stats_single_collection = "ex_single_generalization_" + str(
    datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
)

client = MongoClient(mongo_url)
db = client[stats_db_name]
collection_all = db[stats_collection]


# (use max_calls=1 to release CUDA memory)
# reference: https://docs.ray.io/en/master/ray-core/tasks/using-ray-with-gpus.html
# (no retries)
@ray.remote(max_calls=1, max_retries=0)  # noqa
def eval_one(_args, _size, _model, _checkpoint, callback=None, data_size=1000):
    """ """
    print(f"> Run: size = {_size}, model = {_model}, checkpoint = {_checkpoint}")

    _args.datafile = (
        f"data/generalization-m{_size}_{_args.mechanism}_test-{data_size}.npz"
    )

    _path = f"./log/{_checkpoint}"
    _args.model = "models." + _model
    _args.model_path = _path

    # FIXME also set size in args, sometimes used!
    # TODO need to override!
    _args.maze_size = _size
    # _args.maze_size = 23  # test randomly set one

    if callback is not None:
        _args = callback(_args, _size, _model)

    _stats = eval_runner(_args, eval_on_test=True)
    _stats_save = {
        _k: _stats[_k]
        for _k in [
            "avg_error",
            "avg_optimal",
            "avg_success",
            "avg_SPL",
            "avg_planner_loss",
        ]
    }
    _stats_save["algorithm"] = _model
    _stats_save["size"] = _size
    _stats_save["checkpoint"] = _checkpoint

    # TODO add all model args to stats saving
    _stats_save.update(vars(_args))
    _stats_save["args"] = vars(_args)

    # > save each single run - open a mongo client in each remote function
    _client = MongoClient(mongo_url)
    _db = _client[stats_db_name]
    collection_single = _db[stats_single_collection]
    res_id = collection_single.insert_one(_stats_save)

    time.sleep(1)
    _stats_save["mongo_id"] = str(res_id.inserted_id)

    # > convert
    _stats_df_row = pd.DataFrame([_stats_save])

    print()
    return _stats_df_row


def save(_stats_list, note="generalization with fixed/var K"):
    # > retrieve
    time.sleep(3)
    stats_df = pd.concat(_stats_list)

    # > save to mongo
    result_id = collection_all.insert_one(
        {
            "time": datetime.datetime.now(),
            "time_str": str(datetime.datetime.now().strftime("%m-%d-%H-%M-%S")),
            # 'algorithms': model2checkpoint,
            "stats": stats_df.to_dict(orient="records"),
            "note": note,
        }
    )
    print(result_id)


def run_ray(
    args, model2checkpoint, map_sizes, run_models, model2resource, callback=None
):
    stats_id_list = []

    for _size in map_sizes:
        for _model, _checkpoints in model2checkpoint.items():
            if _model not in run_models:
                continue
            for _checkpoint in _checkpoints:
                print(
                    f"> Added to queue: size = {_size}, model = {_model}, checkpoint = {_checkpoint}"
                )

                _stats_row = eval_one.options(
                    num_cpus=2, num_gpus=model2resource[_model]
                ).remote(
                    _args=copy.deepcopy(args),
                    _size=_size,
                    _model=_model,
                    _checkpoint=_checkpoint,
                    callback=callback,
                )
                stats_id_list.append(_stats_row)

    # > start parallel running
    n_total = len(stats_id_list)
    ready_ids_all = []
    print(f"> #total: {n_total}")

    ready_ids, remaining_ids = ray.wait(stats_id_list, num_returns=1, timeout=3)
    ready_ids_all.extend(ready_ids)

    while remaining_ids:
        # (ray.wait needs to take unfinished ids)
        ready_ids, remaining_ids = ray.wait(remaining_ids, num_returns=1, timeout=3)

        # > handle error here
        try:
            ready_res = ray.get(ready_ids)
        except ray.exceptions.RayTaskError as e:
            print("\n>>>>>>>>>>>>>>>>> Failed run(s). Print error:")
            print(e, "\n")

        ready_ids_all.extend(ready_ids)

        n_ready = len(ready_ids_all)
        n_remaining = len(remaining_ids)

        print(
            f"#ready: {n_ready}, #remaining: {n_remaining}, #total: {n_total}, %ready: {n_ready / n_total: .3}"
        )

    return ready_ids


def retrieve_finished(ready_ids):
    """
    retrieve finished objects
    (probably not working as expected; not verfied)
    """

    try:
        stats_list = ray.get(ready_ids)
        save(stats_list)
    except ray.exceptions.RayTaskError:
        print("> Failed run! -> Go for loop")

        for _i, _id in enumerate(ready_ids):
            success_stats_list = []
            print(_i, _id)

            try:
                _stat = ray.get(_id)
                success_stats_list.append(_stat)
                save(success_stats_list)
            except ray.exceptions.RayTaskError:
                print("> Failed individual run; skip")
