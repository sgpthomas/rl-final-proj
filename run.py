#!/usr/bin/env python3

import argparse
import ray
import yaml
from pathlib import Path
from utils import remote, saving
import time
import os
import tf_models

from env_wrapper import RLlibEnvWrapper
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.a3c import A3CTrainer
from ray.rllib.agents.dqn import DQNTrainer
from ray.tune.logger import pretty_print

def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", type=str, help="Path to config for this run."
    )
    parser.add_argument(
        "--output", type=str, help="Output directory for this run."
    )
    parser.add_argument(
        "--debug", action="store_true"
    )

    args = parser.parse_args()

    # read config and parse
    config = args.config
    assert Path(config).exists()
    with open(config, "r") as f:
        run_configuration = yaml.safe_load(f)

    # make new directory
    run_directory = Path(args.output)

    return run_directory, run_configuration, args.debug


def build_trainer(run_configuration):
    """Finalize the trainer config"""
    trainer_config = run_configuration.get("trainer")

    # env
    env_config = {
        "env_config_dict": run_configuration.get("env"),
        "num_envs_per_worker": trainer_config.get("num_envs_per_worker")
    }

    # seed
    if trainer_config["seed"] is None:
        try:
            start_seed = int(run_configuration["metadata"]["launch_time"])
        except KeyError:
            start_seed = int(time.time())
    else:
        start_seed = int(trainer_config["seed"])

    final_seed = int(start_seed % (2 ** 16)) * 1000

    # multi-agent policies
    dummy_env = RLlibEnvWrapper(env_config)

    # policy tuples for agent/planner policy types
    agent_policy_tuple = (
        None,
        dummy_env.observation_space,
        dummy_env.action_space,
        run_configuration.get("agent_policy")
    )
    planner_policy_tuple = (
        None,
        dummy_env.observation_space_pl,
        dummy_env.action_space_pl,
        run_configuration.get("planner_policy"),
    )

    policies = {"a": agent_policy_tuple, "p": planner_policy_tuple}

    def policy_mapping_fun(i):
        if str(i).isdigit() or i == "a":
            return "a"
        return "p"

    # Which policies to train
    if run_configuration["general"]["train_planner"]:
        policies_to_train = ["a", "p"]
    else:
        policies_to_train = ["a"]

    # === Finalize and create ===
    trainer_config.update(
        {
            "env_config": env_config,
            "seed": final_seed,
            "multiagent": {
                "policies": policies,
                "policies_to_train": policies_to_train,
                "policy_mapping_fn": policy_mapping_fun,
            },
            "metrics_smoothing_episodes": trainer_config.get("num_workers")
            * trainer_config.get("num_envs_per_worker"),
        }
    )

    algo = run_configuration.get("algo", "ppo")
    if algo == "ppo":
        return PPOTrainer(
            env=RLlibEnvWrapper,
            config=trainer_config,
        )
    elif algo == "a3c":
        return A3CTrainer(
            env=RLlibEnvWrapper,
            config=trainer_config,
        )
    elif algo == "dqn":
        return DQNTrainer(
            env=RLlibEnvWrapper,
            config=trainer_config
        )


def set_up_dirs_and_maybe_restore(run_directory, run_configuration, trainer_obj):
    # === Set up Logging & Saving, or Restore ===
    # All model parameters are always specified in the settings YAML.
    # We do NOT overwrite / reload settings from the previous checkpoint dir.
    # 1. For new runs, the only object that will be loaded from the checkpoint dir
    #    are model weights.
    # 2. For crashed and restarted runs, load_snapshot will reload the full state of
    #    the Trainer(s), including metadata, optimizer, and models.
    (
        dense_log_directory,
        ckpt_directory,
        restore_from_crashed_run,
    ) = saving.fill_out_run_dir(run_directory)

    # If this is a starting from a crashed run, restore the last trainer snapshot
    if restore_from_crashed_run:
        print(
            "ckpt_dir already exists! Planning to restore using latest snapshot from "
            "earlier (crashed) run with the same ckpt_dir %s",
            ckpt_directory,
        )

        at_loads_a_ok = saving.load_snapshot(
            trainer_obj, run_directory, load_latest=True
        )

        # at this point, we need at least one good ckpt restored
        if not at_loads_a_ok:
            print(
                f"restore_from_crashed_run -> restore_run_dir {run_directory},but no good ckpts "
                "found/loaded!",
            )
            exit(-1)

        # === Trainer-specific counters ===
        training_step_last_ckpt = (
            int(trainer_obj._timesteps_total) if trainer_obj._timesteps_total else 0
        )
        epis_last_ckpt = (
            int(trainer_obj._episodes_total) if trainer_obj._episodes_total else 0
        )

    else:
        print("Not restoring trainer...")
        # === Trainer-specific counters ===
        training_step_last_ckpt = 0
        epis_last_ckpt = 0

        # For new runs, load only tf checkpoint weights
        starting_weights_path_agents = run_configuration["general"].get(
            "restore_tf_weights_agents", ""
        )
        if starting_weights_path_agents:
            print("Restoring agents TF weights...")
            saving.load_tf_model_weights(trainer_obj, starting_weights_path_agents)
        else:
            print("Starting with fresh agent TF weights.")

        starting_weights_path_planner = run_configuration["general"].get(
            "restore_tf_weights_planner", ""
        )
        if starting_weights_path_planner:
            print("Restoring planner TF weights...")
            saving.load_tf_model_weights(trainer_obj, starting_weights_path_planner)
        else:
            print("Starting with fresh planner TF weights.")

    return (
        dense_log_directory,
        ckpt_directory,
        restore_from_crashed_run,
        training_step_last_ckpt,
        epis_last_ckpt,
    )


def maybe_store_dense_log(
    trainer_obj, result_dict, dense_log_freq, dense_log_directory
):
    if result_dict["episodes_this_iter"] > 0 and dense_log_freq > 0:
        episodes_per_replica = (
            result_dict["episodes_total"] // result_dict["episodes_this_iter"]
        )
        if episodes_per_replica == 1 or (episodes_per_replica % dense_log_freq) == 0:
            log_dir = os.path.join(
                dense_log_directory,
                "logs_{:016d}".format(result_dict["timesteps_total"]),
            )
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
            saving.write_dense_logs(trainer_obj, log_dir)
            print(f">> Wrote dense logs to: {log_dir}")


def maybe_save(trainer_obj, result_dict, ckpt_freq, ckpt_directory, trainer_step_last_ckpt):
    global_step = result_dict["timesteps_total"]

    # Check if saving this iteration
    if (
        result_dict["episodes_this_iter"] > 0
    ):  # Don't save if midway through an episode.

        if ckpt_freq > 0:
            if global_step - trainer_step_last_ckpt >= ckpt_freq:
                saving.save_snapshot(trainer_obj, ckpt_directory, suffix="")
                saving.save_tf_model_weights(
                    trainer_obj, ckpt_directory, global_step, suffix="agent"
                )
                saving.save_tf_model_weights(
                    trainer_obj, ckpt_directory, global_step, suffix="planner"
                )

                trainer_step_last_ckpt = int(global_step)

                print(f"Checkpoint saved @ step {global_step}")

    return trainer_step_last_ckpt


def run_phase(run_dir, run_config, phase_n=0):

    print(f"run_dir={run_dir}")
    if run_dir.exists():
        print("Skipping phase because it already exists.")
        return

    run_dir.mkdir(parents=True)

    # create the trainer
    trainer = build_trainer(run_config)

    (
        dense_log_dir,
        ckpt_dir,
        restore_from_crashed_run,
        step_last_ckpt,
        num_parallel_episodes_done,
    ) = set_up_dirs_and_maybe_restore(run_dir, run_config, trainer)

    dense_log_frequency = run_config["env"].get("dense_log_frequency", 0)
    ckpt_frequency = run_config["general"].get("ckpt_frequency_steps", 0)
    global_step = int(step_last_ckpt)

    while num_parallel_episodes_done < run_config["general"]["episodes"]:

        # Training
        result = trainer.train()

        # === Counters++ ===
        num_parallel_episodes_done = result["episodes_total"]
        global_step = result["timesteps_total"]
        curr_iter = result["training_iteration"]

        print(
            "Iter {}: steps this-iter {} total {} -> {}/{} episodes done".format(
                curr_iter,
                result.get("timesteps_this_iter", None),
                global_step,
                num_parallel_episodes_done,
                run_config["general"]["episodes"],
            ))

        if curr_iter == 1 or result["episodes_this_iter"] > 0:
            print(pretty_print(result))

        # === Dense logging ===
        maybe_store_dense_log(trainer, result, dense_log_frequency, dense_log_dir)

        # === Saving ===
        step_last_ckpt = maybe_save(
            trainer, result, ckpt_frequency, ckpt_dir, step_last_ckpt
        )

    print("Completing! Saving final snapshot...\n\n")
    saving.save_snapshot(trainer, ckpt_dir)
    saving.save_tf_model_weights(trainer, ckpt_dir, global_step, suffix="agent")
    saving.save_tf_model_weights(trainer, ckpt_dir, global_step, suffix="planner")
    print(f"Phase {phase_n} Complete!")


def phase1(run_dir, run_config):
    """Train phase1 of the agent. This is without taxes."""
    run_dir = run_dir / "phase1"

    # disable taxes
    run_config["env"]["components"][-1]["PeriodicBracketTax"]["disable_taxes"] = True
    run_config["general"]["train_planner"] = False

    run_phase(run_dir, run_config, phase_n=1)


def phase2(run_dir, run_config):
    """
    Run phase2 of the training. Start from previous agent weights,
    and gradually enable government taxes.
    """
    phase2_run_dir = run_dir / "phase2"

    # enable taxes and planner training
    run_config["env"]["components"][-1]["PeriodicBracketTax"]["disable_taxes"] = False
    run_config["general"]["train_planner"] = True

    # find most recent agent weights file
    ckpts = (run_dir / "phase1" / "ckpts").glob("agent.tf.weights.global-step-*")
    ckpts_w_time = [(p, int(str(p).split("-")[-1])) for p in ckpts]
    sorted_ckpts = sorted(ckpts_w_time, key=lambda x: x[1])
    most_recent_ckpt = sorted_ckpts[-1][0]

    # error if not found
    if not most_recent_ckpt.exists():
        raise Exception(f"{most_recent_ckpt} doesn't exist.")

    # update config
    run_config["general"]["restore_tf_weights_agents"] = most_recent_ckpt

    run_phase(phase2_run_dir, run_config, phase_n=2)


def main():

    # read command line args
    run_dir, run_config, debug = process_args()

    if debug and run_dir.exists():
        import shutil
        shutil.rmtree(run_dir)

    # save the config used in this trial
    run_dir.mkdir(exist_ok=True)
    with (run_dir / "config.yaml").open("w") as f:
       yaml.dump(run_config, f) 

    # start whatever ray training server(?)
    ray.init(
        webui_host="0.0.0.0",  # make public
        log_to_driver=(not debug)
    )

    phase1(run_dir, run_config)
    phase2(run_dir, run_config)

    ray.shutdown()

if __name__ == "__main__":
    main()
