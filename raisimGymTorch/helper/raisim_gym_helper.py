from shutil import copyfile
import datetime
import os
import ntpath
import torch


class ConfigurationSaver:
    def __init__(self, log_dir, save_items):
        self._data_dir = log_dir + '/' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(self._data_dir)

        if save_items is not None:
            for save_item in save_items:
                base_file_name = ntpath.basename(save_item)
                copyfile(save_item, self._data_dir + '/' + base_file_name)

    @property
    def data_dir(self):
        return self._data_dir


def tensorboard_launcher(directory_path):
    from tensorboard import program
    import webbrowser
    # learning visualizer
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', directory_path, '--port=24682', '--host=0.0.0.0'])
    url = tb.launch()
    print("[RAISIM_GYM] Tensorboard session created: "+url)
    # webbrowser.open_new(url)


def load_param(weight_path, env, actor, critic, policy_optimizer, value_optimizer, policy_scheduler, value_scheduler, data_dir):
    if weight_path == "":
        raise Exception("\nCan't find the pre-trained weight, please provide a pre-trained weight with --weight switch\n")
    print("\nRetraining from the checkpoint:", weight_path+"\n")

    iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
    weight_dir = weight_path.rsplit('/', 1)[0] + '/'

    mean_csv_path = weight_dir + 'mean' + iteration_number + '.csv'
    var_csv_path = weight_dir + 'var' + iteration_number + '.csv'
    value_mean_csv_path = weight_dir + 'value_obs_mean' + iteration_number + '.csv'
    value_var_csv_path = weight_dir + 'value_obs_var' + iteration_number + '.csv'
    items_to_save = [weight_path, mean_csv_path, var_csv_path, value_mean_csv_path, value_var_csv_path,
                     weight_dir + "cfg.yaml", weight_dir + "Environment.hpp", weight_dir + "RaiboController.hpp",
                     weight_dir + "runner.py", weight_dir + "module.py", weight_dir + "ppo.py"]

    if items_to_save is not None:
        pretrained_data_dir = data_dir + '/pretrained_' + weight_path.rsplit('/', 1)[0].rsplit('/', 1)[1]
        os.makedirs(pretrained_data_dir)
        for item_to_save in items_to_save:
            copyfile(item_to_save, pretrained_data_dir+'/'+item_to_save.rsplit('/', 1)[1])

    # load actor and critic parameters from full checkpoint
    checkpoint = torch.load(weight_path)
    actor.architecture.load_state_dict(checkpoint['actor_architecture_state_dict'])
    actor.distribution.load_state_dict(checkpoint['actor_distribution_state_dict'])
    actor.distribution.update()
    critic.architecture.load_state_dict(checkpoint['critic_architecture_state_dict'])
    policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
    value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
    policy_scheduler.load_state_dict(checkpoint['policy_scheduler_state_dict'])
    value_scheduler.load_state_dict(checkpoint['value_scheduler_state_dict'])
    return int(iteration_number)
