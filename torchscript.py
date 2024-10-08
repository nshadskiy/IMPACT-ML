import os
import optparse
import logging
import yaml
import json
import torch

import Data
from models.Models import NNModel
from NeuralNets import Network
import helper.plotting as plot
import configs.featureSets as featureSets
import configs.classSets as classSets

parser = optparse.OptionParser()

parser.add_option(
    "-i",
    "--inputdata",
    dest="inputdata",
    help="Absolute path to the preselected input data",
    metavar="/ceph/USER/SOME/PATH/FOLDER",
)

parser.add_option(
    "-c",
    "--config-file",
    dest="config_file",
    help="Path to the network configuration file",
    metavar="configs/FILE.yaml",
)

parser.add_option("-y", "--era", dest="era", help="Data-taking era", metavar="2018")

parser.add_option(
    "-k", "--channel", dest="channel", help="Analysis channel", metavar="mt"
)

parser.add_option(
    "-s",
    "--event-split",
    dest="event_split",
    default="even",
    help="Split of data into training and testing based on the event ID, to options are possible: even or odd; default: even",
    metavar="even",
)

parser.add_option(
    "-o",
    "--outputdir",
    dest="savedir",
    default="test_training",
    help="Name of the directory where all the output files will be stored (relative path inside workdir/); default: test_training",
    metavar="test_training",
)

parser.add_option(
    "-e",
    "--epochs",
    dest="epochs",
    default=500,
    help="Integer number of training epochs; default: 500",
    metavar="500",
)

(options, args) = parser.parse_args()


log = logging.getLogger("training")
log.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
# add the handlers to logger
log.addHandler(ch)

with open(options.config_file, "r") as file:
    config = yaml.load(file, yaml.FullLoader)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log.info("-" * 50)
log.info("CUDA is available: " + str(torch.cuda.is_available()))
log.info("GPU Device: " + str(torch.cuda.current_device()))
log.info("GPU: " + str(torch.cuda.get_device_name(torch.cuda.current_device())))


data = Data.Data(
    feature_list=featureSets.variables[config["features"]],
    class_dict=classSets.classes[config["classes"]],
    config=config,
    event_split=options.event_split,
)

data.load_data(sample_path=options.inputdata, era=options.era, channel=options.channel, shuffle_seed=None, val_fraction=0.25)

data.transform(type="standard", index_parametrization=config["index_parametrization"])
data.shuffling(seed=None)
# data.split_data(val_fraction=0.25)
data.prepare_for_training()

model = NNModel(
    n_input_features=len(data.features + data.param_features),
    n_output_nodes=len(data.classes),
    hidden_layer=config["hidden_layers"],
    dropout_p=config["dropout_p"],
)

workdir = os.getcwd()

savedir = options.savedir

if not os.path.exists(workdir + "/workdir/" + savedir):
    os.makedirs(workdir + "/workdir/" + savedir)
    
with open(workdir + "/workdir/" + savedir + f"/{options.channel}_feature_transformation_{options.event_split}.json", "w") as file_transform:
    json.dump(data.transform_feature_dict, file_transform, indent=4)
if config["index_parametrization"]:
    with open(workdir + "/workdir/" + savedir + f"/{options.channel}_mass_transformation.json", "w") as file_transform:
        json.dump(data.mass_indizes, file_transform, indent=4)
else:
    with open(workdir + "/workdir/" + savedir + f"/{options.channel}_mass_transformation_{options.event_split}.json", "w") as file_transform:
        json.dump(data.transform_param_feature_dict, file_transform, indent=4)

for comb in data.plot_mass_combinations:
    if not os.path.exists(
        workdir
        + "/workdir/"
        + savedir
        + f"/massX_{comb['massX']}_massY_{comb['massY']}"
    ):
        os.makedirs(
            workdir
            + "/workdir/"
            + savedir
            + f"/massX_{comb['massX']}_massY_{comb['massY']}"
        )

net = Network(
    model=model,
    data=data,
    config=config,
    device_to_run=device,
    save_path=workdir + "/workdir/" + savedir,
)

net.train(epochs=int(options.epochs))
# net.predict()
net.predict_for_mass_points()

plot.loss(net)
plot.confusion(net)
plot.multiclass_nodes(net)
plot.multiclass_classes(net)
